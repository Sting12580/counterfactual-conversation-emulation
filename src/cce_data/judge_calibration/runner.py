from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cce_data.judge_calibration.metrics import compare_models_metrics, group_metrics, regression_metrics
from cce_data.judge_calibration.reporting import write_report
from cce_data.judge_calibration.splits import (
    SplitAssignment,
    grouped_random_split,
    leave_one_column_value_out_splits,
    leave_one_dataset_out_splits,
    write_split_manifest,
)
from cce_data.judge_calibration.weighted_ensemble import (
    fit_best_single_judge,
    predict_median,
    predict_unweighted_mean,
    select_ridge_stacking_model,
    select_simplex_model,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) else value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _parse_lambda_grid(values: list[float] | str) -> list[float]:
    if isinstance(values, str):
        return [float(item) for item in values.split(",") if item.strip()]
    return [float(value) for value in values]


def _dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "datasets": sorted(df["dataset_id"].astype(str).unique().tolist()) if "dataset_id" in df else [],
        "score_dimensions": sorted(df["score_dimension"].astype(str).unique().tolist())
        if "score_dimension" in df
        else [],
        "answer_sources": sorted(df["answer_source"].astype(str).unique().tolist())
        if "answer_source" in df
        else [],
    }


def _sample_weights(df: pd.DataFrame, group_balance: bool, group_balance_col: str | None) -> np.ndarray | None:
    if not group_balance:
        return None
    if group_balance_col and group_balance_col in df.columns:
        groups = df[group_balance_col].astype(str)
    elif {"dataset_id", "score_dimension"}.issubset(df.columns):
        groups = df["dataset_id"].astype(str) + "::" + df["score_dimension"].astype(str)
    else:
        groups = pd.Series(["all"] * len(df), index=df.index)
    counts = groups.value_counts()
    weights = groups.map(lambda group: 1.0 / counts[group]).to_numpy(dtype=float)
    return weights / weights.sum()


def _worst_groups(group_tables: dict[str, pd.DataFrame], top_k: int = 10) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, table in group_tables.items():
        if table.empty:
            continue
        for _, row in table.iterrows():
            payload = row.to_dict()
            payload["group_table"] = group_name
            payload["abs_bias"] = abs(float(payload.get("bias", 0.0)))
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_cols = [column for column in ["rmse", "abs_bias"] if column in df.columns]
    return df.sort_values(sort_cols, ascending=False).head(top_k)


def _leave_one_judge_sensitivity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    judge_columns: list[str],
    judge_prefix: str,
    lambda_values: list[float],
    fit_options: list[bool],
    group_balance: bool,
    group_balance_col: str | None,
    clip_predictions: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if len(judge_columns) <= 1:
        return pd.DataFrame(rows)
    for dropped in judge_columns:
        kept = [column for column in judge_columns if column != dropped]
        judge_names = [column.removeprefix(judge_prefix) for column in kept]
        sample_weight = _sample_weights(train_df, group_balance, group_balance_col)
        model, diagnostics = select_simplex_model(
            train_df[kept].to_numpy(dtype=float),
            train_df["human_score"].to_numpy(dtype=float),
            val_df[kept].to_numpy(dtype=float),
            val_df["human_score"].to_numpy(dtype=float),
            lambda_grid=lambda_values,
            fit_intercept_options=fit_options,
            sample_weight_train=sample_weight,
            judge_names=judge_names,
            clip_predictions=clip_predictions,
        )
        pred = model.predict(test_df[kept].to_numpy(dtype=float))
        metrics = regression_metrics(test_df["human_score"].to_numpy(dtype=float), pred)
        rows.append(
            {
                "dropped_judge": dropped.removeprefix(judge_prefix),
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
                "test_bias": metrics["mean_bias"],
                "selected_lambda": diagnostics.get("selected_lambda"),
                "selected_fit_intercept": diagnostics.get("selected_fit_intercept"),
                "max_weight": diagnostics.get("max_weight"),
                "effective_num_judges": diagnostics.get("effective_num_judges"),
            }
        )
    return pd.DataFrame(rows)


def _split_counts(split: SplitAssignment) -> dict[str, int]:
    return {str(key): int(value) for key, value in split.assignments.value_counts().to_dict().items()}


def _make_splits(
    df: pd.DataFrame,
    split_protocol: str,
    seed: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> list[SplitAssignment]:
    if split_protocol == "grouped_random":
        return [
            grouped_random_split(
                df,
                group_col="split_group",
                train_frac=train_frac,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
            )
        ]
    if split_protocol == "leave_one_dataset_out":
        return list(leave_one_dataset_out_splits(df, val_frac=val_frac, seed=seed))
    if split_protocol == "leave_one_score_dimension_out":
        return list(
            leave_one_column_value_out_splits(
                df, column="score_dimension", group_col="split_group", val_frac=val_frac, seed=seed
            )
        )
    if split_protocol == "leave_one_answer_source_out":
        return list(
            leave_one_column_value_out_splits(
                df, column="answer_source", group_col="split_group", val_frac=val_frac, seed=seed
            )
        )
    raise ValueError(f"Unsupported split protocol: {split_protocol}")


def run_weighted_judge_combination(
    score_matrix: str | Path,
    out_dir: str | Path,
    split_protocol: str = "grouped_random",
    seed: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    lambda_grid: list[float] | str = "0,0.0001,0.001,0.01,0.1,1.0",
    fit_intercept_options: list[bool] | None = None,
    group_balance: bool = False,
    group_balance_col: str | None = None,
    clip_predictions: bool = True,
    judge_prefix: str = "judge::",
    min_complete_judge_coverage: float | None = None,
) -> dict[str, Any]:
    del min_complete_judge_coverage
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(score_matrix)
    judge_columns = [column for column in df.columns if column.startswith(judge_prefix)]
    if not judge_columns:
        raise ValueError(f"No judge columns found with prefix {judge_prefix!r}")
    rows_before = len(df)
    df = df.dropna(subset=judge_columns).reset_index(drop=True)
    rows_dropped = rows_before - len(df)
    if df.empty:
        raise ValueError("No complete rows remain after dropping missing judge scores")
    if "split_group" not in df.columns:
        df["split_group"] = df["dataset_id"].astype(str) + "::" + df["question_id"].astype(str) + "::" + df[
            "score_dimension"
        ].astype(str)

    lambda_values = _parse_lambda_grid(lambda_grid)
    fit_options = fit_intercept_options if fit_intercept_options is not None else [True, False]
    splits = _make_splits(df, split_protocol, seed, train_frac, val_frac, test_frac)
    aggregate_rows: list[dict[str, Any]] = []
    judge_correlation = df[judge_columns].corr()
    judge_correlation.to_csv(output_root / "judge_correlation.csv")

    for split in splits:
        split_dir = output_root if len(splits) == 1 else output_root / split.name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_df = df.copy()
        split_df["split"] = split.assignments.values
        train_df = split_df[split_df["split"] == "train"]
        val_df = split_df[split_df["split"] == "val"]
        test_df = split_df[split_df["split"] == "test"]
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"Split {split.name} has an empty train/val/test partition")

        X_train = train_df[judge_columns].to_numpy(dtype=float)
        y_train = train_df["human_score"].to_numpy(dtype=float)
        X_val = val_df[judge_columns].to_numpy(dtype=float)
        y_val = val_df["human_score"].to_numpy(dtype=float)
        X_test = test_df[judge_columns].to_numpy(dtype=float)
        sample_weight = _sample_weights(train_df, group_balance, group_balance_col)
        judge_names = [column.removeprefix(judge_prefix) for column in judge_columns]

        simplex_model, simplex_diag = select_simplex_model(
            X_train,
            y_train,
            X_val,
            y_val,
            lambda_grid=lambda_values,
            fit_intercept_options=fit_options,
            sample_weight_train=sample_weight,
            judge_names=judge_names,
            clip_predictions=clip_predictions,
        )
        best_single = fit_best_single_judge(X_train, y_train, X_val, y_val, judge_names=judge_names)
        ridge_model, ridge_diag = select_ridge_stacking_model(
            X_train, y_train, X_val, y_val, lambda_grid=lambda_values, fit_intercept=True
        )

        predictions = test_df.copy()
        predictions["pred_synthetic_judge"] = simplex_model.predict(X_test)
        predictions["pred_unweighted_mean"] = np.clip(predict_unweighted_mean(X_test), 0.0, 1.0)
        predictions["pred_median"] = np.clip(predict_median(X_test), 0.0, 1.0)
        predictions["pred_best_single_judge"] = np.clip(best_single.predict(X_test), 0.0, 1.0)
        predictions["pred_ridge_stacking"] = np.clip(ridge_model.predict(X_test), 0.0, 1.0)
        pred_cols = [
            "pred_synthetic_judge",
            "pred_unweighted_mean",
            "pred_median",
            "pred_best_single_judge",
            "pred_ridge_stacking",
        ]
        metrics_overall = compare_models_metrics(predictions, "human_score", pred_cols)
        group_tables: dict[str, pd.DataFrame] = {}
        for group_name, columns in {
            "dataset": ["dataset_id"],
            "score_dimension": ["score_dimension"],
            "answer_source": ["answer_source"],
            "dataset_score_dimension": ["dataset_id", "score_dimension"],
        }.items():
            if all(column in predictions.columns for column in columns):
                group_tables[group_name] = group_metrics(
                    predictions, "human_score", "pred_synthetic_judge", columns
                )
        worst_group_table = _worst_groups(group_tables)
        leave_one_judge = _leave_one_judge_sensitivity(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            judge_columns=judge_columns,
            judge_prefix=judge_prefix,
            lambda_values=lambda_values,
            fit_options=fit_options,
            group_balance=group_balance,
            group_balance_col=group_balance_col,
            clip_predictions=clip_predictions,
        )

        weights_payload = {
            **simplex_diag,
            "judge_columns": judge_columns,
            "best_single_judge": best_single.to_dict(),
            "ridge_stacking": ridge_diag,
            "group_balance": group_balance,
            "group_balance_col": group_balance_col,
        }
        predictions.to_csv(split_dir / "predictions.csv", index=False)
        metrics_overall.to_json(split_dir / "metrics_overall.json", orient="records", indent=2)
        for group_name, table in group_tables.items():
            table.to_csv(split_dir / f"metrics_by_{group_name}.csv", index=False)
        worst_group_table.to_csv(split_dir / "worst_groups.csv", index=False)
        leave_one_judge.to_csv(split_dir / "leave_one_judge_sensitivity.csv", index=False)
        (split_dir / "weights.json").write_text(
            json.dumps(weights_payload, indent=2, default=_json_default), encoding="utf-8"
        )
        write_split_manifest(split, split_dir / "split_manifest.json")

        split_summary = {
            "protocol": split_protocol,
            "name": split.name,
            "counts": _split_counts(split),
            "held_out_column": split.held_out_column,
            "held_out_value": split.held_out_value,
        }
        write_report(
            split_dir / "report.md",
            dataset_summary=_dataset_summary(df),
            coverage_summary={
                "judge_columns": judge_columns,
                "rows_dropped_missing_judges": rows_dropped,
            },
            split_summary=split_summary,
            weights=weights_payload,
            metrics_overall=metrics_overall,
            group_tables=group_tables,
            diagnostics_tables={
                "Worst Groups": worst_group_table,
                "Leave One Judge Sensitivity": leave_one_judge,
            },
            limitations=[
                "Hyperparameters are selected on validation rows only.",
                "Rows with missing donor judge scores are dropped for this first implementation.",
            ],
        )
        synthetic_metrics = metrics_overall[metrics_overall["model"] == "pred_synthetic_judge"].iloc[0]
        aggregate_rows.append(
            {
                "split_name": split.name,
                "held_out_column": split.held_out_column,
                "held_out_value": split.held_out_value,
                "test_n": int(synthetic_metrics["n"]),
                "synthetic_rmse": float(synthetic_metrics["rmse"]),
                "synthetic_mae": float(synthetic_metrics["mae"]),
                "synthetic_bias": float(synthetic_metrics["mean_bias"]),
                "selected_lambda": weights_payload.get("selected_lambda"),
                "selected_fit_intercept": weights_payload.get("selected_fit_intercept"),
                "max_weight": weights_payload.get("max_weight"),
                "effective_num_judges": weights_payload.get("effective_num_judges"),
            }
        )

    summary = {
        "score_matrix": str(score_matrix),
        "out_dir": str(output_root),
        "split_protocol": split_protocol,
        "n_splits": len(splits),
        "rows_before_drop_missing_judges": rows_before,
        "rows_after_drop_missing_judges": int(len(df)),
        "rows_dropped_missing_judges": rows_dropped,
    }
    if len(aggregate_rows) > 1:
        aggregate_df = pd.DataFrame(aggregate_rows)
        aggregate_df.to_csv(output_root / "aggregate_summary.csv", index=False)
    (output_root / "run_summary.json").write_text(
        json.dumps({**summary, "splits": aggregate_rows}, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary
