from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cce_data.judge_calibration.metrics import regression_metrics
from cce_data.judge_calibration.splits import grouped_random_split
from cce_data.judge_calibration.weighted_ensemble import select_simplex_model


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


def parse_float_grid(values: list[float] | str) -> list[float]:
    if isinstance(values, str):
        return [float(item) for item in values.split(",") if item.strip()]
    return [float(value) for value in values]


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


def _complete_score_matrix(df: pd.DataFrame, judge_prefix: str) -> tuple[pd.DataFrame, list[str]]:
    judge_columns = [column for column in df.columns if column.startswith(judge_prefix)]
    if not judge_columns:
        raise ValueError(f"No judge columns found with prefix {judge_prefix!r}")
    complete = df.dropna(subset=judge_columns).reset_index(drop=True)
    if complete.empty:
        raise ValueError("No complete rows remain after dropping missing judge scores")
    if "split_group" not in complete.columns:
        complete["split_group"] = (
            complete["dataset_id"].astype(str)
            + "::"
            + complete["question_id"].astype(str)
            + "::"
            + complete["score_dimension"].astype(str)
        )
    return complete, judge_columns


def bootstrap_weight_stability(
    score_matrix: str | Path,
    n_bootstrap: int = 100,
    seed: int = 0,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    lambda_grid: list[float] | str = "0,0.0001,0.001,0.01,0.1,1.0",
    fit_intercept_options: list[bool] | None = None,
    group_col: str = "split_group",
    group_balance: bool = False,
    group_balance_col: str | None = None,
    clip_predictions: bool = True,
    judge_prefix: str = "judge::",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    df, judge_columns = _complete_score_matrix(pd.read_csv(score_matrix), judge_prefix=judge_prefix)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}")

    split = grouped_random_split(
        df,
        group_col=group_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    split_df = df.copy()
    split_df["split"] = split.assignments.values
    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "val"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Base grouped split has an empty train/val/test partition")

    train_groups = sorted(train_df[group_col].astype(str).unique().tolist())
    train_group_frames = {group: rows for group, rows in train_df.groupby(train_df[group_col].astype(str))}
    rng = np.random.default_rng(seed)
    lambda_values = parse_float_grid(lambda_grid)
    fit_options = fit_intercept_options if fit_intercept_options is not None else [True, False]
    judge_names = [column.removeprefix(judge_prefix) for column in judge_columns]
    val_scores = val_df[judge_columns].to_numpy(dtype=float)
    val_labels = val_df["human_score"].to_numpy(dtype=float)
    test_scores = test_df[judge_columns].to_numpy(dtype=float)
    test_labels = test_df["human_score"].to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for replicate in range(n_bootstrap):
        sampled_groups = rng.choice(train_groups, size=len(train_groups), replace=True)
        boot_train = pd.concat([train_group_frames[str(group)] for group in sampled_groups], ignore_index=True)
        sample_weight = _sample_weights(boot_train, group_balance, group_balance_col)
        model, diagnostics = select_simplex_model(
            boot_train[judge_columns].to_numpy(dtype=float),
            boot_train["human_score"].to_numpy(dtype=float),
            val_scores,
            val_labels,
            lambda_grid=lambda_values,
            fit_intercept_options=fit_options,
            sample_weight_train=sample_weight,
            judge_names=judge_names,
            clip_predictions=clip_predictions,
        )
        test_metrics = regression_metrics(test_labels, model.predict(test_scores))
        row: dict[str, Any] = {
            "replicate": replicate,
            "selected_lambda": diagnostics["selected_lambda"],
            "selected_fit_intercept": diagnostics["selected_fit_intercept"],
            "validation_rmse": diagnostics["validation_rmse"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_bias": test_metrics["mean_bias"],
            "intercept": diagnostics["intercept"],
            "max_weight": diagnostics["max_weight"],
            "effective_num_judges": diagnostics["effective_num_judges"],
        }
        for judge_name, weight in diagnostics["weights"].items():
            row[f"weight::{judge_name}"] = weight
        rows.append(row)

    samples = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for column in [column for column in samples.columns if column.startswith("weight::")]:
        values = samples[column].to_numpy(dtype=float)
        summary_rows.append(
            {
                "judge_id": column.removeprefix("weight::"),
                "weight_mean": float(np.mean(values)),
                "weight_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "weight_p025": float(np.quantile(values, 0.025)),
                "weight_p50": float(np.quantile(values, 0.5)),
                "weight_p975": float(np.quantile(values, 0.975)),
                "zero_weight_rate": float(np.mean(np.isclose(values, 0.0))),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("weight_mean", ascending=False).reset_index(drop=True)
    config = {
        "score_matrix": str(score_matrix),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "lambda_grid": lambda_values,
        "fit_intercept_options": fit_options,
        "group_col": group_col,
        "group_balance": group_balance,
        "group_balance_col": group_balance_col,
        "judge_columns": judge_columns,
        "base_split_counts": {str(key): int(value) for key, value in split.assignments.value_counts().to_dict().items()},
    }
    return samples, summary, config


def write_bootstrap_weight_stability(
    score_matrix: str | Path,
    out_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    output_root = Path(out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    samples, summary, config = bootstrap_weight_stability(score_matrix, **kwargs)
    samples.to_csv(output_root / "bootstrap_weight_samples.csv", index=False)
    summary.to_csv(output_root / "bootstrap_weight_summary.csv", index=False)
    (output_root / "bootstrap_config.json").write_text(
        json.dumps(config, indent=2, default=_json_default), encoding="utf-8"
    )
    return {
        "out_dir": str(output_root),
        "n_bootstrap": int(config["n_bootstrap"]),
        "n_judges": len(config["judge_columns"]),
    }
