"""Summarize expert best-embedding diagnostics into a readable report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_INPUTS = {
    "support": "support_summary.csv",
    "density_sweep": "density_sweep.csv",
    "density_baseline": "density_baseline_summary.csv",
    "feature_ablation": "feature_ablation.csv",
    "surface_shift": "surface_feature_shift.csv",
}


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    tables = load_inputs(input_dir)
    report = build_report(tables)
    write_report(report, input_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/expert_best_embedding_diagnostics"),
    )
    return parser.parse_args()


def load_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in REQUIRED_INPUTS.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required diagnostic input: {path}")
        tables[key] = pd.read_csv(path)
    return tables


def build_report(tables: dict[str, pd.DataFrame]) -> dict:
    support = tables["support"].copy()
    baseline = tables["density_baseline"].copy()
    density = tables["density_sweep"].copy()
    features = tables["feature_ablation"].copy()
    shifts = tables["surface_shift"].copy()

    support_rows = build_support_rows(support)
    baseline_rows = build_baseline_density_rows(baseline)
    best_density_rows = build_best_density_rows(density)
    feature_delta_rows = build_feature_delta_rows(features)
    recommendations = build_recommendations(
        support=support,
        baseline=baseline,
        density=density,
        feature_deltas=pd.DataFrame(feature_delta_rows),
        shifts=shifts,
    )

    executive = build_executive_summary(recommendations)
    markdown = render_markdown(
        executive=executive,
        support_rows=support_rows,
        baseline_rows=baseline_rows,
        best_density_rows=best_density_rows,
        feature_delta_rows=feature_delta_rows,
        recommendations=recommendations,
    )

    return {
        "metadata": {
            "input_files": REQUIRED_INPUTS,
            "output_scope": "outputs/expert_best_embedding_diagnostics",
            "uses_agent_rewards_for_training_or_tuning": False,
            "agent_rewards_used_for": "diagnostic bias/ranking columns only",
            "ci_coverage_status": "not_fixed_or_claimed",
            "rules": [
                "support/positivity: severe support plus unsafe clipping/overlap evidence",
                "missing reward-relevant features: embedding_plus_surface MIPS abs-bias improves >=25% without ESS concern",
                "density-ratio regularization/calibration: safe density row improves MIPS abs-bias >=25% with ESS_fraction>=0.20 and clip_rate<=0.20",
                "otherwise mixed/unresolved",
            ],
        },
        "executive_summary": executive,
        "support_rows": support_rows,
        "baseline_density_rows": baseline_rows,
        "best_density_rows": best_density_rows,
        "feature_delta_rows": feature_delta_rows,
        "recommendations": recommendations,
        "markdown": markdown,
    }


def build_support_rows(support: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in support.sort_values("dataset").iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "n": int(row["n"]),
                "support_severity": normalize_severity(row["support_severity"]),
                "required_topk_fraction": as_float(row["required_topk_fraction"]),
                "knn10_gap_to_agent_mean": as_float(row["knn10_gap_to_agent_mean"]),
                "frac_agent_reward_above_knn10_max": as_float(
                    row["frac_agent_reward_above_knn10_max"]
                ),
                "nn_distance_min_p90": as_float(row["nn_distance_min_p90"]),
            }
        )
    return rows


def build_baseline_density_rows(baseline: pd.DataFrame) -> list[dict]:
    rows = []
    for _, row in baseline.sort_values("dataset").iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "v_snips": as_float(row["v_snips"]),
                "bias_snips": as_float(row["bias_snips"]),
                "abs_bias_snips": as_float(row["abs_bias_snips"]),
                "ess_fraction": as_float(row["ess_fraction"]),
                "clip_rate": as_float(row["clip_rate"]),
                "top10_weight_mass": as_float(row["top10_weight_mass"]),
                "corr_w_y_behavior": as_float(row["corr_w_y_behavior"]),
            }
        )
    return rows


def build_best_density_rows(density: pd.DataFrame) -> list[dict]:
    rows = []
    for dataset, group in density.groupby("dataset", sort=True):
        best = group.sort_values(
            ["abs_bias_snips", "ess_fraction", "clip_rate"],
            ascending=[True, False, True],
        ).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "C": as_float(best["C"]),
                "clip": clip_text(best["clip_label"]),
                "calibrate": bool(best["calibrate"]),
                "v_snips": as_float(best["v_snips"]),
                "abs_bias_snips": as_float(best["abs_bias_snips"]),
                "ess_fraction": as_float(best["ess_fraction"]),
                "clip_rate": as_float(best["clip_rate"]),
                "safe_by_report_rule": bool(
                    best["ess_fraction"] >= 0.20 and best["clip_rate"] <= 0.20
                ),
            }
        )
    return rows


def build_feature_delta_rows(features: pd.DataFrame) -> list[dict]:
    mips = features[features["estimator"] == "MIPS"].copy()
    rows = []
    for dataset, group in mips.groupby("dataset", sort=True):
        by_mode = group.set_index("mode")
        if "embedding_only" not in by_mode.index:
            raise ValueError(f"Missing embedding_only MIPS row for {dataset}")
        base_abs = as_float(by_mode.loc["embedding_only", "abs_bias"])
        base_v = as_float(by_mode.loc["embedding_only", "v_hat"])
        for mode in ("surface_only", "embedding_plus_surface"):
            if mode not in by_mode.index:
                continue
            row = by_mode.loc[mode]
            abs_bias = as_float(row["abs_bias"])
            rows.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "v_hat": as_float(row["v_hat"]),
                    "embedding_only_v_hat": base_v,
                    "abs_bias": abs_bias,
                    "embedding_only_abs_bias": base_abs,
                    "abs_bias_delta_vs_embedding_only": float(abs_bias - base_abs),
                    "abs_bias_reduction_fraction": reduction_fraction(base_abs, abs_bias),
                    "feature_dim_behavior": int(row["feature_dim_behavior"]),
                    "extra_dim": int(row["extra_dim"]),
                }
            )
    return rows


def build_recommendations(
    support: pd.DataFrame,
    baseline: pd.DataFrame,
    density: pd.DataFrame,
    feature_deltas: pd.DataFrame,
    shifts: pd.DataFrame,
) -> list[dict]:
    support_idx = support.set_index("dataset")
    baseline_idx = baseline.set_index("dataset")
    recommendations = []
    for dataset in sorted(support_idx.index):
        s = support_idx.loc[dataset]
        b = baseline_idx.loc[dataset]
        d = density[density["dataset"] == dataset]
        fd = feature_deltas[feature_deltas["dataset"] == dataset]

        density_eval = evaluate_density_rule(b, d)
        feature_eval = evaluate_feature_rule(b, fd)
        support_eval = evaluate_support_rule(s, b, d, density_eval, feature_eval)
        top_shift = top_surface_shift(shifts, dataset)

        if support_eval["triggered"]:
            primary_issue = "support/positivity"
        elif feature_eval["triggered"]:
            primary_issue = "missing reward-relevant features"
        elif density_eval["triggered"]:
            primary_issue = "density-ratio regularization/calibration"
        else:
            primary_issue = "mixed/unresolved"

        recommendations.append(
            {
                "dataset": dataset,
                "primary_issue": primary_issue,
                "support_severity": normalize_severity(s["support_severity"]),
                "baseline_abs_bias_snips": as_float(b["abs_bias_snips"]),
                "baseline_ess_fraction": as_float(b["ess_fraction"]),
                "best_safe_density_abs_bias": density_eval["best_safe_abs_bias"],
                "best_safe_density_reduction_fraction": density_eval[
                    "best_safe_reduction_fraction"
                ],
                "embedding_plus_surface_reduction_fraction": feature_eval[
                    "embedding_plus_surface_reduction_fraction"
                ],
                "top_surface_shift": top_shift,
                "evidence": make_evidence_text(support_eval, density_eval, feature_eval, top_shift),
                "recommendation": recommendation_text(primary_issue, dataset),
            }
        )
    return recommendations


def evaluate_density_rule(baseline_row: pd.Series, density_rows: pd.DataFrame) -> dict:
    baseline_abs = as_float(baseline_row["abs_bias_snips"])
    safe = density_rows[
        (density_rows["ess_fraction"] >= 0.20) & (density_rows["clip_rate"] <= 0.20)
    ].copy()
    if safe.empty:
        return {
            "triggered": False,
            "best_safe_abs_bias": None,
            "best_safe_reduction_fraction": 0.0,
            "best_safe_row": None,
        }
    best_safe = safe.sort_values(
        ["abs_bias_snips", "ess_fraction"],
        ascending=[True, False],
    ).iloc[0]
    reduction = reduction_fraction(baseline_abs, as_float(best_safe["abs_bias_snips"]))
    return {
        "triggered": bool(reduction >= 0.25),
        "best_safe_abs_bias": as_float(best_safe["abs_bias_snips"]),
        "best_safe_reduction_fraction": reduction,
        "best_safe_row": {
            "C": as_float(best_safe["C"]),
            "clip": clip_text(best_safe["clip_label"]),
            "calibrate": bool(best_safe["calibrate"]),
            "ess_fraction": as_float(best_safe["ess_fraction"]),
            "clip_rate": as_float(best_safe["clip_rate"]),
        },
    }


def evaluate_feature_rule(baseline_row: pd.Series, feature_rows: pd.DataFrame) -> dict:
    plus = feature_rows[feature_rows["mode"] == "embedding_plus_surface"]
    if plus.empty:
        return {
            "triggered": False,
            "embedding_plus_surface_reduction_fraction": 0.0,
            "ess_check_source": "unavailable",
        }
    plus_row = plus.iloc[0]
    reduction = as_float(plus_row["abs_bias_reduction_fraction"])
    # The feature ablation CSV does not include per-mode ESS. Use the baseline
    # density ESS as a conservative available guardrail.
    ess_ok = bool(as_float(baseline_row["ess_fraction"]) >= 0.20)
    return {
        "triggered": bool(reduction >= 0.25 and ess_ok),
        "embedding_plus_surface_reduction_fraction": reduction,
        "ess_ok": ess_ok,
        "ess_check_source": "baseline_density_ess_fraction",
    }


def evaluate_support_rule(
    support_row: pd.Series,
    baseline_row: pd.Series,
    density_rows: pd.DataFrame,
    density_eval: dict,
    feature_eval: dict,
) -> dict:
    severity = normalize_severity(support_row["support_severity"])
    severe = severity == "severe"
    higher_clip = higher_clip_improves_with_ess_collapse(baseline_row, density_rows)
    severe_overlap = bool(
        severe
        and (
            as_float(support_row["required_topk_fraction"]) < 0.15
            or as_float(support_row["knn10_gap_to_agent_mean"]) > 0.15
            or as_float(support_row["frac_agent_reward_above_knn10_max"]) > 0.40
            or as_float(baseline_row["ess_fraction"]) < 0.20
        )
    )
    triggered = bool(
        severe
        and (
            higher_clip["triggered"]
            or (severe_overlap and not density_eval["triggered"] and not feature_eval["triggered"])
        )
    )
    return {
        "triggered": triggered,
        "severity": severity,
        "higher_clip_improves_with_ess_collapse": higher_clip,
        "severe_overlap": severe_overlap,
    }


def higher_clip_improves_with_ess_collapse(
    baseline_row: pd.Series,
    density_rows: pd.DataFrame,
) -> dict:
    baseline_abs = as_float(baseline_row["abs_bias_snips"])
    baseline_clip = as_float(baseline_row["clip"])
    baseline_c = as_float(baseline_row["C"])
    baseline_calibrate = bool(baseline_row["calibrate"])
    candidates = density_rows[
        (density_rows["C"] == baseline_c) & (density_rows["calibrate"] == baseline_calibrate)
    ].copy()
    candidates = candidates[
        candidates["clip_label"].apply(lambda value: is_higher_clip(value, baseline_clip))
    ]
    if candidates.empty:
        return {"triggered": False}
    best = candidates.sort_values("abs_bias_snips").iloc[0]
    improves = as_float(best["abs_bias_snips"]) < baseline_abs
    collapse = as_float(best["ess_fraction"]) < 0.20
    return {
        "triggered": bool(improves and collapse),
        "best_higher_clip": clip_text(best["clip_label"]),
        "best_higher_clip_abs_bias": as_float(best["abs_bias_snips"]),
        "best_higher_clip_ess_fraction": as_float(best["ess_fraction"]),
    }


def top_surface_shift(shifts: pd.DataFrame, dataset: str) -> str:
    subset = shifts[shifts["dataset"] == dataset].copy()
    if subset.empty:
        return ""
    subset["abs_shift"] = subset["standardized_difference_agent_minus_behavior"].abs()
    top = subset.sort_values("abs_shift", ascending=False).iloc[0]
    return (
        f"{top['feature']} "
        f"({as_float(top['standardized_difference_agent_minus_behavior']):+.2f} SD)"
    )


def make_evidence_text(
    support_eval: dict,
    density_eval: dict,
    feature_eval: dict,
    top_shift: str,
) -> str:
    parts = [
        f"support={support_eval['severity']}",
        f"safe_density_reduction={format_pct(density_eval['best_safe_reduction_fraction'])}",
        f"surface_plus_reduction={format_pct(feature_eval['embedding_plus_surface_reduction_fraction'])}",
    ]
    if top_shift:
        parts.append(f"largest_surface_shift={top_shift}")
    return "; ".join(parts)


def recommendation_text(primary_issue: str, dataset: str) -> str:
    if primary_issue == "support/positivity":
        return (
            "Treat overlap as the main blocker before changing estimator defaults. "
            "Inspect high-reward target regions, avoid relying on high-variance clipping fixes, "
            "and consider restricted-scope evaluation or more logged behavior examples."
        )
    if primary_issue == "missing reward-relevant features":
        return (
            "Surface signals materially reduce diagnostic MIPS bias. Investigate these "
            "features as pre-registered diagnostic covariates before any production use."
        )
    if primary_issue == "density-ratio regularization/calibration":
        return (
            "A safe density setting improves diagnostic MIPS bias. Treat it as a candidate "
            "regularization/calibration direction, not as a final setting selected on target rewards."
        )
    return (
        "No single deterministic diagnosis dominates. Keep estimator defaults unchanged and "
        "inspect support, density, and feature-shift evidence jointly."
    )


def build_executive_summary(recommendations: list[dict]) -> list[str]:
    counts = pd.Series([r["primary_issue"] for r in recommendations]).value_counts().to_dict()
    return [
        "This report integrates support, density-ratio, and surface-feature diagnostics for the six expert-scored tasks using BGE-M3 + learned concat-PCA.",
        "This phase diagnoses point-estimate bias first; it does not claim that 95% CI coverage has been fixed.",
        "Oracle bias comparisons use y_agent_score only after fitting, for diagnostic evaluation and report ranking, not for training or production hyperparameter selection.",
        "Primary issue counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
    ]


def render_markdown(
    executive: list[str],
    support_rows: list[dict],
    baseline_rows: list[dict],
    best_density_rows: list[dict],
    feature_delta_rows: list[dict],
    recommendations: list[dict],
) -> str:
    lines = [
        "# Expert Best-Embedding Diagnostic Report",
        "",
        "## Executive Summary",
        "",
    ]
    lines.extend(f"- {item}" for item in executive)
    lines.extend(
        [
            "",
            "## Support / Positivity Severity",
            "",
            markdown_table(
                support_rows,
                [
                    "dataset",
                    "n",
                    "support_severity",
                    "required_topk_fraction",
                    "knn10_gap_to_agent_mean",
                    "frac_agent_reward_above_knn10_max",
                    "nn_distance_min_p90",
                ],
            ),
            "",
            "## Baseline MIPS Density Diagnostics",
            "",
            "Baseline means `C=0.1`, `clip=20`, `calibrate=True` under BGE-M3 + learned concat-PCA.",
            "",
            markdown_table(
                baseline_rows,
                [
                    "dataset",
                    "v_snips",
                    "abs_bias_snips",
                    "ess_fraction",
                    "clip_rate",
                    "top10_weight_mass",
                    "corr_w_y_behavior",
                ],
            ),
            "",
            "## Best Density Sweep Rows",
            "",
            "Warning: this table ranks rows by oracle diagnostic abs bias using `y_agent_score`. Use it to diagnose mechanisms, not to select production hyperparameters.",
            "",
            markdown_table(
                best_density_rows,
                [
                    "dataset",
                    "C",
                    "clip",
                    "calibrate",
                    "v_snips",
                    "abs_bias_snips",
                    "ess_fraction",
                    "clip_rate",
                    "safe_by_report_rule",
                ],
            ),
            "",
            "## Feature Ablation Deltas vs Embedding-Only",
            "",
            "MIPS rows only; negative delta means lower absolute bias than embedding-only.",
            "",
            markdown_table(
                feature_delta_rows,
                [
                    "dataset",
                    "mode",
                    "abs_bias",
                    "embedding_only_abs_bias",
                    "abs_bias_delta_vs_embedding_only",
                    "abs_bias_reduction_fraction",
                    "feature_dim_behavior",
                    "extra_dim",
                ],
            ),
            "",
            "## Final Recommendations",
            "",
            markdown_table(
                recommendations,
                [
                    "dataset",
                    "primary_issue",
                    "support_severity",
                    "baseline_abs_bias_snips",
                    "baseline_ess_fraction",
                    "best_safe_density_reduction_fraction",
                    "embedding_plus_surface_reduction_fraction",
                    "top_surface_shift",
                    "recommendation",
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_cell(row.get(col)) for col in columns) + " |")
    return "\n".join([header, divider, *body])


def write_report(report: dict, input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    md_path = input_dir / "diagnostic_report.md"
    json_path = input_dir / "diagnostic_report.json"
    md_path.write_text(report["markdown"] + "\n", encoding="utf-8")
    json_payload = {key: value for key, value in report.items() if key != "markdown"}
    json_path.write_text(json.dumps(json_payload, indent=2, default=safe_json), encoding="utf-8")
    print(f"Saved {md_path}")
    print(f"Saved {json_path}")


def normalize_severity(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"high", "severe"}:
        return "severe"
    if text in {"medium", "moderate"}:
        return "medium"
    if text in {"low", "none"}:
        return "low"
    return text


def is_higher_clip(value: Any, baseline_clip: float) -> bool:
    text = str(value).strip().lower()
    if text == "none":
        return True
    try:
        return float(text) > baseline_clip
    except ValueError:
        return False


def clip_text(value: Any) -> str:
    text = str(value).strip()
    if text.lower() == "none":
        return "none"
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else f"{number:g}"


def reduction_fraction(baseline_abs: float, candidate_abs: float) -> float:
    if baseline_abs <= 0:
        return 0.0
    return float((baseline_abs - candidate_abs) / baseline_abs)


def as_float(value: Any) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if abs(value) < 1 and value != 0:
            return f"{value:.3f}"
        return f"{value:.4f}"
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


def format_pct(value: Any) -> str:
    try:
        number = as_float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.1%}"


def safe_json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


if __name__ == "__main__":
    main()
