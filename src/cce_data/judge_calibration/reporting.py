from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _format_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    preview = df.head(max_rows).copy()
    columns = [str(column) for column in preview.columns]
    rows = [columns]
    for _, row in preview.iterrows():
        rows.append([_format_cell(row[column]) for column in preview.columns])
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(columns))]
    header = "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(rows[0])) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows[1:]
    ]
    return "\n".join([header, separator, *body])


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report(
    path: str | Path,
    dataset_summary: dict[str, Any],
    coverage_summary: dict[str, Any],
    split_summary: dict[str, Any],
    weights: dict[str, Any],
    metrics_overall: pd.DataFrame,
    group_tables: dict[str, pd.DataFrame],
    diagnostics_tables: dict[str, pd.DataFrame] | None = None,
    limitations: list[str] | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weight_items = weights.get("weights", {})
    sorted_weights = sorted(weight_items.items(), key=lambda item: item[1], reverse=True)
    limitations = limitations or []

    lines = [
        "# Weighted Judge Combination Report",
        "",
        "## Dataset Summary",
        "",
        f"- Rows: {dataset_summary.get('n_rows')}",
        f"- Datasets: {dataset_summary.get('datasets')}",
        f"- Score dimensions: {dataset_summary.get('score_dimensions')}",
        f"- Answer sources: {dataset_summary.get('answer_sources')}",
        "",
        "## Donor Judge Coverage",
        "",
        f"- Judge columns: {coverage_summary.get('judge_columns')}",
        f"- Rows dropped for missing judges: {coverage_summary.get('rows_dropped_missing_judges')}",
        "",
        "## Split Protocol",
        "",
        f"- Protocol: {split_summary.get('protocol')}",
        f"- Split name: {split_summary.get('name')}",
        f"- Counts: {split_summary.get('counts')}",
        f"- Held out: {split_summary.get('held_out_column')}={split_summary.get('held_out_value')}",
        "",
        "## Selected Model",
        "",
        f"- Lambda: {weights.get('selected_lambda', weights.get('lambda_reg'))}",
        f"- Fit intercept: {weights.get('selected_fit_intercept', weights.get('fit_intercept'))}",
        f"- Intercept: {weights.get('intercept')}",
        f"- Max weight: {weights.get('max_weight')}",
        f"- Effective number of judges: {weights.get('effective_num_judges')}",
        "",
        "### Learned Weights",
        "",
    ]
    if sorted_weights:
        lines.extend([f"- {name}: {value:.6f}" for name, value in sorted_weights])
    else:
        lines.append("_No weights recorded._")

    lines.extend(["", "## Overall Metrics", "", _format_table(metrics_overall)])
    for name, table in group_tables.items():
        lines.extend(["", f"## By {name}", "", _format_table(table)])
    for name, table in (diagnostics_tables or {}).items():
        lines.extend(["", f"## {name}", "", _format_table(table)])

    lines.extend(["", "## Limitations And Next Actions", ""])
    if limitations:
        lines.extend([f"- {item}" for item in limitations])
    else:
        lines.append("- Interpret mock-score runs only as pipeline smoke tests.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
