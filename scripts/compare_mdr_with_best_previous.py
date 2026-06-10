"""Compare MDR with the best prior DM/MIPS/OffCEM estimator.

The comparison uses the same five-dataset view and two embeddings as
``run_mdr_five_datasets.py``. It reruns DM/MIPS/OffCEM with n_boot=0 for a
matched point-estimate comparison, selects the prior estimator with the
smallest absolute bias within each dataset x embedding cell, and plots that
against the MDR point result.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_mdr_five_datasets import (  # noqa: E402
    _base_specs,
    _learned_config,
    _load_records,
    _safe_json,
)
from run_phase5 import make_bge_m3_embedder  # noqa: E402

from cce_data.estimators.real_runner import (  # noqa: E402
    apply_learned_action_embedding,
    featurize_records,
    run_estimators_on_data,
)


EMBEDDINGS = {
    "BGE-M3": "bge",
    "BGE-M3 + learned concat-PCA": "bge_pca128_learn64",
}


def _load_mdr_rows(path: Path) -> dict[tuple[str, str], dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(row["dataset_key"], row["embedding"]): row for row in rows}


def _best_previous_row(spec, embedding: str, report: dict) -> dict:
    best_name, best_result = min(
        report["results"].items(),
        key=lambda item: abs(float(item[1]["bias"])),
    )
    return {
        "dataset": spec.display_name,
        "dataset_key": spec.name,
        "embedding": embedding,
        "n": report["n"],
        "v_true_b": report["v_true_b"],
        "v_true_agent": report["v_true_agent"],
        "true_effect": report["true_effect"],
        "estimator": best_name,
        "v_hat": best_result["v_hat"],
        "bias": best_result["bias"],
        "abs_bias": abs(best_result["bias"]),
        "rel_bias": best_result["rel_bias"],
        "abs_rel_bias": abs(best_result["rel_bias"]),
        "direction_correct": best_result["direction_correct"],
    }


def _comparison_rows(best_rows: list[dict], mdr_by_key: dict[tuple[str, str], dict]) -> list[dict]:
    rows: list[dict] = []
    for best in best_rows:
        key = (best["dataset_key"], best["embedding"])
        mdr = mdr_by_key[key]
        rows.append(
            {
                "dataset": best["dataset"],
                "dataset_key": best["dataset_key"],
                "embedding": best["embedding"],
                "n": best["n"],
                "v_true_b": best["v_true_b"],
                "v_true_agent": best["v_true_agent"],
                "true_effect": best["true_effect"],
                "previous_best_estimator": best["estimator"],
                "previous_best_v_hat": best["v_hat"],
                "previous_best_bias": best["bias"],
                "previous_best_abs_bias": best["abs_bias"],
                "previous_best_rel_bias": best["rel_bias"],
                "previous_best_direction_correct": best["direction_correct"],
                "mdr_v_hat": mdr["v_hat"],
                "mdr_bias": mdr["bias"],
                "mdr_abs_bias": abs(mdr["bias"]),
                "mdr_rel_bias": mdr["rel_bias"],
                "mdr_direction_correct": mdr["direction_correct"],
                "mdr_minus_previous_abs_bias": abs(mdr["bias"]) - best["abs_bias"],
                "winner": "MDR" if abs(mdr["bias"]) < best["abs_bias"] else "Previous best",
            }
        )
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _write_summary(rows: list[dict], path: Path, wall_time: float) -> None:
    wins = {
        "Previous best": sum(row["winner"] == "Previous best" for row in rows),
        "MDR": sum(row["winner"] == "MDR" for row in rows),
    }
    lines = [
        "# MDR vs Previous Best Estimator",
        "",
        "Comparison scope: five datasets x two embeddings. The previous-best estimator is selected",
        "from DM/MIPS/OffCEM by smallest absolute point-estimate bias in the same cell.",
        "",
        f"- Wall time for matched previous-estimator point rerun: {wall_time:.1f}s",
        f"- Previous best wins: {wins['Previous best']} / {len(rows)}",
        f"- MDR wins: {wins['MDR']} / {len(rows)}",
        "",
        "| Dataset | Embedding | Prev best | Prev abs bias | MDR abs bias | Delta MDR-prev | Winner |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["embedding"],
                    row["previous_best_estimator"],
                    f"{row['previous_best_abs_bias']:.4f}",
                    f"{row['mdr_abs_bias']:.4f}",
                    f"{row['mdr_minus_previous_abs_bias']:+.4f}",
                    row["winner"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Detailed point estimates:",
            "",
            "| Dataset | Embedding | V_agent | Prev V_hat | Prev rel bias | MDR V_hat | MDR rel bias |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["embedding"],
                    f"{row['v_true_agent']:.4f}",
                    f"{row['previous_best_v_hat']:.4f}",
                    _pct(row["previous_best_rel_bias"]),
                    f"{row['mdr_v_hat']:.4f}",
                    _pct(row["mdr_rel_bias"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_plot(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    dataset_order = ["ACI-Bench", "MTS-Dialog", "PriMock57", "CounselBench-Eval pooled", "Ayers AskDocs"]
    embedding_order = ["BGE-M3", "BGE-M3 + learned concat-PCA"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, embedding in zip(axes, embedding_order):
        sub = [row for row in rows if row["embedding"] == embedding]
        sub = sorted(sub, key=lambda row: dataset_order.index(row["dataset"]))
        x = np.arange(len(sub))
        width = 0.36
        prev = [row["previous_best_abs_bias"] for row in sub]
        mdr = [row["mdr_abs_bias"] for row in sub]
        bars_prev = ax.bar(
            x - width / 2,
            prev,
            width,
            label="Previous best",
            color="#2F6B9A",
        )
        bars_mdr = ax.bar(
            x + width / 2,
            mdr,
            width,
            label="MDR",
            color="#D17A22",
        )
        for bar, row in zip(bars_prev, sub):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                row["previous_best_estimator"],
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar in bars_mdr:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                "MDR",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(embedding)
        ax.set_xticks(x)
        ax.set_xticklabels([row["dataset"].replace("CounselBench-Eval ", "CB-") for row in sub])
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", color="#D8DEE6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(max(prev), max(mdr)) + 0.07)

    axes[0].set_ylabel("Absolute bias vs V_agent")
    axes[0].legend(loc="upper left")
    fig.suptitle("MDR vs best previous estimator (lower absolute bias is better)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase5/mdr_vs_best_previous"))
    parser.add_argument(
        "--mdr-summary-rows",
        type=Path,
        default=Path("data/phase5/mdr_five_datasets_point/summary_rows.json"),
    )
    parser.add_argument(
        "--counselbench-mode",
        choices=["pooled", "gpt4", "llama3", "gemini"],
        default="pooled",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mdr_by_key = _load_mdr_rows(args.mdr_summary_rows)

    embed_fn = make_bge_m3_embedder()
    learned_config = _learned_config(args.seed)
    best_rows: list[dict] = []
    all_reports = {}

    t0 = time.time()
    for spec in _base_specs(args.counselbench_mode):
        print()
        print("=" * 80)
        print(f"Dataset: {spec.display_name}")
        print("=" * 80)
        records = _load_records(spec)
        data = featurize_records(records, embed_fn)

        plain_report = run_estimators_on_data(
            data,
            n_boot=0,
            seed=args.seed,
            estimator_names=["DM", "MIPS", "OffCEM"],
        )
        best_rows.append(_best_previous_row(spec, "BGE-M3", plain_report))
        all_reports[(spec.name, "BGE-M3")] = plain_report

        learned_data, learned_diagnostics = apply_learned_action_embedding(data, learned_config)
        learned_report = run_estimators_on_data(
            learned_data,
            n_boot=0,
            seed=args.seed,
            estimator_names=["DM", "MIPS", "OffCEM"],
            learned_diagnostics=learned_diagnostics,
        )
        best_rows.append(_best_previous_row(spec, "BGE-M3 + learned concat-PCA", learned_report))
        all_reports[(spec.name, "BGE-M3 + learned concat-PCA")] = learned_report

    wall_time = time.time() - t0
    rows = _comparison_rows(best_rows, mdr_by_key)
    rows = sorted(rows, key=lambda row: (row["dataset_key"], EMBEDDINGS[row["embedding"]]))

    comparison_json = args.output_dir / "comparison_rows.json"
    comparison_json.write_text(json.dumps(rows, indent=2, default=_safe_json), encoding="utf-8")
    _write_csv(rows, args.output_dir / "comparison_rows.csv")
    _write_summary(rows, args.output_dir / "summary.md", wall_time)
    _make_plot(rows, args.output_dir / "mdr_vs_best_previous_abs_bias.png")

    reports_json = args.output_dir / "previous_estimator_point_reports.json"
    serializable_reports = {f"{key[0]}::{key[1]}": value for key, value in all_reports.items()}
    reports_json.write_text(json.dumps(serializable_reports, indent=2, default=_safe_json), encoding="utf-8")

    print()
    print(f"Saved {comparison_json}")
    print(f"Saved {args.output_dir / 'comparison_rows.csv'}")
    print(f"Saved {args.output_dir / 'summary.md'}")
    print(f"Saved {args.output_dir / 'mdr_vs_best_previous_abs_bias.png'}")
    print(f"Saved {reports_json}")


if __name__ == "__main__":
    main()
