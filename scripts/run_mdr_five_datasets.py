"""Run MDR on the five dataset view requested for Phase 5 follow-up.

Datasets:
  - ACI-Bench source split, n=207
  - MTS-Dialog source split, n=142
  - PriMock57 source split, n=48
  - CounselBench-Eval, pooled target responders by default
  - Ayers AskDocs composite reward, n=195

For each dataset, the script computes BGE-M3 embeddings once, then evaluates:
  - BGE-M3 + MDR
  - BGE-M3 + learned concat-PCA + MDR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase5 import load_jsonl, make_bge_m3_embedder, remap_score_fields  # noqa: E402

from cce_data.estimators.learned_embedding import LearnedEmbeddingConfig  # noqa: E402
from cce_data.estimators.real_runner import (  # noqa: E402
    apply_learned_action_embedding,
    featurize_records,
    format_headline_table,
    run_estimators_on_data,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    input_paths: tuple[Path, ...]
    score_field: str | None = None
    agent_score_field: str | None = None
    notes: str = ""


def _safe_json(value):
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


def _base_specs(counselbench_mode: str) -> list[DatasetSpec]:
    source_score = "y_score_claude_sonnet46"
    source_agent_score = "y_agent_score_claude_sonnet46"
    specs = [
        DatasetSpec(
            name="aci_bench",
            display_name="ACI-Bench",
            input_paths=(Path("data/phase3/by_source/agent_scored_sonnet46_aci_bench.jsonl"),),
            score_field=source_score,
            agent_score_field=source_agent_score,
            notes="Sonnet 4.6 judged source split from the mixed medical dialogue dataset.",
        ),
        DatasetSpec(
            name="mts_dialog",
            display_name="MTS-Dialog",
            input_paths=(Path("data/phase3/by_source/agent_scored_sonnet46_mts_dialog.jsonl"),),
            score_field=source_score,
            agent_score_field=source_agent_score,
            notes="Sonnet 4.6 judged source split from the mixed medical dialogue dataset.",
        ),
        DatasetSpec(
            name="primock57",
            display_name="PriMock57",
            input_paths=(Path("data/phase3/by_source/agent_scored_sonnet46_primock57.jsonl"),),
            score_field=source_score,
            agent_score_field=source_agent_score,
            notes="Sonnet 4.6 judged source split from the mixed medical dialogue dataset.",
        ),
    ]
    counselbench_paths = {
        "gemini": (Path("data/counselbench/phase3_gemini_expert_scored.jsonl"),),
        "gpt4": (Path("data/counselbench/phase3_gpt4_expert_scored.jsonl"),),
        "llama3": (Path("data/counselbench/phase3_llama3_expert_scored.jsonl"),),
        "pooled": (
            Path("data/counselbench/phase3_gemini_expert_scored.jsonl"),
            Path("data/counselbench/phase3_gpt4_expert_scored.jsonl"),
            Path("data/counselbench/phase3_llama3_expert_scored.jsonl"),
        ),
    }
    specs.append(
        DatasetSpec(
            name=f"counselbench_{counselbench_mode}",
            display_name=(
                "CounselBench-Eval pooled"
                if counselbench_mode == "pooled"
                else f"CounselBench-Eval {counselbench_mode}"
            ),
            input_paths=counselbench_paths[counselbench_mode],
            notes=(
                "Pooled GPT-4, LLaMA-3, and Gemini target responders."
                if counselbench_mode == "pooled"
                else f"Single target responder: {counselbench_mode}."
            ),
        )
    )
    specs.append(
        DatasetSpec(
            name="ayers_askdocs",
            display_name="Ayers AskDocs",
            input_paths=(Path("data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl"),),
            notes="Composite healthcare-professional reward: mean quality/empathy.",
        )
    )
    return specs


def _load_records(spec: DatasetSpec) -> list[dict]:
    records: list[dict] = []
    for path in spec.input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing input for {spec.name}: {path}")
        records.extend(load_jsonl(path))
    return remap_score_fields(records, spec.score_field, spec.agent_score_field)


def _learned_config(seed: int) -> LearnedEmbeddingConfig:
    return LearnedEmbeddingConfig(
        latent_dim=64,
        hidden_dim=64,
        merge_strategy="concat-pca",
        pca_dim=128,
        max_epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-2,
        validation_fraction=0.2,
        patience=50,
        seed=seed,
    )


def _row_from_report(spec: DatasetSpec, embedding: str, report: dict, output_path: Path) -> dict:
    result = report["results"]["MDR"]
    row = {
        "dataset": spec.display_name,
        "dataset_key": spec.name,
        "embedding": embedding,
        "n": report["n"],
        "v_true_b": report["v_true_b"],
        "v_true_agent": report["v_true_agent"],
        "true_effect": report["true_effect"],
        "v_hat": result["v_hat"],
        "bias": result["bias"],
        "rel_bias": result["rel_bias"],
        "direction_correct": result["direction_correct"],
        "output": str(output_path),
    }
    if "rmse" in result:
        row.update(
            {
                "rmse": result["rmse"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "ci_covers_truth": result["ci_covers_truth"],
                "direction_rate": result["direction_rate"],
            }
        )
    return row


def _fmt(value: float | int | bool | str | None, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_summary(rows: list[dict], path: Path, n_boot: int, counselbench_mode: str) -> None:
    lines = [
        "# MDR Five-Dataset Experiment",
        "",
        "Estimator: standard marginalized doubly robust (MDR) with unnormalized residual correction.",
        "",
        f"- Bootstrap iterations: {n_boot}",
        f"- CounselBench mode: `{counselbench_mode}`",
        "- Embeddings: `BGE-M3`; `BGE-M3 + learned concat-PCA` with PCA=128 and learned dim=64",
        "",
        "| Dataset | Embedding | n | V_b | V_agent | Effect | MDR V_hat | Bias | RelBias | RMSE | 95% CI | Dir% | Cov |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        ci = ""
        if "ci_low" in row:
            ci = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        dir_rate = row.get("direction_rate")
        dir_text = f"{dir_rate:.1%}" if isinstance(dir_rate, float) else ("yes" if row["direction_correct"] else "no")
        rel_bias = f"{row['rel_bias']:.2%}"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["embedding"],
                    str(row["n"]),
                    _fmt(row["v_true_b"]),
                    _fmt(row["v_true_agent"]),
                    f"{row['true_effect']:+.4f}",
                    _fmt(row["v_hat"]),
                    f"{row['bias']:+.4f}",
                    rel_bias,
                    _fmt(row.get("rmse")),
                    ci,
                    dir_text,
                    _fmt(row.get("ci_covers_truth")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/phase5/mdr_five_datasets"))
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--counselbench-mode",
        choices=["pooled", "gpt4", "llama3", "gemini"],
        default="pooled",
        help="Treat CounselBench as one pooled dataset by default.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset keys to run, e.g. aci_bench mts_dialog ayers_askdocs.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = _base_specs(args.counselbench_mode)
    if args.datasets:
        requested = set(args.datasets)
        missing = sorted(requested - {spec.name for spec in specs})
        if missing:
            raise ValueError(f"Unknown dataset key(s): {', '.join(missing)}")
        specs = [spec for spec in specs if spec.name in requested]

    embed_fn = make_bge_m3_embedder()
    learned_config = _learned_config(args.seed)
    rows: list[dict] = []
    manifest = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "counselbench_mode": args.counselbench_mode,
        "estimator": "MDR",
        "embeddings": ["bge", "bge_pca128_learn64"],
        "datasets": {},
    }

    t0 = time.time()
    for spec in specs:
        print()
        print("=" * 80)
        print(f"Dataset: {spec.display_name}")
        print("=" * 80)
        records = _load_records(spec)
        print(f"  Loaded {len(records)} records from {', '.join(str(p) for p in spec.input_paths)}")
        data = featurize_records(records, embed_fn)

        plain_report = run_estimators_on_data(
            data,
            n_boot=args.n_boot,
            seed=args.seed,
            estimator_names=["MDR"],
        )
        plain_report["dataset"] = {
            "name": spec.name,
            "display_name": spec.display_name,
            "input_paths": [str(path) for path in spec.input_paths],
            "notes": spec.notes,
        }
        plain_report["embedding"] = {"name": "BGE-M3", "learned_action_embedding": False}
        plain_path = args.output_dir / f"headline_{spec.name}_bge_mdr_boot{args.n_boot}.json"
        plain_path.write_text(json.dumps(plain_report, indent=2, default=_safe_json), encoding="utf-8")
        print(format_headline_table(plain_report))
        print(f"Saved {plain_path}")
        rows.append(_row_from_report(spec, "BGE-M3", plain_report, plain_path))

        print(
            "  Learning reward-informed action embedding "
            f"(latent_dim={learned_config.latent_dim}, merge={learned_config.merge_strategy}) ..."
        )
        learned_data, learned_diagnostics = apply_learned_action_embedding(data, learned_config)
        learned_report = run_estimators_on_data(
            learned_data,
            n_boot=args.n_boot,
            seed=args.seed,
            estimator_names=["MDR"],
            learned_diagnostics=learned_diagnostics,
        )
        learned_report["dataset"] = plain_report["dataset"]
        learned_report["embedding"] = {
            "name": "BGE-M3 + learned concat-PCA",
            "learned_action_embedding": True,
            "pca_dim": learned_config.pca_dim,
            "learned_dim": learned_config.latent_dim,
        }
        learned_path = args.output_dir / (
            f"headline_{spec.name}_bge_pca128_learn64_mdr_boot{args.n_boot}.json"
        )
        learned_path.write_text(
            json.dumps(learned_report, indent=2, default=_safe_json), encoding="utf-8"
        )
        print(format_headline_table(learned_report))
        print(f"Saved {learned_path}")
        rows.append(_row_from_report(spec, "BGE-M3 + learned concat-PCA", learned_report, learned_path))

        manifest["datasets"][spec.name] = {
            "display_name": spec.display_name,
            "input_paths": [str(path) for path in spec.input_paths],
            "notes": spec.notes,
            "outputs": {
                "bge": str(plain_path),
                "bge_pca128_learn64": str(learned_path),
            },
        }

    manifest["wall_time_seconds"] = time.time() - t0
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_safe_json), encoding="utf-8")
    rows_path = args.output_dir / "summary_rows.json"
    rows_path.write_text(json.dumps(rows, indent=2, default=_safe_json), encoding="utf-8")
    summary_path = args.output_dir / "summary.md"
    _write_summary(rows, summary_path, args.n_boot, args.counselbench_mode)

    print()
    print(f"Saved {manifest_path}")
    print(f"Saved {rows_path}")
    print(f"Saved {summary_path}")
    print(f"Total wall time: {manifest['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
