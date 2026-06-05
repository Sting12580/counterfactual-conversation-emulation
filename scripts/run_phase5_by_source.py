"""Split the mixed Phase 3 dataset by source and run Phase 5 per source.

Default settings use the current best point-estimation setup from the Phase 5
summary: BGE-M3 + learned concat-PCA with PCA=128 and latent=64.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_phase5 import (  # noqa: E402
    load_jsonl,
    make_bge_m3_embedder,
    make_medcpt_bge_embedder,
    make_medcpt_embedder,
    make_openai_embedder,
    make_sbert_embedder,
    remap_score_fields,
)

from cce_data.build import write_jsonl  # noqa: E402
from cce_data.estimators.learned_embedding import LearnedEmbeddingConfig  # noqa: E402
from cce_data.estimators.real_runner import format_headline_table, run_phase5_headline  # noqa: E402


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def split_by_source(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        source = str(record.get("source") or "unknown")
        groups[source].append(record)
    return dict(sorted(groups.items()))


def build_learned_config(args: argparse.Namespace) -> LearnedEmbeddingConfig | None:
    if not args.learned_action_embedding:
        return None
    return LearnedEmbeddingConfig(
        latent_dim=args.learned_dim,
        hidden_dim=args.learned_hidden_dim,
        merge_strategy=args.learned_merge,
        pca_dim=args.pca_dim,
        max_epochs=args.learned_epochs,
        learning_rate=args.learned_lr,
        weight_decay=args.learned_weight_decay,
        validation_fraction=args.learned_validation_fraction,
        patience=args.learned_patience,
        seed=args.seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl"),
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/phase3/by_source"),
        help="Where to write one remapped JSONL per source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/phase5/by_source"),
        help="Where to write one Phase 5 report JSON per source.",
    )
    parser.add_argument(
        "--run-name",
        default="sonnet46",
        help="Name used in split/result filenames.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional subset of source names to run, e.g. aci_bench mts_dialog.",
    )
    parser.add_argument(
        "--score-field",
        default="y_score_claude_sonnet46",
        help="Source field to expose as y_score. Use '' to keep existing y_score.",
    )
    parser.add_argument(
        "--agent-score-field",
        default="y_agent_score_claude_sonnet46",
        help="Source field to expose as y_agent_score. Use '' to keep existing y_agent_score.",
    )
    parser.add_argument(
        "--embedder",
        choices=["sbert", "openai", "medcpt", "bge", "medcpt-bge"],
        default="bge",
    )
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learned-action-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the learned action embedding layer; enabled by default.",
    )
    parser.add_argument("--learned-dim", type=int, default=64)
    parser.add_argument("--learned-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--learned-merge",
        choices=["replace", "concat-pca"],
        default="concat-pca",
    )
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--learned-epochs", type=int, default=100)
    parser.add_argument("--learned-lr", type=float, default=1e-3)
    parser.add_argument("--learned-weight-decay", type=float, default=1e-2)
    parser.add_argument("--learned-patience", type=int, default=50)
    parser.add_argument("--learned-validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only write the per-source JSONL files; do not run Phase 5.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing input: {args.input}")

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    records = remap_score_fields(
        records,
        score_field=args.score_field or None,
        agent_score_field=args.agent_score_field or None,
    )

    groups = split_by_source(records)
    if args.sources:
        requested = set(args.sources)
        missing = sorted(requested - set(groups))
        if missing:
            raise ValueError(f"Requested source(s) not found: {', '.join(missing)}")
        groups = {source: groups[source] for source in sorted(requested)}

    args.split_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input": str(args.input),
        "run_name": args.run_name,
        "score_field": args.score_field or None,
        "agent_score_field": args.agent_score_field or None,
        "embedder": args.embedder,
        "learned_action_embedding": args.learned_action_embedding,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "sources": {},
    }

    print("Writing per-source inputs:")
    split_paths: dict[str, Path] = {}
    for source, source_records in groups.items():
        safe_source = _safe_name(source)
        split_path = args.split_dir / f"agent_scored_{args.run_name}_{safe_source}.jsonl"
        write_jsonl(source_records, split_path)
        split_paths[source] = split_path
        manifest["sources"][source] = {
            "n_input": len(source_records),
            "split_path": str(split_path),
        }
        print(f"  {source}: {len(source_records)} records -> {split_path}")

    if args.split_only:
        manifest_path = args.output_dir / f"manifest_{args.run_name}_by_source.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved {manifest_path}")
        return

    embedder_factories = {
        "sbert": make_sbert_embedder,
        "openai": make_openai_embedder,
        "medcpt": make_medcpt_embedder,
        "bge": make_bge_m3_embedder,
        "medcpt-bge": make_medcpt_bge_embedder,
    }
    embed_fn = embedder_factories[args.embedder]()
    learned_config = build_learned_config(args)

    label_parts = [args.run_name, "{source}", args.embedder]
    if learned_config is not None:
        if learned_config.merge_strategy == "concat-pca":
            label_parts.append(f"pca{learned_config.pca_dim}")
        label_parts.append(f"learn{learned_config.latent_dim}")

    t0 = time.time()
    for source, source_records in groups.items():
        safe_source = _safe_name(source)
        output_name = "headline_" + "_".join(label_parts).format(source=safe_source) + ".json"
        output_path = args.output_dir / output_name

        print()
        print("=" * 80)
        print(f"Running source={source} n={len(source_records)} -> {output_path}")
        print("=" * 80)
        report = run_phase5_headline(
            source_records,
            embed_fn,
            n_boot=args.n_boot,
            seed=args.seed,
            learned_embedding=learned_config,
        )
        print()
        print(format_headline_table(report))

        report["source"] = source
        report["input_path"] = str(split_paths[source])
        report["run_name"] = args.run_name
        output_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        manifest["sources"][source]["n_phase5"] = report["n"]
        manifest["sources"][source]["output_path"] = str(output_path)
        print(f"Saved {output_path}")

    manifest_path = args.output_dir / f"manifest_{args.run_name}_by_source.json"
    manifest["wall_time_seconds"] = time.time() - t0
    manifest_path.write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    print()
    print(f"Saved {manifest_path}")
    print(f"Total wall time: {manifest['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
