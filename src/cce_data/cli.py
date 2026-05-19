from __future__ import annotations

import argparse
import json
from pathlib import Path

from cce_data.agent import generate_agent_actions
from cce_data.build import build_dataset
from cce_data.download import download_sources
from cce_data.effect import compute_ground_truth_effect
from cce_data.freeze import freeze_dataset
from cce_data.inspect import inspect_dataset
from cce_data.rubric import score_dataset


def _parse_sources(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def download_main() -> None:
    parser = argparse.ArgumentParser(description="Download raw public data sources.")
    parser.add_argument("--sources", help="Comma-separated source names. Defaults to enabled sources.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = download_sources(
        sources=_parse_sources(args.sources),
        raw_dir=Path(args.raw_dir),
        config_path=Path(args.config),
        force=args.force,
    )
    print(json.dumps(results, indent=2))


def build_main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Phase 2 dataset.")
    parser.add_argument("--sources", help="Comma-separated source names. Defaults to enabled sources.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--config", default="configs/sources.yaml")
    parser.add_argument("--included-only", action="store_true")
    parser.add_argument("--include-dialog-only", action="store_true")
    parser.add_argument("--sample-per-source", type=int)
    args = parser.parse_args()
    manifest = build_dataset(
        sources=_parse_sources(args.sources),
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        include_excluded=not args.included_only,
        include_dialog_only=args.include_dialog_only,
        sample_per_source=args.sample_per_source,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def score_main() -> None:
    parser = argparse.ArgumentParser(description="Score canonical dataset with a rubric judge.")
    parser.add_argument("--input", default="data/processed/phase2_dataset.jsonl")
    parser.add_argument("--output", default="data/processed/phase2_dataset_scored.jsonl")
    parser.add_argument(
        "--provider",
        default="none",
        choices=["none", "openai", "anthropic", "google", "gemini"],
    )
    parser.add_argument("--model", help="Judge model. Defaults to configs/rubric_judge.yaml.")
    parser.add_argument("--rubric-config", default="configs/rubric_judge.yaml")
    parser.add_argument("--action-field", default="a_clinician")
    parser.add_argument("--score-field", default="y_score")
    parser.add_argument("--rubric-field", default="y_rubric")
    parser.add_argument("--source-field", default="y_source")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = score_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        provider=args.provider,
        model=args.model,
        rubric_config_path=Path(args.rubric_config),
        action_field=args.action_field,
        score_field=args.score_field,
        rubric_field=args.rubric_field,
        source_field=args.source_field,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact view of generated examples.")
    parser.add_argument("--path", default="data/processed/phase2_dataset.jsonl")
    parser.add_argument("-n", type=int, default=3)
    args = parser.parse_args()
    print(inspect_dataset(Path(args.path), n=args.n))


def freeze_main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a dataset version by hashing it.")
    parser.add_argument("--input", default="data/processed_included/phase2_dataset.jsonl")
    parser.add_argument("--output", default="dataset_freezes/phase2_v1.json")
    parser.add_argument("--label", default="phase2_v1")
    parser.add_argument("--manifest", default="data/processed_included/manifest.json")
    args = parser.parse_args()
    freeze = freeze_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        label=args.label,
        manifest_path=Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(freeze, indent=2, ensure_ascii=False))


def generate_agent_main() -> None:
    parser = argparse.ArgumentParser(description="Generate target-agent actions for Phase 3.")
    parser.add_argument("--input", default="data/phase3/clinician_scored_all.jsonl")
    parser.add_argument("--output", default="data/phase3/agent_actions_all.jsonl")
    parser.add_argument("--provider", default="openai", choices=["openai"])
    parser.add_argument("--model", help="Agent model. Defaults to configs/agent_policy.yaml.")
    parser.add_argument("--agent-config", default="configs/agent_policy.yaml")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate_agent_actions(
        input_path=Path(args.input),
        output_path=Path(args.output),
        provider=args.provider,
        model=args.model,
        agent_config_path=Path(args.agent_config),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def effect_main() -> None:
    parser = argparse.ArgumentParser(description="Compute Phase 3 ground-truth effect.")
    parser.add_argument("--input", default="data/phase3/agent_scored_all.jsonl")
    parser.add_argument("--output", default="data/phase3/ground_truth_effect.json")
    parser.add_argument("--clinician-score-field", default="y_score")
    parser.add_argument("--agent-score-field", default="y_agent_score")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()
    result = compute_ground_truth_effect(
        input_path=Path(args.input),
        output_path=Path(args.output),
        clinician_score_field=args.clinician_score_field,
        agent_score_field=args.agent_score_field,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
