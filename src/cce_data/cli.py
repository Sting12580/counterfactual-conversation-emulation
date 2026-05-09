from __future__ import annotations

import argparse
import json
from pathlib import Path

from cce_data.build import build_dataset
from cce_data.download import download_sources
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
    parser.add_argument("--provider", default="none", choices=["none", "openai"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = score_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        provider=args.provider,
        model=args.model,
        limit=args.limit,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def inspect_main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact view of generated examples.")
    parser.add_argument("--path", default="data/processed/phase2_dataset.jsonl")
    parser.add_argument("-n", type=int, default=3)
    args = parser.parse_args()
    print(inspect_dataset(Path(args.path), n=args.n))
