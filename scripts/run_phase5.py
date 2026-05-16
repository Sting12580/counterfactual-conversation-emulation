"""
Phase 5 main experiment: run DM / MIPS / OffCEM on real Phase 3 data.

Input:  data/phase3/agent_scored_all_judge_gpt4o.jsonl  (default)
Output: prints headline table; writes results to data/phase5/headline.json

Usage:
    python scripts/run_phase5.py
    python scripts/run_phase5.py --embedder openai   # if OPENAI_API_KEY set
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cce_data.estimators.real_runner import (
    format_headline_table,
    run_phase5_headline,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def make_sbert_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Local sentence-BERT embedder (default; no API key needed)."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading {model_name} ...")
    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return embed


def make_openai_embedder(model_name: str = "text-embedding-3-small"):
    """OpenAI embedder (requires OPENAI_API_KEY)."""
    from openai import OpenAI

    client = OpenAI()

    def embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 64):
            batch = texts[i:i + 64]
            resp = client.embeddings.create(model=model_name, input=batch)
            vecs.extend(d.embedding for d in resp.data)
        return np.array(vecs, dtype=np.float32)
    return embed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/phase3/agent_scored_all_judge_gpt4o.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/phase5/headline.json"),
    )
    parser.add_argument(
        "--embedder", choices=["sbert", "openai"], default="sbert",
    )
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input: {args.input}")

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    print(f"  {len(records)} records loaded")

    if args.embedder == "sbert":
        embed_fn = make_sbert_embedder()
    else:
        embed_fn = make_openai_embedder()

    t0 = time.time()
    report = run_phase5_headline(records, embed_fn, n_boot=args.n_boot, seed=args.seed)
    print()
    print(format_headline_table(report))
    print()
    print(f"Total wall time: {time.time() - t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
