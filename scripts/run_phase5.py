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
import sys
import time
from pathlib import Path

import numpy as np

from cce_data.estimators.learned_embedding import LearnedEmbeddingConfig
from cce_data.estimators.real_runner import (
    format_headline_table,
    run_phase5_headline,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def remap_score_fields(
    records: list[dict],
    score_field: str | None = None,
    agent_score_field: str | None = None,
) -> list[dict]:
    """Expose judge-specific score fields as the standard Phase 5 score names."""
    if not score_field and not agent_score_field:
        return records

    out = []
    for record in records:
        row = dict(record)
        if score_field:
            row["y_score"] = row.get(score_field)
        if agent_score_field:
            row["y_agent_score"] = row.get(agent_score_field)
        out.append(row)
    return out


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


def _hf_mean_pool_embedder(
    model_name: str,
    out_dim: int,
    batch_size: int = 16,
    max_length: int = 512,
):
    """Generic HuggingFace mean-pool + L2-normalize embedder.

    Returns float32 (n, out_dim). Picks MPS / CUDA / CPU automatically.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  Loading {model_name} on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.inference_mode()
    def embed(texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), out_dim), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(
                batch, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            h = model(**enc).last_hidden_state           # (b, L, d)
            mask = enc["attention_mask"].unsqueeze(-1)   # (b, L, 1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[i:i + len(batch)] = pooled.float().cpu().numpy()
        return out
    return embed


def make_medcpt_embedder(model_name: str = "ncbi/MedCPT-Article-Encoder"):
    """MedCPT (medical-specific) embedder, 768-d, mean-pooled + L2-normalized."""
    return _hf_mean_pool_embedder(model_name, out_dim=768, batch_size=16)


def make_bge_m3_embedder(model_name: str = "BAAI/bge-m3"):
    """BGE-M3 (multilingual long-text) embedder, 1024-d dense, L2-normalized."""
    return _hf_mean_pool_embedder(model_name, out_dim=1024, batch_size=8, max_length=512)


def make_medcpt_bge_embedder():
    """Concat MedCPT (768) + BGE-M3 (1024) -> 1792-d."""
    medcpt = make_medcpt_embedder()
    bge = make_bge_m3_embedder()

    def embed(texts: list[str]) -> np.ndarray:
        return np.concatenate([medcpt(texts), bge(texts)], axis=1).astype(np.float32)
    return embed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

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
        "--embedder",
        choices=["sbert", "openai", "medcpt", "bge", "medcpt-bge"],
        default="sbert",
    )
    parser.add_argument(
        "--score-field",
        default="",
        help="Optional source field to expose as y_score before running Phase 5.",
    )
    parser.add_argument(
        "--agent-score-field",
        default="",
        help="Optional source field to expose as y_agent_score before running Phase 5.",
    )
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--learned-action-embedding",
        action="store_true",
        help=(
            "Train a FineTune-style reward-informed projection on logged "
            "clinician rewards before running DM/MIPS/OffCEM."
        ),
    )
    parser.add_argument("--learned-dim", type=int, default=128)
    parser.add_argument("--learned-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--learned-merge",
        choices=["replace", "concat-pca"],
        default="replace",
        help=(
            "How learned embeddings replace or preserve the base text embedding. "
            "'replace' uses only learned z; 'concat-pca' appends learned z to "
            "a PCA projection of the frozen base embedding."
        ),
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=0,
        help="PCA dimension for --learned-merge concat-pca.",
    )
    parser.add_argument("--learned-epochs", type=int, default=500)
    parser.add_argument("--learned-lr", type=float, default=1e-3)
    parser.add_argument("--learned-weight-decay", type=float, default=1e-3)
    parser.add_argument("--learned-dropout", type=float, default=0.0)
    parser.add_argument("--learned-patience", type=int, default=50)
    parser.add_argument("--learned-validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input: {args.input}")

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    records = remap_score_fields(
        records,
        score_field=args.score_field or None,
        agent_score_field=args.agent_score_field or None,
    )
    print(f"  {len(records)} records loaded")

    embedder_factories = {
        "sbert": make_sbert_embedder,
        "openai": make_openai_embedder,
        "medcpt": make_medcpt_embedder,
        "bge": make_bge_m3_embedder,
        "medcpt-bge": make_medcpt_bge_embedder,
    }
    embed_fn = embedder_factories[args.embedder]()
    learned_config = None
    if args.learned_action_embedding:
        learned_config = LearnedEmbeddingConfig(
            latent_dim=args.learned_dim,
            hidden_dim=args.learned_hidden_dim,
            merge_strategy=args.learned_merge,
            pca_dim=args.pca_dim,
            max_epochs=args.learned_epochs,
            learning_rate=args.learned_lr,
            weight_decay=args.learned_weight_decay,
            dropout=args.learned_dropout,
            validation_fraction=args.learned_validation_fraction,
            patience=args.learned_patience,
            seed=args.seed,
        )

    t0 = time.time()
    report = run_phase5_headline(
        records,
        embed_fn,
        n_boot=args.n_boot,
        seed=args.seed,
        learned_embedding=learned_config,
    )
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
