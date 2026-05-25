"""Compute jackknife+ conformal CIs for learned action embeddings.

This mirrors the Phase 6 learned-action-embedding runner, then replaces the
plain bootstrap interval with a jackknife+ interval built from each
estimator's per-sample psi values.

Default settings reproduce the current best learned run:

    BGE-M3 + concat-pca(pca_dim=128) + learned_dim=64

Usage:
    PYTHONPATH=src python scripts/compute_learned_conformal_ci.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase5 import (  # noqa: E402
    load_jsonl,
    make_bge_m3_embedder,
    make_medcpt_bge_embedder,
    make_medcpt_embedder,
    make_openai_embedder,
    make_sbert_embedder,
)

from cce_data.estimators.conformal_ci import jackknife_plus_ci  # noqa: E402
from cce_data.estimators.learned_embedding import LearnedEmbeddingConfig  # noqa: E402
from cce_data.estimators.real_runner import (  # noqa: E402
    RealData,
    apply_learned_action_embedding,
    dm_real,
    featurize_records,
    mips_real,
    offcem_real,
)


Estimator = Callable[[RealData, int], dict]


def remap_score_fields(
    records: list[dict],
    score_field: str | None,
    agent_score_field: str | None,
) -> list[dict]:
    """Return records with selected judge fields exposed as y_score names."""
    out = []
    for record in records:
        row = dict(record)
        if score_field:
            row["y_score"] = row.get(score_field)
        if agent_score_field:
            row["y_agent_score"] = row.get(agent_score_field)
        out.append(row)
    return out


def compute_learned_conformal_report(
    records: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
    learned_config: LearnedEmbeddingConfig,
    *,
    alpha: float = 0.05,
    seed: int = 0,
    existing_headline: dict | None = None,
) -> dict:
    """Embed data, apply learned projection, and compute conformal CIs."""
    data = featurize_records(records, embed_fn)
    data, learned_diagnostics = apply_learned_action_embedding(data, learned_config)

    v_true_b = float(data.y_clinician.mean())
    v_true_agent = float(data.y_agent.mean())
    true_effect = v_true_agent - v_true_b

    results = {}
    estimators: list[tuple[str, Estimator]] = [
        ("DM", dm_real),
        ("MIPS", mips_real),
        ("OffCEM", offcem_real),
    ]
    for name, estimator in estimators:
        out = estimator(data, seed=seed)
        psi = np.asarray(out["psi"], dtype=float)
        point, ci_low, ci_high = jackknife_plus_ci(psi, alpha=alpha)
        if not np.isclose(point, out["v_hat"], atol=1e-6):
            raise ValueError(
                f"{name} psi mean mismatch: mean(psi)={point}, v_hat={out['v_hat']}"
            )

        entry = {
            "v_hat": float(out["v_hat"]),
            "bias": float(out["v_hat"] - v_true_agent),
            "rel_bias": float((out["v_hat"] - v_true_agent) / v_true_agent),
            "conformal_ci_low": float(ci_low),
            "conformal_ci_high": float(ci_high),
            "conformal_ci_covers_truth": bool(ci_low <= v_true_agent <= ci_high),
            "conformal_half_width": float((ci_high - ci_low) / 2),
            "psi_std": float(psi.std(ddof=1)),
            "psi_min": float(psi.min()),
            "psi_max": float(psi.max()),
        }

        if existing_headline is not None:
            existing = existing_headline.get("results", {}).get(name)
            if existing is not None:
                entry.update({
                    "existing_v_hat": float(existing["v_hat"]),
                    "point_delta_vs_existing": float(out["v_hat"] - existing["v_hat"]),
                })
                if "ci_low" in existing and "ci_high" in existing:
                    plain_low = float(existing["ci_low"])
                    plain_high = float(existing["ci_high"])
                    entry.update({
                        "plain_ci_low": plain_low,
                        "plain_ci_high": plain_high,
                        "plain_ci_covers_truth": bool(
                            plain_low <= v_true_agent <= plain_high
                        ),
                        "plain_half_width": float((plain_high - plain_low) / 2),
                    })
        results[name] = entry

    learned_diagnostics["conformal_reuses_fitted_learned_embedding"] = True
    return {
        "n": data.n,
        "alpha": alpha,
        "v_true_b": v_true_b,
        "v_true_agent": v_true_agent,
        "true_effect": true_effect,
        "learned_embedding": learned_diagnostics,
        "results": results,
    }


def format_conformal_table(report: dict) -> str:
    """Pretty-print conformal results against existing bootstrap CIs."""
    lines = [
        "=" * 104,
        f"Learned Action Embedding Conformal CI (n={report['n']}, alpha={report['alpha']})",
        "=" * 104,
        f"V_true(pi_b)     = {report['v_true_b']:.4f}",
        f"V_true(pi_agent) = {report['v_true_agent']:.4f}",
        f"True effect      = {report['true_effect']:+.4f}",
        "",
        (
            f"{'Estimator':<10} {'V_hat':>8} {'Bias':>9} {'Rel Bias':>9} "
            f"{'plain CI':>21} {'Cov':>4} {'conformal CI':>21} {'Cov':>4} "
            f"{'Half Width':>18}"
        ),
        "-" * 104,
    ]
    for name, result in report["results"].items():
        plain = "n/a"
        plain_cov = "n/a"
        width = f"{result['conformal_half_width']:.3f}"
        if "plain_ci_low" in result:
            plain = f"[{result['plain_ci_low']:.3f},{result['plain_ci_high']:.3f}]"
            plain_cov = "yes" if result["plain_ci_covers_truth"] else "no"
            width = f"{result['plain_half_width']:.3f}->{result['conformal_half_width']:.3f}"
        conformal = (
            f"[{result['conformal_ci_low']:.3f},{result['conformal_ci_high']:.3f}]"
        )
        conformal_cov = "yes" if result["conformal_ci_covers_truth"] else "no"
        lines.append(
            f"{name:<10} {result['v_hat']:>8.4f} {result['bias']:>+9.4f} "
            f"{result['rel_bias']:>+9.2%} {plain:>21} {plain_cov:>4} "
            f"{conformal:>21} {conformal_cov:>4} {width:>18}"
        )
    return "\n".join(lines)


def _load_optional_headline(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        print(f"  Existing headline not found; skipping comparison: {path}")
        return None
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase5/conformal_sonnet46_bge_pca128_learn64.json"),
    )
    parser.add_argument(
        "--existing-headline",
        type=Path,
        default=Path("data/phase5/headline_sonnet46_bge_pca128_learn64.json"),
        help="Optional Phase 5 JSON with plain bootstrap CIs for comparison.",
    )
    parser.add_argument(
        "--score-field",
        default="y_score_claude_sonnet46",
        help="Source field to expose as y_score. Use '' to keep existing y_score.",
    )
    parser.add_argument(
        "--agent-score-field",
        default="y_agent_score_claude_sonnet46",
        help=(
            "Source field to expose as y_agent_score. Use '' to keep existing "
            "y_agent_score."
        ),
    )
    parser.add_argument(
        "--embedder",
        choices=["sbert", "openai", "medcpt", "bge", "medcpt-bge"],
        default="bge",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
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
    return parser.parse_args()


def main() -> None:
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
    print(f"  {len(records)} records loaded")

    embedder_factories = {
        "sbert": make_sbert_embedder,
        "openai": make_openai_embedder,
        "medcpt": make_medcpt_embedder,
        "bge": make_bge_m3_embedder,
        "medcpt-bge": make_medcpt_bge_embedder,
    }
    embed_fn = embedder_factories[args.embedder]()
    learned_config = LearnedEmbeddingConfig(
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

    existing_headline = _load_optional_headline(args.existing_headline)
    t0 = time.time()
    report = compute_learned_conformal_report(
        records,
        embed_fn,
        learned_config,
        alpha=args.alpha,
        seed=args.seed,
        existing_headline=existing_headline,
    )
    print()
    print(format_conformal_table(report))
    print()
    print(f"Total wall time: {time.time() - t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
