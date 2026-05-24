"""
Post-process Phase 5 headline JSONs to add BCa CIs without re-running.

Reads bootstrap_v_hat arrays stored in each headline*.json, computes BCa
CI (with z0 only since no jackknife was run), and writes a comparison
table for the embedding ablation doc.

Usage:
    .venv/bin/python scripts/recompute_bca_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cce_data.estimators.bca_ci import bca_ci


ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("OpenAI",      "headline_sonnet46_openai.json"),
    ("MedCPT",      "headline_sonnet46_medcpt.json"),
    ("BGE-M3",      "headline_sonnet46_bge.json"),
    ("MedCPT+BGE",  "headline_sonnet46_medcpt_bge.json"),
]


def main() -> None:
    rows = []
    for label, fname in RUNS:
        path = ROOT / "data" / "phase5" / fname
        if not path.exists():
            print(f"  SKIP (missing): {fname}")
            continue
        d = json.loads(path.read_text())
        v_true = d["v_true_agent"]
        for est, r in d["results"].items():
            bv = np.array(r["bootstrap_v_hat"])
            point = r["v_hat"]
            ci_lo_plain = r["ci_low"]
            ci_hi_plain = r["ci_high"]
            covers_plain = ci_lo_plain <= v_true <= ci_hi_plain

            ci_lo_bca, ci_hi_bca, diag = bca_ci(bv, point, alpha=0.05)
            covers_bca = ci_lo_bca <= v_true <= ci_hi_bca

            rows.append({
                "embedding": label,
                "estimator": est,
                "v_hat": point,
                "v_true_agent": v_true,
                "ci_plain": [ci_lo_plain, ci_hi_plain],
                "covers_plain": covers_plain,
                "ci_bca": [ci_lo_bca, ci_hi_bca],
                "covers_bca": covers_bca,
                "z0": diag["z0"],
                "p_lo": diag["p_lo"],
                "p_hi": diag["p_hi"],
                "shift_high": ci_hi_bca - ci_hi_plain,
                "gap_to_truth_plain": v_true - ci_hi_plain,
                "gap_to_truth_bca": v_true - ci_hi_bca,
            })

    # Pretty print
    print()
    print("BCa CI vs plain quantile bootstrap CI (sonnet46 judge, n_boot=100)")
    print(f"V_true(pi_agent) = {rows[0]['v_true_agent']:.4f}")
    print("=" * 110)
    print(
        f"{'Embedding':<13} {'Est':<7} {'V̂':>7} "
        f"{'plain CI':>17} {'cov':>4} "
        f"{'BCa CI':>17} {'cov':>4} "
        f"{'gap (plain→BCa)':>18} {'z0':>7}"
    )
    print("-" * 110)
    cur_emb = None
    for r in rows:
        if r["embedding"] != cur_emb:
            cur_emb = r["embedding"]
            sep_line = True
        else:
            sep_line = False
        plain = f"[{r['ci_plain'][0]:.3f},{r['ci_plain'][1]:.3f}]"
        bca = f"[{r['ci_bca'][0]:.3f},{r['ci_bca'][1]:.3f}]"
        gap_str = f"{r['gap_to_truth_plain']:+.3f}→{r['gap_to_truth_bca']:+.3f}"
        cov_p = "yes" if r["covers_plain"] else "no"
        cov_b = "✓YES" if r["covers_bca"] else "no"
        emb = r["embedding"] if sep_line else ""
        print(
            f"{emb:<13} {r['estimator']:<7} {r['v_hat']:>7.4f} "
            f"{plain:>17} {cov_p:>4} "
            f"{bca:>17} {cov_b:>4} "
            f"{gap_str:>18} {r['z0']:>+7.3f}"
        )

    # Counts
    n = len(rows)
    n_plain = sum(r["covers_plain"] for r in rows)
    n_bca = sum(r["covers_bca"] for r in rows)
    print()
    print(f"Coverage tally: plain {n_plain}/{n}   BCa {n_bca}/{n}")

    # Save JSON
    out = ROOT / "data" / "phase5" / "bca_recomputed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "tally": {"plain": n_plain, "bca": n_bca, "total": n}}, indent=2))
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
