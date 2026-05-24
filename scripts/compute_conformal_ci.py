"""Compute jackknife+ conformal CIs for all 4 embeddings, sonnet46 judge.

For each embedding:
  1. Load + embed sonnet46 dataset
  2. Fit DM, MIPS, OffCEM once on full data; extract per-sample psi
  3. Apply jackknife_plus_ci(psi) for finite-sample distribution-free CI
  4. Compare to existing plain-bootstrap CI from headline_sonnet46_*.json

Usage:
    OPENAI_API_KEY=... .venv/bin/python scripts/compute_conformal_ci.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase5 import (
    load_jsonl,
    make_bge_m3_embedder,
    make_medcpt_bge_embedder,
    make_medcpt_embedder,
    make_openai_embedder,
)
from cce_data.estimators.conformal_ci import jackknife_plus_ci
from cce_data.estimators.real_runner import (
    dm_real, mips_real, offcem_real, featurize_records,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "phase3" / "agent_scored_all_judge_claude_sonnet46.jsonl"
PHASE5 = ROOT / "data" / "phase5"


def run_one(label: str, fname_existing: str, embed_factory):
    print(f"\n--- {label} ---")
    print(f"  Loading {DATA} ...")
    records = load_jsonl(DATA)
    print(f"  {len(records)} records")

    print(f"  Building embedder for {label} ...")
    t0 = time.time()
    embed_fn = embed_factory()
    data = featurize_records(records, embed_fn)
    print(f"  Embedded in {time.time() - t0:.0f}s; phi_x dim = {data.phi_x.shape[1]}")

    existing = json.loads((PHASE5 / fname_existing).read_text())
    v_true = existing["v_true_agent"]
    results = {}

    for est_name, est in [("DM", dm_real), ("MIPS", mips_real), ("OffCEM", offcem_real)]:
        t0 = time.time()
        out = est(data, seed=0)
        psi = np.asarray(out["psi"])
        # Sanity: mean(psi) should equal v_hat
        assert abs(psi.mean() - out["v_hat"]) < 1e-6, f"psi mean mismatch for {est_name}"
        point, lo, hi = jackknife_plus_ci(psi, alpha=0.05)
        covers = lo <= v_true <= hi

        plain = existing["results"][est_name]
        plain_lo, plain_hi = plain["ci_low"], plain["ci_high"]
        plain_covers = plain_lo <= v_true <= plain_hi

        results[est_name] = {
            "v_hat": float(out["v_hat"]),
            "plain_ci": [plain_lo, plain_hi],
            "plain_covers": plain_covers,
            "conformal_ci": [lo, hi],
            "conformal_covers": covers,
            "conformal_half_width": (hi - lo) / 2,
            "plain_half_width": (plain_hi - plain_lo) / 2,
            "fit_time_s": time.time() - t0,
        }
        print(
            f"  {est_name:7s} v_hat={out['v_hat']:.4f} "
            f"plain=[{plain_lo:.3f},{plain_hi:.3f}] cov={plain_covers} "
            f"conformal=[{lo:.3f},{hi:.3f}] cov={covers} "
            f"half_width: {(plain_hi-plain_lo)/2:.3f} -> {(hi-lo)/2:.3f}"
        )

    return {"v_true_agent": v_true, "label": label, "results": results}


def main():
    runs = [
        ("OpenAI",     "headline_sonnet46_openai.json",     make_openai_embedder),
        ("MedCPT",     "headline_sonnet46_medcpt.json",     make_medcpt_embedder),
        ("BGE-M3",     "headline_sonnet46_bge.json",        make_bge_m3_embedder),
        ("MedCPT+BGE", "headline_sonnet46_medcpt_bge.json", make_medcpt_bge_embedder),
    ]
    all_results = []
    for label, existing, factory in runs:
        try:
            all_results.append(run_one(label, existing, factory))
        except Exception as e:
            print(f"  FAILED {label}: {e}")
            raise

    out_path = PHASE5 / "conformal_recomputed.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    print(f"\nWrote {out_path.relative_to(ROOT)}")

    # Summary table
    print("\n" + "=" * 110)
    print("Conformal (jackknife+) vs plain bootstrap CI — sonnet46 judge, n=397, alpha=0.05")
    print(f"V_true(pi_agent) = {all_results[0]['v_true_agent']:.4f}")
    print("=" * 110)
    print(
        f"{'Embedding':<13} {'Est':<7} {'V̂':>7} "
        f"{'plain CI':>17} {'cov':>4} "
        f"{'conformal CI':>17} {'cov':>5} "
        f"{'half_w plain→conformal':>26}"
    )
    print("-" * 110)
    plain_total = bca_total = conformal_total = 0
    n = 0
    for run in all_results:
        for est, r in run["results"].items():
            n += 1
            if r["plain_covers"]: plain_total += 1
            if r["conformal_covers"]: conformal_total += 1
            cov_p = "yes" if r["plain_covers"] else "no"
            cov_c = "✓YES" if r["conformal_covers"] else "no"
            print(
                f"{run['label']:<13} {est:<7} {r['v_hat']:>7.4f} "
                f"[{r['plain_ci'][0]:.3f},{r['plain_ci'][1]:.3f}]   {cov_p:>4} "
                f"[{r['conformal_ci'][0]:.3f},{r['conformal_ci'][1]:.3f}]   {cov_c:>5} "
                f"{r['plain_half_width']:>10.3f} → {r['conformal_half_width']:.3f}"
            )
    print()
    print(f"Coverage tally: plain {plain_total}/{n}  conformal {conformal_total}/{n}")


if __name__ == "__main__":
    main()
