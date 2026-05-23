# Phase 5 Embedding Ablation — MedCPT vs OpenAI (Sonnet 4.6 judge)

Embedding ablation testing whether a medical-specific, lower-dim
embedding (MedCPT, 768-d) improves CI coverage relative to the OpenAI
generic embedding (text-embedding-3-small, 1536-d) on the
[Sonnet 4.6 judge baseline](phase5_sonnet46_judge_ablation.md). Motivated
by the observation that all six CIs (gpt-4o × {DM, MIPS, OffCEM} and
sonnet-4-6 × same three) missed `V_true(π_agent)` because estimator bias
exceeded sampling variance by 3–9×.

## Setup
- **Dataset**: `data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl` (397 included; same as the Sonnet 4.6 baseline)
- **Judge**: `anthropic:claude-sonnet-4-6`
- **Behavior policy** π_b: clinician final assessment + plan extracted from notes
- **Target policy** π_agent: `openai:gpt-4.1`
- **Embedder under test**: `ncbi/MedCPT-Article-Encoder` (768-d, mean-pool + L2-normalize, MPS-accelerated)
- **Baseline embedder**: OpenAI `text-embedding-3-small` (1536-d, from [phase5_sonnet46_judge_ablation.md](phase5_sonnet46_judge_ablation.md))
- **Estimators**: DM (Jaques 2019), MIPS (Saito & Joachims 2022), OffCEM (Saito 2023)
- **Featurization**: `[phi(x) ; phi(a) ; phi(x) * phi(a)]` (→ 2304-d for MedCPT vs 4608-d for OpenAI)
- **Bootstrap**: 100 iterations with full refit per resample
- **Classifier**: LogisticRegression (C=0.1) + CalibratedClassifierCV sigmoid Platt cv=5. **No StandardScaler** — see Caveats.

## Ground truth (unchanged from sonnet46 baseline)

| Quantity | Value |
|---|---|
| V_true(π_b) | 0.6178 |
| V_true(π_agent) | 0.8413 |
| True effect | **+0.2236** (95% CI [0.206, 0.241]) |

## Result — MedCPT embedding

`data/phase5/headline_sonnet46_medcpt.json` — 100-bootstrap run (~82 min wall time).

| Estimator | V̂ | Bias | Rel Bias | RMSE | 95% CI | Cov | Dir % | bias / CI half-width |
|---|---|---|---|---|---|---|---|---|
| DM | 0.6951 | −0.1462 | −17.38% | 0.1577 | [0.665, 0.705] | no | 100.0% | 7.39× |
| MIPS | 0.7525 | **−0.0888** | **−10.56%** ✓ | **0.0846** | **[0.726, 0.789]** | no | 100.0% | **2.83×** |
| OffCEM | 0.7000 | −0.1413 | −16.79% | 0.1545 | [0.668, 0.709] | no | 100.0% | 7.02× |

**Plan v2 success criterion** (rel bias < 20% AND direction correct):
**all three estimators PASS** under MedCPT (vs only MIPS passing under OpenAI
with the same judge). MIPS is again the strongest estimator on every
column. Coverage still fails because bias > sampling SE, but MIPS's
upper CI bound (0.789) is now within **0.052** of `V_true(π_agent) = 0.841`
— closer than any prior estimator-embedding combination on this judge.

## Side-by-side: OpenAI vs MedCPT (both with Sonnet 4.6 judge)

```
                      OpenAI (1536-d → 4608-d feat)              MedCPT (768-d → 2304-d feat)
Estimator     V̂    Rel Bias  RMSE    Dir %       V̂    Rel Bias  RMSE    Dir %
─────────────────────────────────────────────────────────────────────────────────────────
DM          0.636  -24.4%   0.211    91.0%      0.695  -17.4%   0.158   100.0%
MIPS        0.685  -18.6%   0.153    99.0%      0.753  -10.6%   0.085   100.0%   ★ best
OffCEM      0.643  -23.6%   0.209    92.0%      0.700  -16.8%   0.155   100.0%

V_true(π_agent) = 0.8413                  V_true(π_agent) = 0.8413
                                          (none of the six CIs covers truth)
```

## Key findings

1. **MedCPT cuts MIPS bias by 43% relative** (−0.156 → −0.089). DM and
   OffCEM both improve by ~29%. The improvement is monotone across all
   three estimators, suggesting the embedding swap helps both the
   outcome-model extrapolation (DM/OffCEM) and the density-ratio
   estimation (MIPS) — i.e., both bias channels were partially driven
   by OpenAI's generic-text-vs-medical-text mismatch.

2. **MIPS direction rate jumps from 99% → 100%.** Direction agreement
   was already near-perfect under OpenAI; MedCPT pushes it to certainty.
   DM direction rate also jumps 91% → 100%. **For deployment decisions
   (which agent to ship), MedCPT + MIPS is unambiguous.**

3. **MIPS upper CI bound is now within 0.052 of `V_true(π_agent)`**
   ([0.726, 0.789] vs truth 0.841). Under OpenAI this gap was 0.107.
   The CI did not move much in width (0.063 vs 0.099) — the shift came
   almost entirely from a higher point estimate. This is consistent with
   "embedding fixed a chunk of the systematic bias, didn't change the
   sampling variance."

4. **Coverage still fails, but the binding constraint shifted from bias
   to method.** Under MedCPT, MIPS's bias-to-CI-half-width ratio is 2.83×
   (down from 3.1× under OpenAI). At this ratio, a bias-corrected
   bootstrap (BCa) or split-conformal CI plausibly closes the coverage
   gap; pure quantile bootstrap cannot. **Embedding alone won't deliver
   coverage; embedding + non-bias-naive CI method might.** This is the
   recommended next iteration.

5. **DM/OffCEM remain limited by `q̂` extrapolation, not embedding.**
   Their bias-to-half-width ratios are still 7–7.4× — embedding cut bias
   ~29% but variance also dropped (lower-dim feature → less GBDT noise),
   so the ratio barely moved. To improve DM/OffCEM coverage, the right
   lever is outcome-model regularization or KL-control implementation,
   not further embedding work.

6. **MedCPT runs faster.** 82 min vs OpenAI's 152 min — same n_boot=100,
   smaller feature dim (2304 vs 4608) → GBDT splits ~half as expensive.
   For Phase 6 ablations (multiple embeddings × multiple judges),
   MedCPT-class embedders give meaningful budget savings.

## Caveats

- **Classifier fix that didn't pan out**. Plan originally proposed wrapping
  the density-ratio classifier in `Pipeline(StandardScaler,
  LogisticRegression)` with tighter regularization to suppress thousands
  of LBFGS matmul warnings on the high-dim OpenAI features. Two pilots
  (C=1.0 and C=10.0 with StandardScaler) showed the scaler **destroys
  the density-ratio signal**: MIPS bias under the "fixed" pipeline was
  **worse** than baseline (−0.21 vs −0.16), with V̂ collapsing to ≈ V_b.
  Root cause: L2-normalized embeddings (OpenAI/MedCPT/BGE-M3 all
  normalize) live on the unit sphere, so feature-wise standardization
  shifts rows off the sphere and breaks the geometric structure the
  linear classifier relies on. The fix was reverted; warnings are now
  suppressed via `warnings.catch_warnings()` in
  `fit_density_ratio_classifier`. Reverted defaults: C=0.1, no scaler.

- **MedCPT positivity unknown**. We don't yet report ESS or weight-tail
  diagnostics per run, so we can't quantify how much of the bias drop
  came from better positivity vs. better NDE vs. simply lower-dim
  noise. Adding ESS to the headline JSON is a Phase 6 follow-up.

- **MedCPT was tuned for retrieval, not embedding-space density ratio**.
  MedCPT was trained as a query/article encoder for PubMed retrieval.
  The fact that it transfers cleanly to clinician-note OPE is encouraging
  but not theoretically guaranteed. A causal-contrastive fine-tune
  (plan-v2 U1) would be the principled upgrade for paper claims.

- **Hugging Face cache & MPS**. MedCPT loads on Apple Silicon MPS by
  default (~1 GB weights, 199 sharded files). First run downloads from
  HF; subsequent runs are fast.

## What still needs to run

- [x] Phase 6 ablation #1: embedding choice (MedCPT) — completed 2026-05-22, ~82 min. **MIPS bias reduced 43%.**
- [ ] Phase 6 ablation #1 follow-up: BGE-M3 alone (1024-d → 3072-d feat) — bilingual general-text alternative to MedCPT.
- [ ] Phase 6 ablation #1 follow-up: MedCPT + BGE-M3 concat (1792-d → 5376-d feat) — plan-v2's locked-in default; tests whether information gain outweighs dim-overfit.
- [ ] Phase 6 ablation #3: positivity diagnostics — emit ESS, weight tail histogram per run.
- [ ] CI method upgrade: split-conformal CI on per-sample influence functions; integrate from reference repo `src/ccema/estimators/conformal_ci.py`.
- [ ] BCa bootstrap CI as second-line bias-aware CI (cheap addition; one scipy import).

## Reproduction

```bash
# 1. Make sure transformers + torch are installed (already in pyproject)
.venv/bin/python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"

# 2. Run phase5 with MedCPT embedder (downloads weights on first run)
.venv/bin/python scripts/run_phase5.py \
    --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
    --output data/phase5/headline_sonnet46_medcpt.json \
    --embedder medcpt --n-boot 100 --seed 0

# 3. Other new embedders (deferred runs):
.venv/bin/python scripts/run_phase5.py --embedder bge --input ...      # BGE-M3 alone
.venv/bin/python scripts/run_phase5.py --embedder medcpt-bge --input ... # concat
```

Outputs land under `data/phase5/` (gitignored). Compare to ground-truth
at `data/phase3/ground_truth_effect_judge_claude_sonnet46.json` and to
OpenAI baseline at `data/phase5/headline_sonnet46_openai.json` (gitignored;
table in [phase5_sonnet46_judge_ablation.md](phase5_sonnet46_judge_ablation.md)).
