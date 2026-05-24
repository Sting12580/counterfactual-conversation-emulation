# Phase 5 Embedding Ablation — OpenAI vs MedCPT vs BGE-M3 vs Concat (Sonnet 4.6 judge)

Four-way embedding ablation testing whether medical-specific or
multilingual embeddings improve CI coverage relative to the OpenAI
generic embedding (text-embedding-3-small, 1536-d) on the
[Sonnet 4.6 judge baseline](phase5_sonnet46_judge_ablation.md). Motivated
by the observation that all six CIs (gpt-4o × {DM, MIPS, OffCEM} and
sonnet-4-6 × same three) missed `V_true(π_agent)` because estimator bias
exceeded sampling variance by 3–9×.

## Setup
- **Dataset**: `data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl` (397 included)
- **Judge**: `anthropic:claude-sonnet-4-6`
- **Behavior policy** π_b: clinician final assessment + plan extracted from notes
- **Target policy** π_agent: `openai:gpt-4.1`
- **Embedders under test**:
  - OpenAI `text-embedding-3-small` (1536-d) — generic baseline
  - `ncbi/MedCPT-Article-Encoder` (768-d) — medical-specific, mean-pool + L2-normalize
  - `BAAI/bge-m3` (1024-d, dense) — multilingual long-text, mean-pool + L2-normalize
  - MedCPT + BGE-M3 concat (1792-d) — plan-v2's locked-in fusion
- **Estimators**: DM (Jaques 2019), MIPS (Saito & Joachims 2022), OffCEM (Saito 2023)
- **Featurization**: `[phi(x) ; phi(a) ; phi(x) * phi(a)]`
- **Bootstrap**: 100 iterations with full refit per resample
- **Classifier**: LogisticRegression (C=0.1) + CalibratedClassifierCV sigmoid Platt cv=5. **No StandardScaler** — see Caveats.

## Ground truth (unchanged from sonnet46 baseline)

| Quantity | Value |
|---|---|
| V_true(π_b) | 0.6178 |
| V_true(π_agent) | 0.8413 |
| True effect | **+0.2236** (95% CI [0.206, 0.241]) |

## Headline — four embeddings side by side

```
                 OpenAI (4608)            MedCPT (2304)             BGE-M3 (3072)           MedCPT+BGE (5376)
Est.        V̂    RelBias  RMSE  Dir    V̂    RelBias  RMSE  Dir    V̂    RelBias  RMSE  Dir    V̂    RelBias  RMSE  Dir
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DM        0.636  -24.4%  0.211  91%   0.695  -17.4%  0.158 100%   0.711  -15.5%  0.135 100%   0.716  -14.9%  0.134 100%
MIPS      0.685  -18.6%  0.153  99%   0.753  -10.6%  0.085 100%   0.766   -9.0%  0.071 100%★  0.762   -9.4%  0.072 100%
OffCEM    0.643  -23.6%  0.209  92%   0.700  -16.8%  0.155 100%   0.712  -15.3%  0.134 100%   0.718  -14.6%  0.133 100%

V_true(π_agent) = 0.8413 (none of the twelve CIs covers truth)
Feature dim = embedding dim × 3 (phi_x, phi_a, phi_x*phi_a concat)
```

| Embedding | Wall time | feat dim | Notes |
|---|---|---|---|
| OpenAI | 152 min | 4608 | Baseline from [sonnet46 ablation](phase5_sonnet46_judge_ablation.md) |
| MedCPT | 82 min | 2304 | `headline_sonnet46_medcpt.json` |
| **BGE-M3** | 109 min | 3072 | `headline_sonnet46_bge.json` — **lowest MIPS bias** |
| MedCPT+BGE concat | 183 min | 5376 | `headline_sonnet46_medcpt_bge.json` — plan-v2 locked-in |

## Bias-to-CI-width diagnostic (for coverage analysis)

CI coverage fails iff `|bias| > CI half-width`. Ratios below show why:

| Embedding | DM ratio | MIPS ratio | OffCEM ratio | MIPS CI upper vs truth |
|---|---|---|---|---|
| OpenAI | 9.27× | 3.16× | 8.97× | 0.107 short |
| MedCPT | 7.39× | **2.83×** | 7.02× | 0.052 short |
| BGE-M3 | 4.47× | 3.49× | 4.34× | 0.048 short |
| MedCPT+BGE concat | 4.57× | 3.20× | 4.52× | 0.046 short |

**Reading**: MedCPT has the best bias-to-half-width *ratio* for MIPS (2.83×),
but BGE-M3 has the smallest *absolute distance* from CI upper to truth
(0.048 vs MedCPT's 0.052). MedCPT's CI is wider (half-width 0.031 vs BGE's
0.022), so even though it has more bias, more of that bias is "absorbed"
by sampling variance. For pure point-estimate accuracy, BGE-M3 wins; for
coverage-via-bias-aware-CI methods (BCa / split-conformal), MedCPT is
slightly better positioned because its larger half-width gives bias
correction more room to shift the CI.

## Key findings

1. **BGE-M3 wins on point-estimate accuracy**, by a margin.
   - MIPS bias: −0.075 (BGE) vs −0.089 (MedCPT) vs −0.079 (concat) vs −0.156 (OpenAI). **52% relative reduction vs OpenAI**.
   - DM bias: −0.131 (BGE) vs −0.146 (MedCPT) vs −0.125 (concat) vs −0.205 (OpenAI). **36% relative reduction vs OpenAI**.
   - OffCEM mirrors DM. The win is monotone: BGE strictly beats MedCPT on every estimator's bias.

2. **Concat does NOT outperform BGE-M3 alone**, contrary to expectation.
   The plan-v2 default (MedCPT + BGE-M3 concat) was hypothesized to give
   "best of both" by combining medical-domain (MedCPT) and bilingual
   long-context (BGE) signal. Empirically, on MIPS the concat is *slightly
   worse* than BGE alone (−9.4% vs −9.0% rel bias); on DM/OffCEM it is
   marginally better (~0.5% rel bias improvement). **For ~3× the runtime
   cost (183 min vs 109 min) and 75% larger feature dim, the headline
   estimator MIPS does not gain.** Recommendation: drop the concat from
   plan-v2's locked-in choice and adopt BGE-M3 alone as the headline.

3. **DM/OffCEM bias improves dramatically with non-OpenAI embeddings.**
   Their bias-to-half-width ratio drops from ~9× (OpenAI) to ~4.5× (BGE
   or concat). Still misses coverage, but **the binding constraint
   shifts**: with OpenAI, DM was effectively useless for CI work; with
   BGE, it is within striking distance if combined with bias-aware CI.

4. **All twelve CIs still miss `V_true(π_agent)`.** Embedding choice
   alone is insufficient for CI coverage. The smallest gap any
   estimator-embedding combo achieves is **MIPS + concat (0.046 short)**
   or **MIPS + BGE (0.048 short)**. To close this gap one needs a
   bias-aware CI method (BCa is the cheapest; split-conformal gives a
   formal coverage guarantee). Plan-v2's choice of plain quantile
   bootstrap is the binding constraint, not the embedding.

5. **Direction agreement is uniformly 100% across all 12 cells under
   non-OpenAI embeddings.** For deployment decisions (which agent ships),
   any of MedCPT / BGE / concat with any of DM / MIPS / OffCEM is
   sufficient. This metric is now saturated; further embedding work will
   not improve it.

6. **Runtime ordering matches feature dimensionality.** MedCPT (2304-d
   feat, 82 min) < BGE (3072-d, 109 min) < OpenAI (4608-d, 152 min) <
   concat (5376-d, 183 min). For Phase 6 ablations with many cells,
   **BGE-M3 is the right operating point**: best bias, mid runtime.

## Caveats

- **Classifier fix that didn't pan out.** Plan originally proposed wrapping
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

- **Concat hypothesis was wrong.** I initially predicted the concat
  (5376-d feat) would *hurt* DM/OffCEM via small-n overfit on higher
  dimensionality. Empirically it does no worse than BGE alone on
  DM/OffCEM, and marginally worse on MIPS. The relevant constraint at
  n=397 is not "more features → more overfit" but "more features →
  more noise in the classifier's density-ratio output." MedCPT's
  medical-specific signal does not add useful information beyond what
  BGE-M3 already encodes for clinician-text vs agent-text discrimination.

- **No ESS or weight-tail diagnostics reported.** We can't fully
  decompose how much of each embedding's bias-drop came from improved
  positivity vs improved NDE vs reduced classifier noise. Adding ESS to
  the per-run JSON output is a Phase 6 follow-up that should land before
  the paper draft.

- **BGE-M3 was trained for retrieval, not OPE.** Like MedCPT, BGE-M3 is
  a retrieval embedding (multilingual, dense + sparse + colbert). We use
  only the dense output. Causal-contrastive fine-tuning (plan-v2 U1)
  would be the principled upgrade for paper claims, with this run as
  the baseline.

- **Hugging Face cache & MPS.** All HF embedders load on Apple Silicon
  MPS by default. First run downloads from HF (~1 GB MedCPT, ~2.3 GB
  BGE-M3); subsequent runs are fast.

## What still needs to run

- [x] Phase 6 ablation #1: embedding choice (MedCPT) — completed 2026-05-22, ~82 min.
- [x] Phase 6 ablation #1 follow-up: BGE-M3 alone — completed 2026-05-23, ~109 min. **Lowest MIPS bias.**
- [x] Phase 6 ablation #1 follow-up: MedCPT + BGE-M3 concat — completed 2026-05-23, ~183 min. **No improvement over BGE alone.**
- [ ] Phase 6 ablation #3: positivity diagnostics — emit ESS, weight tail histogram per run.
- [ ] CI method upgrade: BCa bootstrap (cheap, scipy.stats.norm) — first try at coverage.
- [ ] CI method upgrade: split-conformal CI on per-sample influence functions; integrate from reference repo `src/ccema/estimators/conformal_ci.py` — formal coverage guarantee.

## Reproduction

```bash
# Make sure transformers + torch are installed
.venv/bin/python -c "import transformers, torch; print(transformers.__version__, torch.__version__)"

# Run each embedder (HF weights download on first call)
.venv/bin/python scripts/run_phase5.py \
    --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
    --output data/phase5/headline_sonnet46_medcpt.json \
    --embedder medcpt --n-boot 100 --seed 0

.venv/bin/python scripts/run_phase5.py \
    --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
    --output data/phase5/headline_sonnet46_bge.json \
    --embedder bge --n-boot 100 --seed 0

.venv/bin/python scripts/run_phase5.py \
    --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
    --output data/phase5/headline_sonnet46_medcpt_bge.json \
    --embedder medcpt-bge --n-boot 100 --seed 0
```

Outputs land under `data/phase5/` (gitignored). Compare to ground-truth
at `data/phase3/ground_truth_effect_judge_claude_sonnet46.json` and to
OpenAI baseline at `data/phase5/headline_sonnet46_openai.json` (gitignored;
table in [phase5_sonnet46_judge_ablation.md](phase5_sonnet46_judge_ablation.md)).
