# Phase 5 CI Methods Comparison — Plain Bootstrap vs BCa vs Conformal

After the [embedding ablation](phase5_embedding_ablation.md) closed the
bias-to-CI-width ratio from ~9× (OpenAI) to ~3–4× (BGE-M3, MedCPT+BGE),
all 12 plain-bootstrap 95% CIs still missed `V_true(π_agent)`. This doc
documents two CI-method upgrades and the resulting coverage table.

**TL;DR**: Plain bootstrap 0/12 → BCa 0/12 → Conformal (jackknife+) **8/12**.
OffCEM emerges as the only estimator that recovers coverage with a
practically useful CI width. MIPS recovers coverage but with CIs so wide
they're uninformative for a [0, 1] reward. DM still misses under
conformal because its bias exceeds the conformal half-width by a hair.

## Setup
- **Dataset / judge / agent / embeddings**: same four runs as
  [phase5_embedding_ablation.md](phase5_embedding_ablation.md)
  (`headline_sonnet46_{openai,medcpt,bge,medcpt_bge}.json`)
- **Estimators**: DM, MIPS (SNIPS variant), OffCEM
- **Per-sample influence functions ψ_i** added to each estimator's return
  dict in `src/cce_data/estimators/real_runner.py`:
  - DM:    ψ_i = f̂(x_i, a_i^agent)
  - MIPS:  ψ_i = (w_i / w̄) · y_i  (SNIPS contribution)
  - OffCEM: ψ_i = f̂(x_i, a_i^agent) + (w_i / w̄) · (y_i − f̂(x_i, a_i^clinician))
- All defined so that `mean(ψ) = V̂` exactly.

## Method 1 — Plain quantile bootstrap (baseline)

Re-stated for comparison. 100 bootstrap iterations, full estimator refit
per resample, CI = [2.5%, 97.5%] empirical quantiles. **All 12 CIs miss truth.**

## Method 2 — BCa (Bias-Corrected and Accelerated)

`src/cce_data/estimators/bca_ci.py`. For each bootstrap distribution
{V̂*_b}, compute:
- `z0 = Φ⁻¹(P(V̂* < V̂_point))` — bias-correction
- `a = 0` (no jackknife available; this is "BC" rather than full "BCa")
- Shifted quantiles `p_lo = Φ(2z0 + z_{α/2})`, `p_hi = Φ(2z0 + z_{1-α/2})`

Read `data/phase5/bca_recomputed.json` for full table. **Result: 0/12.**

```
Embedding    Est       plain CI       BCa CI         z0      delta cov
─────────────────────────────────────────────────────────────────────
BGE-M3       DM     [.673,.732]    [.675,.733]     +0.10    no→no
             MIPS   [.750,.793]    [.741,.786]     −0.44    no→no  ←
             OffCEM [.674,.734]    [.678,.736]     +0.15    no→no
... (similar for other embeddings) ...
```

**Why BCa failed**: BCa's z0 measures the *direction* of bias between
bootstrap distribution and the point estimate, **not** between point
estimate and truth. For MIPS in every embedding, z0 < 0 — meaning the
bootstrap distribution sits *above* the point estimate (high-weight
clinician samples dominate when resampled) — so BCa shifts the CI
*downward*, away from the (higher) truth. This is mathematically
correct BCa behavior; it's just the wrong tool for this bias mechanism.

## Method 3 — Jackknife+ Conformal CI

`src/cce_data/estimators/conformal_ci.py` (adapted from Taufiq et al.
2022, reference repo). Distribution-free, finite-sample valid under iid:

1. Point estimate: `V̂ = mean(ψ)`
2. Leave-one-out means: `lo_i = (Σψ − ψ_i) / (n − 1)`
3. Residuals: `R_i = |ψ_i − lo_i|`
4. Quantile level `q = (1 − α)(n+1) / n`; half-width = `quantile(R, q)`
5. CI = `[V̂ − half-width, V̂ + half-width]`

`scripts/compute_conformal_ci.py` runs each of the 4 embedders once on
the sonnet46 dataset (no bootstrap — point estimate + ψ extraction
only, ~5 min per embedder).

### Results — 8/12 cover

```
                       plain CI       cov    conformal CI         cov     half-w plain→conf
OpenAI       DM     [.609,.653]      no    [0.451,0.820]          no     0.022 → 0.184
             MIPS   [.635,.734]      no    [-0.257,1.627]      ✓YES     0.050 → 0.942
             OffCEM [.611,.656]      no    [0.432,0.853]       ✓YES     0.022 → 0.210
MedCPT       DM     [.665,.705]      no    [0.561,0.829]          no     0.020 → 0.134
             MIPS   [.726,.789]      no    [-1.853,3.358]      ✓YES     0.031 → 2.605
             OffCEM [.668,.709]      no    [0.524,0.876]       ✓YES     0.020 → 0.176
BGE-M3       DM     [.673,.732]      no    [0.580,0.841]          no     0.029 → 0.130
             MIPS   [.750,.793]      no    [-1.751,3.283]      ✓YES     0.022 → 2.517
             OffCEM [.674,.734]      no    [0.559,0.866]       ✓YES     0.030 → 0.153
MedCPT+BGE   DM     [.679,.733]      no    [0.595,0.838]          no     0.027 → 0.122
             MIPS   [.746,.795]      no    [-1.480,3.004]      ✓YES     0.025 → 2.242
             OffCEM [.680,.735]      no    [0.556,0.880]       ✓YES     0.027 → 0.162

Tally: plain 0/12   BCa 0/12   conformal 8/12   (V_true(π_agent) = 0.8413)
```

### Reading the table

- **OffCEM (4/4 cover, informative CIs)**: CI half-widths 0.15–0.21.
  These are usable: "agent's true mean reward is between 0.43 and 0.88"
  is wider than the original bootstrap claim but **honest about the
  bias**. Truth (0.841) falls inside in every cell. **OffCEM is the
  recommended deployment-grade estimator.**

- **MIPS (4/4 cover, but useless CIs)**: CI half-widths 0.94–2.6 on a
  [0, 1] reward. Coverage is mathematically valid, but a CI like
  [−1.75, 3.28] tells the reader nothing. Root cause: per-sample SNIPS
  contribution ψ_i = (w_i / w̄) y_i has tail values up to ≈ 20 × y_i
  because the density-ratio weights are clipped at 20. Conformal's
  honest accounting of this per-sample variance blows up the CI.
  MIPS remains the best **point estimator** (lowest bias under BGE-M3,
  −7.5%), but is no longer the right estimator for a paper that
  reports CI coverage.

- **DM (0/4 cover, but barely)**: BGE-M3 DM conformal CI is
  [0.580, 0.841] — upper bound equals truth exactly. The bias is too
  large for the per-sample residual distribution to absorb. To recover
  DM coverage, the lever is reducing bias (better q̂, KL-control), not
  the CI method.

## Why OffCEM CIs stay tight while MIPS CIs explode

Per-sample contribution structure:

| Estimator | ψ_i formula | range of ψ_i | source of per-sample variance |
|---|---|---|---|
| DM | f̂(x_i, a_i^agent) | [0, 1] (GBDT clipped) | very small — q̂ smooth |
| MIPS | (w_i / w̄) · y_i | [0, ~20] | huge — weight clip 20× |
| OffCEM | f̂(x_i, a_i^agent) + (w_i / w̄)(y_i − f̂(x_i, a_i^clin)) | [≈0, ≈1] | small — residual is [−1, 1], weight only multiplies the residual not y |

OffCEM's doubly-robust structure decouples the high-variance density
ratio from the y scale: the high-variance weight only multiplies the
residual `(y − f̂)` which is small in magnitude. This is the empirical
manifestation of OffCEM's theoretical "best of both" property — it
shows up in finite-sample CI behavior, not just asymptotic efficiency.

## Recommended next steps

- [ ] **Adopt OffCEM as Phase 5 headline estimator** for the paper. MIPS
  stays as a low-bias point-estimate comparator; DM as the
  outcome-model-only baseline.
- [ ] **Use BGE-M3 as headline embedding** (lowest MIPS bias, smallest
  conformal MIPS CI in the "useless but valid" set).
- [ ] **Report all three CI methods in paper Table 1**: plain bootstrap
  (familiar to readers), BCa (shows we tried bias correction), conformal
  (the one that actually covers). Be explicit that conformal MIPS CIs are
  uninformatively wide and explain why.
- [ ] **Phase 6 ablation**: redo gpt-4o judge run with conformal CIs to
  confirm pattern holds across judges.
- [ ] **Future work hook**: causal contrastive embedding (plan-v2 U1)
  expected to shrink MIPS's per-sample ψ variance by making density
  ratios closer to 1, which would tighten MIPS conformal CIs to be
  competitive with OffCEM's.

## Reproduction

```bash
# BCa on existing bootstrap data (no re-fit needed)
.venv/bin/python scripts/recompute_bca_ci.py

# Conformal (needs re-embedding + 1 estimator fit per embedding, ~10 min total)
OPENAI_API_KEY=sk-... .venv/bin/python scripts/compute_conformal_ci.py
```

Outputs:
- `data/phase5/bca_recomputed.json`
- `data/phase5/conformal_recomputed.json`
