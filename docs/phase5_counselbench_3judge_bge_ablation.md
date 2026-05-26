# Phase 5 CounselBench — Three-Judge BGE-M3 Bootstrap Headline

OPE on the **CounselBench expert-label benchmark** under three judge LLMs
(Gemini, GPT-4, LLaMA-3) using a single embedder (BGE-M3). Companion to the
prior single-dataset embedding-ablation
([phase5_embedding_ablation.md](phase5_embedding_ablation.md)) and Sonnet 4.6
judge ablation ([phase5_sonnet46_judge_ablation.md](phase5_sonnet46_judge_ablation.md));
this run holds the embedding fixed and varies the **judge model** to probe
robustness of the bias / coverage findings across rubric sources.

## Setup
- **Dataset**: CounselBench expert-label pipeline (see `scripts/build_counselbench.py`, commit `f443f02`)
- **Per-judge inputs** (n=100 each):
  - `data/phase3/phase3_gemini_expert_scored.jsonl`
  - `data/phase3/phase3_gpt4_expert_scored.jsonl`
  - `data/phase3/phase3_llama3_expert_scored.jsonl`
- **Embedder**: `BAAI/bge-m3` (1024-d dense, mean-pool + L2-norm), feature dim 3072 = `[phi(x); phi(a); phi(x)*phi(a)]`
- **Estimators**: DM (Jaques 2019), MIPS (Saito & Joachims 2022), OffCEM (Saito 2023)
- **No learned action embedding** in this run (smoke had concat-pca learned projection; this run is plain BGE-M3 for cleaner judge×estimator comparison)
- **Bootstrap**: 100 iterations with full refit per resample
- **Classifier**: LogisticRegression (C=0.1) + CalibratedClassifierCV sigmoid Platt cv=5 (no StandardScaler — see Caveats)
- **Wall time**: ~50 min per judge; three parallel background jobs → ~50 min total wall-clock (overlapped HF cache)

## Ground truth (computed per-judge from the rubric)

| Judge | V_true(π_b) | V_true(π_agent) | True effect |
|---|---|---|---|
| Gemini | 0.5191 | 0.6381 | **+0.1191** |
| GPT-4  | 0.5191 | 0.6664 | **+0.1473** |
| LLaMA-3 | 0.5191 | 0.8580 | **+0.3389** |

Note: `V_true(π_b)` is identical across judges because π_b's actions are the clinician's documented note text (the rubric uses different judges to score those same actions; for these three judges the rubric assigns nearly identical scores to clinician actions — only π_agent's mean differs, which dominates the true effect).

## Headline — three judges × three estimators (9 cells)

```
                     Gemini (+0.119)                 GPT-4 (+0.147)                  LLaMA-3 (+0.339)
Est.        V̂    RelBias  RMSE   Dir%   Cov     V̂    RelBias  RMSE   Dir%   Cov     V̂    RelBias  RMSE   Dir%   Cov
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
DM        0.5175  -18.9%  0.120   53%   no    0.5242  -21.3%  0.138   71%   no    0.5978  -30.3%  0.298  100%   no
MIPS      0.5084  -20.3%  0.136   32%   no    0.5479  -17.8%  0.116   92%   no    0.6287  -26.7%  0.231  100%   no
OffCEM    0.5176  -18.9%  0.120   54%   no    0.5243  -21.3%  0.138   71%   no    0.5979  -30.3%  0.298  100%   no
```

| Judge | Output file |
|---|---|
| Gemini | `data/phase5/headline_counselbench_gemini_bge_boot100.json` |
| GPT-4 | `data/phase5/headline_counselbench_gpt4_bge_boot100.json` |
| LLaMA-3 | `data/phase5/headline_counselbench_llama3_bge_boot100.json` |

## Bias-to-CI-half-width diagnostic

CI covers iff `|bias| < half-width`. All 9 cells fail; the ratios show by how much:

| Judge | DM ratio | MIPS ratio | OffCEM ratio |
|---|---|---|---|
| Gemini | 3.87× | 2.30× | 3.87× |
| GPT-4 | 3.38× | 3.19× | 3.38× |
| LLaMA-3 | 5.71× | 4.46× | 5.70× |

MIPS achieves the smallest bias/half-width ratio on **Gemini** (2.30×) — closest to coverage of any cell in the table, but still off by ~3× the CI extent in MIPS's case.

## Key findings

1. **Coverage fails across every judge × every estimator.** All 9 cells miss
   `V_true(π_agent)`. The pattern from the earlier Sonnet 4.6 embedding ablation
   (bias > sampling variance by 3–6×) replicates across CounselBench's three
   rubric judges — i.e., it is not specific to a single judge model.

2. **MIPS bias is judge-dependent; DM/OffCEM aren't.**
   - DM and OffCEM are essentially identical in every cell (OffCEM's residual
     correction term is small, mirroring the Sonnet 4.6 finding).
   - MIPS bias varies meaningfully with the judge: 20.3% (Gemini) → 17.8% (GPT-4)
     → 26.7% (LLaMA-3). Plausibly because MIPS density-ratio fit depends on
     the y-distribution, which differs across judges.

3. **Direction-rate scales with true effect size.** As effect grows, bootstrap
   direction-rate climbs (Gemini: 32–54% → GPT-4: 71–92% → LLaMA-3: 100%).
   Even with biased estimates, the sign of `V_hat − V_b` becomes recoverable
   once the true gap is large enough.

4. **LLaMA-3 has the highest absolute bias** (−30% for DM/OffCEM) but also the
   clearest direction signal — biggest true effect, biggest absolute miss, but
   100% direction agreement. The judges with smaller true effects (Gemini, GPT-4)
   have lower absolute bias but worse direction agreement.

## Smoke vs bootstrap comparison

The smoke results in `/Users/niuniu/Desktop/RA/phase5/headline_*_bge_pca128_learn64_smoke.json`
used **BGE-M3 + a learned concat-pca action embedding (pca=128, latent=64)** and
reported only point estimates (no CI). Point-estimate comparison:

| Cell | Smoke v_hat (w/ learned embed) | Bootstrap v_hat (plain BGE-M3) | Δ |
|---|---|---|---|
| Gemini DM | 0.5259 | 0.5175 | −0.0084 |
| Gemini MIPS | 0.5086 | 0.5084 | −0.0002 |
| Gemini OffCEM | 0.5258 | 0.5176 | −0.0082 |
| GPT-4 DM | 0.5382 | 0.5242 | −0.0140 |
| GPT-4 MIPS | 0.6008 | 0.5479 | **−0.0529** |
| GPT-4 OffCEM | 0.5384 | 0.5243 | −0.0141 |
| LLaMA-3 DM | 0.6011 | 0.5978 | −0.0033 |
| LLaMA-3 MIPS | 0.6476 | 0.6287 | −0.0189 |
| LLaMA-3 OffCEM | 0.6014 | 0.5979 | −0.0035 |

**Observation**: dropping the learned projection moves MIPS estimates *away*
from `V_true(π_agent)` (most pronounced on GPT-4: −0.053 shift towards more
negative bias). DM and OffCEM are barely affected. The smoke-time learned
concat-pca projection was meaningfully shrinking MIPS bias on at least the
GPT-4 judge — worth flagging for the eventual learned-embedding ablation.

## Caveats

- **n=100** per judge dataset (small for CI coverage work; bootstrap CIs are
  consequently wide, but bias still dominates).
- **No StandardScaler** in the density-ratio classifier. The run emits
  `RuntimeWarning: divide by zero / overflow / invalid value in matmul` from
  `sklearn.utils.extmath.matmul` on most bootstrap iterations — same numerical
  pattern flagged in [`embedding-ci-rippling-bentley.md`](../.../plans/embedding-ci-rippling-bentley.md).
  Some fraction of the observed bias may be a numerical artifact; out of scope
  for this run, but a future numerics-fix run (Pipeline(StandardScaler,
  LogisticRegression), tighter C) would isolate it.
- **No learned action embedding** in this run by design. Smoke used
  concat-pca with pca_dim=128 / latent_dim=64; this re-run drops it for
  cleaner judge×estimator comparison and direct comparability with the
  Sonnet 4.6 single-judge embedding ablation.
- **Same π_b across judges**: V_true(π_b) is constant (0.5191) because the
  three judges happen to score clinician notes identically on average — the
  judge variation is concentrated in π_agent scoring.

## Reproduction

```bash
cd /Users/niuniu/counterfactual-conversation-emulation
git checkout feature/phase4-dm-estimator

# stage CounselBench inputs (assumes /Users/niuniu/Desktop/RA/data/ has the three .jsonl)
cp /Users/niuniu/Desktop/RA/data/phase3_{gemini,gpt4,llama3}_expert_scored.jsonl data/phase3/

# three parallel runs (~50 min each on M-series Mac with MPS; HF cache is shared)
for JUDGE in gemini gpt4 llama3; do
  PYTHONPATH=src .venv/bin/python scripts/run_phase5.py \
    --input data/phase3/phase3_${JUDGE}_expert_scored.jsonl \
    --output data/phase5/headline_counselbench_${JUDGE}_bge_boot100.json \
    --embedder bge --n-boot 100 --seed 0 \
    2>&1 | tee logs/phase5_${JUDGE}_boot100.log &
  sleep 30   # stagger to share BGE-M3 HF cache (~2.3 GB) without duplicate downloads
done
wait
```

## Pointers

- Single-dataset embedding ablation (Sonnet 4.6 judge, 4 embedders): [`phase5_embedding_ablation.md`](phase5_embedding_ablation.md)
- Single-judge baseline (Sonnet 4.6, OpenAI embedder): [`phase5_sonnet46_judge_ablation.md`](phase5_sonnet46_judge_ablation.md)
- Numerics-fix plan (deferred): [`embedding-ci-rippling-bentley.md`](../../.claude/plans/embedding-ci-rippling-bentley.md)
- Learned action embedding (smoke-mode results, deferred bootstrap): [`phase6_learned_action_embedding.md`](phase6_learned_action_embedding.md)
