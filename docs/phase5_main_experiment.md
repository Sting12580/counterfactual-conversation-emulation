# Phase 5 Main Experiment

## Setup
- **Dataset**: `data/phase3/agent_scored_all_judge_gpt4o.jsonl` (397 included)
- **Judge**: `openai:gpt-4o` (switched from gpt-4.1 to mitigate self-preference bias)
- **Behavior policy** π_b: clinician final assessment + plan extracted from notes
- **Target policy** π_agent: `openai:gpt-4.1` per `configs/agent_policy.yaml`
- **Estimators**: DM (Jaques 2019), MIPS (Saito & Joachims 2022), OffCEM (Saito 2023)
- **Featurization**: `[phi(x) ; phi(a) ; phi(x) * phi(a)]`
- **Bootstrap**: 100 iterations with full refit per resample

## Ground truth (from `data/phase3/ground_truth_effect_judge_gpt4o.json`)

| Quantity | Value |
|---|---|
| V_true(π_b) | 0.6690 |
| V_true(π_agent) | 0.8585 |
| True effect | **+0.1895** (95% CI [0.168, 0.212]) |
| Direction(agent > clinician) rate | 81.9% |

Switching judge from gpt-4.1 → gpt-4o shrank the agent–clinician gap from
+0.263 → +0.189; direction rate dropped from 96.2% to 81.9%, consistent
with substantial self-preference bias having been removed. The remaining
gap is statistically significant.

## Result 1 — SBERT embedding (sentence-transformers/all-MiniLM-L6-v2, 384-d)

`data/phase5/headline_sbert_v2.json` — 100-bootstrap run with raw values
preserved (45 min wall time). RMSE and direction-agreement rate are
computed exactly from the 100 bootstrap estimates.

| Estimator | V̂ | Bias | Rel Bias | RMSE | 95% CI | Cov | Dir % |
|---|---|---|---|---|---|---|---|
| DM | 0.7330 | −0.1255 | **−14.62%** ✓ | 0.1357 | [0.697, 0.751] | no | **100.0%** ✓ |
| MIPS | 0.6636 | −0.1949 | −22.71% ✗ | 0.2142 | [0.574, 0.719] | no | **32.0%** ✗ |
| OffCEM | 0.7256 | −0.1329 | **−15.48%** ✓ | 0.1401 | [0.695, 0.747] | no | **100.0%** ✓ |

**Plan v2 success criterion** (rel bias < 20% AND direction correct):
DM and OffCEM PASS; MIPS FAILS. The MIPS direction rate of 32 % is the
striking headline failure: across 100 bootstrap resamples, MIPS gets the
sign of the effect right only one third of the time — worse than chance.

## Result 2 — OpenAI embedding (text-embedding-3-small, 1536-d)

`data/phase5/headline_openai_v2.json` — 100-bootstrap with raw values
preserved (~155 min wall time).

| Estimator | V̂ | Bias | Rel Bias | RMSE | 95% CI | Cov | Dir % |
|---|---|---|---|---|---|---|---|
| DM | 0.6884 | −0.1701 | −19.81% | 0.1789 | [0.660, 0.700] | no | **96.0%** ✓ |
| MIPS | 0.7494 | −0.1091 | **−12.71%** ✓ | **0.1068** | [0.706, 0.793] | no | **100.0%** ✓ |
| OffCEM | 0.6921 | −0.1664 | −19.39% | 0.1772 | [0.663, 0.703] | no | **96.0%** ✓ |

**All three estimators recover the direction.** MIPS has the smallest
bias (12.71%), smallest RMSE (0.1068 — lowest of any estimator-embedding
combination in this table), and its CI upper bound (0.793) is the closest
any estimator gets to V_true_agent = 0.8585, missing the truth by only
0.066. DM and OffCEM are both slightly worse than under SBERT for point
estimate, but recover direction reliably (96 %).

## Side-by-side comparison (both with 100-iter bootstrap, raw values saved)

```
                      SBERT (384-d)                              OpenAI (1536-d)
Estimator     V̂    Rel Bias  RMSE    Dir %       V̂    Rel Bias  RMSE    Dir %
─────────────────────────────────────────────────────────────────────────────────────────
DM          0.733  -14.6%   0.1357  100.0%      0.688  -19.8%   0.1789   96.0%
MIPS        0.664  -22.7%   0.2142   32.0%      0.749  -12.7%   0.1068  100.0%   ★ best
OffCEM      0.726  -15.5%   0.1401  100.0%      0.692  -19.4%   0.1772   96.0%

V_true(π_agent) = 0.8585       (none of the six CIs covers truth)
```

## Key findings

1. **The "best" estimator depends on the embedding.**
   - SBERT → DM is best by RMSE 0.1357 (outcome model dominates;
     classifier-based density ratio collapses in 384-d space).
   - OpenAI → MIPS is best by RMSE 0.1068 (richer embedding gives the
     classifier real signal; DM/OffCEM lose to small-n overfit on
     4608-d concatenated features).

2. **MIPS direction rate flips from 32 % under SBERT to 100 % under
   OpenAI.** Under SBERT, MIPS gets the sign wrong on 68 out of 100
   bootstrap resamples — strictly worse than chance. This is the most
   dramatic empirical finding in the paper: classifier-based density
   ratio in a low-dim embedding is not just biased, it is *anti-correlated*
   with truth on bootstrap resamples.

3. **All estimators systematically underestimate V_agent regardless of
   embedding.** All six 95 % CIs lie entirely below the true V_agent.
   Bias dominates over sampling variance; tightening the CI does not
   improve coverage. This is plan v2 §"risk 2" (positivity violation)
   manifesting on real text data.

4. **Plan v2 §"two key assumptions" partially validated empirically.**
   - "No Direct Effect" holds well enough that MIPS over OpenAI embedding
     recovers direction with 12.7 % bias and 100 % direction rate.
   - "Common embedding support" issues are visible in the SBERT run where
     MIPS misses direction with rate 32 %.

5. **OffCEM's doubly-robust promise does not pay off on this dataset.**
   In both embedding regimes, OffCEM lands between DM and MIPS by RMSE
   rather than outperforming both. The paper's nuance: doubly robust is
   only "best of both" when neither component fails outright; here
   embedding choice already determines which component is healthy.

## What still needs to run

- [x] OpenAI bootstrap (CI + coverage) — completed 2026-05-16, ~155 min
- [ ] Phase 6 ablation #2: agent capability (GPT-3.5 weak vs GPT-4.1 strong)
- [ ] Phase 6 ablation #3: positivity diagnostics (weight tail, ESS)
- [ ] Phase 6 ablation #4: conversation length subgroups
- [ ] Phase 6 ablation #5: rubric judge sensitivity (try Claude / Gemini)

## Reproduction

```bash
# SBERT baseline (default)
python scripts/run_phase5.py --n-boot 100

# OpenAI ablation
OPENAI_API_KEY=sk-... python scripts/run_phase5.py \
    --embedder openai --n-boot 100 \
    --output data/phase5/headline_openai.json

# Quick point estimates only (no CI)
python scripts/run_phase5.py --n-boot 0
```

Outputs land under `data/phase5/` (gitignored). Compare to ground-truth
at `data/phase3/ground_truth_effect_judge_gpt4o.json`.
