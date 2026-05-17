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

`data/phase5/headline.json` — full 100-bootstrap run, 40 min wall time.

| Estimator | V̂ | Bias | Rel Bias | 95% CI | CI cov truth | Dir |
|---|---|---|---|---|---|---|
| DM | 0.7330 | −0.1255 | **−14.62%** ✓ | [0.697, 0.751] | no | **yes** ✓ |
| MIPS | 0.6636 | −0.1949 | −22.71% ✗ | [0.574, 0.719] | no | no ✗ |
| OffCEM | 0.7256 | −0.1329 | **−15.48%** ✓ | [0.695, 0.747] | no | **yes** ✓ |

**Plan v2 success criterion** (rel bias < 20% AND direction correct):
DM and OffCEM PASS; MIPS FAILS.

## Result 2 — OpenAI embedding (text-embedding-3-small, 1536-d)

`data/phase5/headline_openai_point.json` — point estimates only (bootstrap
deferred; would take ~80 min due to 4× feature dimensionality).

| Estimator | V̂ | Bias | Rel Bias | Dir |
|---|---|---|---|---|
| DM | 0.6889 | −0.1696 | −19.76% | yes |
| MIPS | 0.7494 | −0.1091 | **−12.71%** ✓ | yes |
| OffCEM | 0.6939 | −0.1646 | −19.18% | yes |

**All three estimators recover the direction.** MIPS now has the smallest
bias (12.71%), DM and OffCEM both slightly worse than under SBERT.

## Side-by-side comparison

```
                    SBERT (384-d)           OpenAI (1536-d)
Estimator     V̂      Bias    Dir         V̂      Bias     Dir
─────────────────────────────────────────────────────────────────────
DM          0.733    -14.6%   ✓         0.689   -19.8%    ✓
MIPS        0.664    -22.7%   ✗         0.749   -12.7%    ✓  <-- flipped
OffCEM      0.726    -15.5%   ✓         0.694   -19.2%    ✓

V_true(π_agent) = 0.8585
```

## Key findings

1. **The "best" estimator depends on the embedding.**
   - SBERT → DM is best (outcome model dominates; classifier-based density
     ratio collapses in low-dim space).
   - OpenAI → MIPS is best (richer embedding gives the classifier real
     signal; DM/OffCEM lose to small-n overfit on 4608-d concatenated
     features).

2. **All estimators systematically underestimate V_agent (point-estimate
   bias is consistently negative).** Bootstrap CIs under SBERT are tight
   (~5pp wide for DM/OffCEM) but none cover the ground-truth V_agent.
   Bias dominates over sampling variance.

3. **Plan v2 §"two key assumptions" partially validated empirically.**
   - "No Direct Effect" holds well enough that MIPS over OpenAI embedding
     recovers direction with 12.7% bias.
   - "Common embedding support" issues are visible in the SBERT run where
     MIPS misses direction entirely.

4. **OffCEM's doubly-robust promise does not pay off on this dataset.**
   In both embedding regimes, OffCEM lands between DM and MIPS rather
   than outperforming both. This is the paper's nuance: doubly robust is
   only "best of both" when neither component fails outright; here
   embedding choice already determines which component is healthy.

## What still needs to run

- [ ] OpenAI bootstrap (CI + coverage) — ~80 min
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
