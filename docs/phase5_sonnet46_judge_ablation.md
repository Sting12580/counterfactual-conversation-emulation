# Phase 5 Sonnet 4.6 Judge Ablation

Rubric judge sensitivity ablation (Phase 6 item #5 from
[phase5_main_experiment.md](phase5_main_experiment.md)): re-score the full
n=397 dataset with `claude-sonnet-4-6` and rerun DM / MIPS / OffCEM under
the same OpenAI embedding to isolate the judge-choice effect.

## Setup
- **Dataset**: `data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl` (397 included)
- **Judge**: `anthropic:claude-sonnet-4-6` (cross-vendor; see [cross_vendor_judge_runbook.md](cross_vendor_judge_runbook.md))
- **Behavior policy** π_b: clinician final assessment + plan extracted from notes (unchanged)
- **Target policy** π_agent: `openai:gpt-4.1` per `configs/agent_policy.yaml` (unchanged)
- **Embedder**: `openai:text-embedding-3-small` (1536-d) — matches Result 2 in main experiment
- **Estimators**: DM (Jaques 2019), MIPS (Saito & Joachims 2022), OffCEM (Saito 2023)
- **Featurization**: `[phi(x) ; phi(a) ; phi(x) * phi(a)]`
- **Bootstrap**: 100 iterations with full refit per resample

## Ground truth (from `data/phase3/ground_truth_effect_judge_claude_sonnet46.json`)

| Quantity | Value |
|---|---|
| V_true(π_b) | 0.6178 |
| V_true(π_agent) | 0.8413 |
| True effect | **+0.2236** (95% CI [0.206, 0.241]) |
| Direction(agent > clinician) rate | 97.7% |

Switching judge from gpt-4o → claude-sonnet-4-6 lowers V_b by 0.05
(0.669 → 0.618) while leaving V_agent roughly unchanged (0.859 → 0.841),
widening the agent–clinician gap from +0.189 to +0.224. The per-record
direction rate also jumps from 81.9% (gpt-4o) to 97.7% (sonnet-4-6),
indicating sonnet-4-6 favors the agent over the clinician note far more
consistently than gpt-4o does — a finding worth tracking as potential
cross-vendor judge bias rather than self-preference removal.

## Result — Sonnet 4.6 judge, OpenAI embedding

`data/phase5/headline_sonnet46_openai.json` — 100-bootstrap run (~152 min
wall time).

| Estimator | V̂ | Bias | Rel Bias | RMSE | 95% CI | Cov | Dir % |
|---|---|---|---|---|---|---|---|
| DM | 0.6364 | −0.2049 | −24.35% | 0.2111 | [0.609, 0.653] | no | 91.0% |
| MIPS | 0.6850 | −0.1563 | **−18.58%** ✓ | **0.1526** | [0.635, 0.734] | no | **99.0%** ✓ |
| OffCEM | 0.6427 | −0.1986 | −23.60% | 0.2088 | [0.611, 0.656] | no | 92.0% |

**Plan v2 success criterion** (rel bias < 20% AND direction correct):
**MIPS PASSES** (rel bias −18.58%, direction rate 99%); DM and OffCEM both
fail the bias threshold (−24.35% and −23.60%) although they still recover
direction at ≥91%.

## Side-by-side: gpt-4o vs sonnet-4-6 judge (both with OpenAI embedding)

```
                      gpt-4o judge                                 sonnet-4-6 judge
Estimator     V̂    Rel Bias  RMSE    Dir %       V̂    Rel Bias  RMSE    Dir %
─────────────────────────────────────────────────────────────────────────────────────────
DM          0.688  -19.8%   0.1789   96.0%      0.636  -24.4%   0.2111   91.0%
MIPS        0.749  -12.7%   0.1068  100.0%      0.685  -18.6%   0.1526   99.0%   ★ best
OffCEM      0.692  -19.4%   0.1772   96.0%      0.643  -23.6%   0.2088   92.0%

V_true(π_agent)    = 0.8585                  V_true(π_agent)    = 0.8413
True effect        = +0.1895                 True effect        = +0.2236
(none of the six CIs covers truth in either column)
```

## Key findings

1. **MIPS remains the best estimator under the sonnet-4-6 judge.**
   MIPS retains the lowest RMSE (0.1526) and highest direction rate
   (99%) among the three estimators. The ranking MIPS > OffCEM > DM
   established under gpt-4o transfers cleanly across judges.

2. **All point estimates shift downward by roughly 0.05.** DM, MIPS, and
   OffCEM all drop ~0.05 from their gpt-4o values, mirroring the
   ground-truth shift in V_b (also −0.05). The estimators move with
   V_true_b, not V_true_agent — consistent with the systematic
   underestimation noted in the main experiment, which persists across
   judges.

3. **Relative bias and RMSE both worsen under sonnet-4-6.** MIPS goes
   from −12.7% / 0.1068 → −18.6% / 0.1526; DM from −19.8% / 0.1789 →
   −24.4% / 0.2111. The wider true effect (+0.224 vs +0.189) gives the
   estimators a harder target to hit, since their downward bias is
   roughly invariant in absolute terms.

4. **CI coverage fails uniformly across both judges.** All six 95% CIs
   miss V_true_agent. The judge swap does not move the failure mode:
   bias dominates sampling variance, and tighter CIs do not buy
   coverage. This re-confirms plan v2 §"risk 2" (positivity violation)
   as the binding constraint on real text data.

5. **Direction rate is the metric that's most stable across judges.**
   MIPS direction rate stays at 99–100%; DM stays at 91–96%. For
   deployment decisions (which agent to ship), the judge choice has
   minimal effect — the OPE ranking transfers. For absolute V̂
   accuracy, judge choice matters more.

## Caveats

- **Cross-vendor judge bias.** Sonnet-4-6 gives the agent a higher
  direction-favored rate (97.7%) than gpt-4o (81.9%). Self-preference
  bias is unlikely (agent is gpt-4.1, judge is Anthropic), but
  cross-vendor preference (e.g., judges favor longer/more formatted
  responses) cannot be ruled out from this run alone.
- **Classifier numerical warnings.** sklearn's `LogisticRegression`
  emitted `divide by zero` / `overflow` / `invalid value` warnings on
  many bootstrap iterations due to the 4608-d feature matrix being
  ill-conditioned on n=397. Results came through but MIPS density-ratio
  fits are not robustly numerically stable in this regime. Mitigation
  candidates: feature standardization, switch to `solver='liblinear'`
  with stronger regularization, or PCA pre-reduction before the
  classifier.

## What still needs to run

- [x] Phase 6 ablation #5: rubric judge sensitivity (Claude sonnet-4-6) — completed 2026-05-22, ~152 min
- [ ] Phase 6 ablation #5 follow-up: third judge (Gemini 2.5 Pro) for triangulation
- [ ] Phase 6 ablation #2: agent capability (GPT-3.5 weak vs GPT-4.1 strong)
- [ ] Phase 6 ablation #3: positivity diagnostics (weight tail, ESS)
- [ ] Phase 6 ablation #4: conversation length subgroups
- [ ] Numerical stability fix for MIPS classifier under high-dim features

## Reproduction

```bash
# 1. Build phase3 jsonl with sonnet-4-6 judge scores
#    (see docs/cross_vendor_judge_runbook.md for the cce-score command)

# 2. Rename score fields so run_phase5.py picks them up:
python -c "
import json
from pathlib import Path
src = Path('data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl')
out = src.with_name(src.stem + '.renamed.jsonl')
with src.open() as fi, out.open('w') as fo:
    for line in fi:
        r = json.loads(line)
        r['y_score'] = r['y_score_claude_sonnet46']
        r['y_agent_score'] = r['y_agent_score_claude_sonnet46']
        fo.write(json.dumps(r) + '\n')
"

# 3. Run phase5 with OpenAI embedder
OPENAI_API_KEY=sk-... python scripts/run_phase5.py \
    --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
    --output data/phase5/headline_sonnet46_openai.json \
    --embedder openai --n-boot 100 --seed 0
```

Outputs land under `data/phase5/` (gitignored). Compare to ground-truth
at `data/phase3/ground_truth_effect_judge_claude_sonnet46.json`.
