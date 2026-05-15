# Phase 3 Ground Truth Results

## Dataset and Policy

- Frozen dataset: `phase2_v1`
- Number of paired examples: `397`
- Agent generator: `gpt-4.1`
- Agent policy config: `configs/agent_policy.yaml`
- Rubric config: `configs/rubric_judge.yaml`
- Rubric version: `phase3_clinical_final_output_v1`

## Main Result: Judge = GPT-4.1

Artifact on CARC:

```text
data/phase3/ground_truth_effect.json
```

Results:

| Metric | Value |
| --- | ---: |
| `V_true(pi_b)` | `0.6658` |
| `V_true(pi_agent)` | `0.9289` |
| `True effect` | `+0.2631` |
| `95% bootstrap CI` | `[0.2441, 0.2844]` |
| `Agent better rate` | `96.22%` |

## Sensitivity Result: Judge = GPT-4o

Artifact on CARC:

```text
data/phase3/ground_truth_effect_judge_gpt4o.json
```

Results:

| Metric | Value |
| --- | ---: |
| `V_true(pi_b)` | `0.6690` |
| `V_true(pi_agent)` | `0.8585` |
| `True effect` | `+0.1895` |
| `95% bootstrap CI` | `[0.1679, 0.2117]` |
| `Agent better rate` | `81.86%` |

## Interpretation

Both judge models estimate a positive agent effect, and both bootstrap confidence intervals are above zero. The effect is smaller under `gpt-4o` than under `gpt-4.1`, which suggests that the first-pass `gpt-4.1` judge may favor the style or structure of `gpt-4.1` agent outputs.

This means Phase 3 v1 is complete, but the result should be treated as a rubric-judge ground truth rather than a validated clinical outcome. The next analysis step is calibration and bias checking: sample cases for human review, inspect whether the judge over-rewards length/structure, and compare additional judge models if available.

