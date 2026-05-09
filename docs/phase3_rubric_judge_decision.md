# Phase 3 Rubric Judge Decision

## Decision

Use a structured LLM-as-judge rubric to score each `(X, A)` pair on a continuous `[0, 1]` scale. The Phase 3 default rubric is defined in:

```text
configs/rubric_judge.yaml
```

Recommended judge setup:

- Default scalable judge: `gpt-5.4-mini`
- High-rigor sensitivity judge: `gpt-5.5`
- Fallback/comparability judge: `gpt-4.1`

The target agent model should be recorded separately from the judge model. When possible, do not use the exact same model/configuration as both generator and judge.

## Why This Rubric

The project needs a scalar reward for final clinical outputs, not a static multiple-choice accuracy label. `onboarding.pdf` explicitly recommends a continuous LLM-as-judge rubric over binary accuracy and defines Phase 3 as the point where clinician and agent actions are scored with the same judge.

Current medical AI evaluation literature supports this direction but also imposes reliability constraints:

- HealthBench uses open-ended health conversations and physician-authored rubric criteria rather than only multiple-choice QA. It covers behavior axes such as accuracy, communication quality, and context seeking, with a physician-consensus subset for critical dimensions.
- AMIE's diagnostic-conversation evaluation uses specialist physician ratings over diagnosis and management axes, including escalation appropriateness and confabulation absence, plus communication-related dimensions.
- Recent clinical LLM evaluation papers emphasize diagnosis, management, safety, fabricated facts, and expert oversight as central dimensions.
- LLM-as-judge clinical-summary work shows that rubric-based LLM judging can align with physician ratings in some clinical language-generation tasks, but also that transfer to other tasks requires validation.

Therefore, the Phase 3 rubric uses five dimensions:

| Dimension | Weight | Reason |
| --- | ---: | --- |
| `diagnostic_quality` | 0.30 | Captures diagnosis/differential correctness and reasoning. |
| `management_quality` | 0.30 | Captures tests, treatment, medication, referral, and follow-up quality. |
| `safety_and_escalation` | 0.20 | Captures urgent escalation, contraindications, unsafe omissions, and harmful advice. |
| `context_use_and_factuality` | 0.15 | Penalizes hallucinated facts and rewards patient-specific use of `X`. |
| `communication_and_uncertainty` | 0.05 | Rewards uncertainty handling and clear caveats; lower weight because the action is a final clinical output, not the full patient conversation. |

The score is weighted because Phase 4/5 OPE estimators need a single continuous reward, but the dimension scores are preserved in `y_rubric` for diagnostics and ablations.

## Calibration Requirement

Before treating the judge score as ground truth:

1. Sample 100 included cases.
2. Score each `(X, A_clinician)` and a matched set of `(X, A_agent)` outputs with the rubric judge.
3. Have at least one clinical reviewer score the same cases, blind to source when possible.
4. Compute Spearman correlation between human and judge scores.
5. If Spearman `< 0.70`, revise the rubric, switch judge model, or move to a task with a gold label.

This follows the fallback threshold already specified in `onboarding.pdf`.

## Phase 3 Frozen Dataset

Freeze the dataset before generating agent outputs:

```bash
cce-build --included-only --output-dir data/processed_included
cce-freeze \
  --input data/processed_included/phase2_dataset.jsonl \
  --manifest data/processed_included/manifest.json \
  --output dataset_freezes/phase2_v1.json \
  --label phase2_v1
```

The freeze manifest stores the dataset SHA256 hash, row counts, source counts, current git commit, and reproduction command. Later Phase 3 artifacts should record the freeze label and hash.

## Phase 3 Scoring Commands

Dry run that marks rows as needing scoring:

```bash
cce-score \
  --input data/processed_included/phase2_dataset.jsonl \
  --output data/phase3/clinician_needs_scoring.jsonl \
  --provider none
```

Small OpenAI judge run:

```bash
pip install -e '.[judge]'
export OPENAI_API_KEY=...
cce-score \
  --input data/processed_included/phase2_dataset.jsonl \
  --output data/phase3/clinician_scored_20.jsonl \
  --provider openai \
  --limit 20
```

Override judge model when needed:

```bash
cce-score \
  --input data/processed_included/phase2_dataset.jsonl \
  --output data/phase3/clinician_scored_20_gpt41.jsonl \
  --provider openai \
  --model gpt-4.1 \
  --limit 20
```

## Agent Generation and Ground Truth Effect

After clinician actions are scored, generate target-agent actions with the frozen target policy:

```bash
cce-generate-agent \
  --input data/phase3/clinician_scored_all.jsonl \
  --output data/phase3/agent_actions_all.jsonl \
  --provider openai \
  --model gpt-4.1
```

Score generated agent actions with the same rubric judge:

```bash
cce-score \
  --input data/phase3/agent_actions_all.jsonl \
  --output data/phase3/agent_scored_all.jsonl \
  --provider openai \
  --model gpt-4.1 \
  --action-field a_agent \
  --score-field y_agent_score \
  --rubric-field y_agent_rubric \
  --source-field y_agent_source
```

Compute Phase 3 ground truth:

```bash
cce-effect \
  --input data/phase3/agent_scored_all.jsonl \
  --output data/phase3/ground_truth_effect.json \
  --bootstrap 1000
```

The resulting JSON reports:

- `V_true(pi_b) = mean(y_score)`
- `V_true(pi_agent) = mean(y_agent_score)`
- `true_effect = V_true(pi_agent) - V_true(pi_b)`
- bootstrap 95% confidence interval

## Literature Notes

- HealthBench: Evaluating Large Language Models Towards Improved Human Health. https://arxiv.org/abs/2505.08775
- OpenAI HealthBench overview. https://openai.com/index/healthbench/
- Towards conversational diagnostic artificial intelligence. https://www.nature.com/articles/s41586-025-08866-7
- A prospective clinical feasibility study of a conversational diagnostic AI in an ambulatory primary care clinic. https://arxiv.org/abs/2603.08448
- Evaluating clinical AI summaries with large language models as judges. https://www.nature.com/articles/s41746-025-02005-2
- Multidisciplinary blinded randomized expert evaluation of large language models for clinical diagnosis and management. https://www.nature.com/articles/s43856-026-01576-9
- Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs. https://arxiv.org/abs/2601.18706
- CliBench: A Multifaceted and Multigranular Evaluation of Large Language Models for Clinical Decision Making. https://arxiv.org/abs/2406.09923
- AgentClinic: a multimodal agent benchmark to evaluate AI in simulated clinical environments. https://arxiv.org/abs/2405.07960
