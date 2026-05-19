# Cross-vendor Phase 3 judge runbook

Use this when rerunning Phase 3 ground truth with a judge outside OpenAI. The target
agent can remain `openai:gpt-4.1`; the judge should be a different provider to reduce
self-preference risk.

## Recommended judges

| Role | Provider | Model | Why |
| --- | --- | --- | --- |
| Primary external sensitivity | Anthropic | `claude-sonnet-4-20250514` | Strong reasoning, lower cost/latency than Opus. |
| High-rigor adjudication | Anthropic | `claude-opus-4-1-20250805` | Best for final sensitivity or disagreement review; more expensive. |
| Second-vendor check | Google | `gemini-2.5-pro` | Stable Gemini model with structured JSON output support. |

## Install and keys

```bash
pip install -e '.[judge]'

# Anthropic
export ANTHROPIC_API_KEY=...

# Google Gemini
export GEMINI_API_KEY=...
# or:
export GOOGLE_API_KEY=...
```

## Claude Sonnet 4 judge

This assumes `data/phase3/agent_actions_all.jsonl` already contains both
`a_clinician` and `a_agent`. It writes new score fields so existing GPT-based scores
are not overwritten.

```bash
cce-score \
  --input data/phase3/agent_actions_all.jsonl \
  --output data/phase3/agent_actions_all_judge_claude_sonnet4_clinician.jsonl \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --action-field a_clinician \
  --score-field y_score_claude_sonnet4 \
  --rubric-field y_rubric_claude_sonnet4 \
  --source-field y_source_claude_sonnet4

cce-score \
  --input data/phase3/agent_actions_all_judge_claude_sonnet4_clinician.jsonl \
  --output data/phase3/agent_scored_all_judge_claude_sonnet4.jsonl \
  --provider anthropic \
  --model claude-sonnet-4-20250514 \
  --action-field a_agent \
  --score-field y_agent_score_claude_sonnet4 \
  --rubric-field y_agent_rubric_claude_sonnet4 \
  --source-field y_agent_source_claude_sonnet4

cce-effect \
  --input data/phase3/agent_scored_all_judge_claude_sonnet4.jsonl \
  --output data/phase3/ground_truth_effect_judge_claude_sonnet4.json \
  --clinician-score-field y_score_claude_sonnet4 \
  --agent-score-field y_agent_score_claude_sonnet4 \
  --bootstrap 1000
```

## Gemini 2.5 Pro judge

```bash
cce-score \
  --input data/phase3/agent_actions_all.jsonl \
  --output data/phase3/agent_actions_all_judge_gemini25pro_clinician.jsonl \
  --provider google \
  --model gemini-2.5-pro \
  --action-field a_clinician \
  --score-field y_score_gemini25pro \
  --rubric-field y_rubric_gemini25pro \
  --source-field y_source_gemini25pro

cce-score \
  --input data/phase3/agent_actions_all_judge_gemini25pro_clinician.jsonl \
  --output data/phase3/agent_scored_all_judge_gemini25pro.jsonl \
  --provider google \
  --model gemini-2.5-pro \
  --action-field a_agent \
  --score-field y_agent_score_gemini25pro \
  --rubric-field y_agent_rubric_gemini25pro \
  --source-field y_agent_source_gemini25pro

cce-effect \
  --input data/phase3/agent_scored_all_judge_gemini25pro.jsonl \
  --output data/phase3/ground_truth_effect_judge_gemini25pro.json \
  --clinician-score-field y_score_gemini25pro \
  --agent-score-field y_agent_score_gemini25pro \
  --bootstrap 1000
```

## Smoke test first

Before running all 397 examples, run each provider with `--limit 5` and inspect the
JSON in the corresponding rubric field. The result should contain `score`,
`dimensions`, `safety_override`, `major_issues`, and `rationale`.
