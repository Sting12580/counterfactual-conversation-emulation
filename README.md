# Counterfactual Conversation Emulation

This repository contains Phase 1 and Phase 2 artifacts for evaluating medical AI agents with counterfactual conversation emulation.

Phase 2 builds a canonical dataset

```text
D = {(X_i, A_i^b, E_i, Y_i)}_{i=1}^n
```

from public medical conversation sources. Raw data are downloaded on demand and are not committed to git.

## Quickstart on CARC

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd counterfactual_conversation_emulation

python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cce-download
cce-build
cce-inspect -n 5
```

Expected outputs:

```text
data/processed/phase2_dataset.jsonl
data/processed/phase2_dataset.csv
data/processed/manifest.json
data/processed/dataset_card.md
data/processed/qa_counts.csv
data/processed/splits/*.jsonl
```

The default sources are:

- `aci_bench`: full encounter dialogue + clinical note.
- `mts_dialog`: short dialogue + clinical note section.
- `primock57`: small mock primary-care consultation dataset for smoke tests.

Optional sources:

- `meddialog`: large dialogue-only source; disabled by default because it lacks clinician notes.
- `notechat`: synthetic dialogue-note data; disabled by default and should be used only for sensitivity or augmentation experiments.

## Main Commands

Download default raw data sources:

```bash
cce-download
```

Download selected sources:

```bash
cce-download --sources aci_bench,mts_dialog,primock57
```

Build canonical dataset:

```bash
cce-build
```

Build only included examples:

```bash
cce-build --included-only
```

Freeze the included dataset before Phase 3:

```bash
cce-build --included-only --output-dir data/processed_included
cce-freeze \
  --input data/processed_included/phase2_dataset.jsonl \
  --manifest data/processed_included/manifest.json \
  --output dataset_freezes/phase2_v1.json \
  --label phase2_v1
```

Run a small smoke build:

```bash
cce-build --sample-per-source 5
```

Mark rows as needing rubric scoring without calling an API:

```bash
cce-score --provider none
```

Score a small batch with an OpenAI-compatible judge:

```bash
pip install -e '.[judge]'
export OPENAI_API_KEY=...
cce-score --provider openai --model gpt-4.1-mini --limit 20
```

Score with a cross-vendor judge to reduce same-provider self-preference:

```bash
export ANTHROPIC_API_KEY=...
cce-score --provider anthropic --model claude-sonnet-4-20250514 --limit 20

export GEMINI_API_KEY=...
cce-score --provider google --model gemini-2.5-pro --limit 20
```

The default Phase 3 rubric judge config is in `configs/rubric_judge.yaml`. See [docs/phase3_rubric_judge_decision.md](docs/phase3_rubric_judge_decision.md).
For full clinician-vs-agent reruns with separate score fields, see [docs/cross_vendor_judge_runbook.md](docs/cross_vendor_judge_runbook.md).

After clinician scoring, generate and score target-agent outputs:

```bash
cce-generate-agent \
  --input data/phase3/clinician_scored_all.jsonl \
  --output data/phase3/agent_actions_all.jsonl \
  --provider openai \
  --model gpt-4.1

cce-score \
  --input data/phase3/agent_actions_all.jsonl \
  --output data/phase3/agent_scored_all.jsonl \
  --provider openai \
  --model gpt-4.1 \
  --action-field a_agent \
  --score-field y_agent_score \
  --rubric-field y_agent_rubric \
  --source-field y_agent_source

cce-effect \
  --input data/phase3/agent_scored_all.jsonl \
  --output data/phase3/ground_truth_effect.json
```

## CounselBench Expert Labels

CounselBench-Eval can be converted into the same Phase 2/3 shape, but its reward comes from
mental-health professional annotations rather than an LLM judge. The converter aggregates the
five expert annotations for each `(questionID, responder)`, uses the human therapist response as
the behavior-policy baseline, and writes paired expert-scored files for the logged LLM responders.

```bash
pip install -e '.[hf]'
cce-build-counselbench --output-dir data/counselbench
```

Expected outputs:

```text
data/counselbench/phase2_dataset.jsonl
data/counselbench/phase3_gpt4_expert_scored.jsonl
data/counselbench/phase3_llama3_expert_scored.jsonl
data/counselbench/phase3_gemini_expert_scored.jsonl
data/counselbench/ground_truth_effect_gpt4.json
data/counselbench/ground_truth_effect_llama3.json
data/counselbench/ground_truth_effect_gemini.json
```

The default reward is a normalized composite of expert `overall`, `empathy`, `specificity`,
`factual_consistency`, low `toxicity`, and low unauthorized `medical_advice` rate. Use
`--reward-mode overall` if you want the expert overall score only.

## Canonical Schema

| Column | Meaning |
| --- | --- |
| `example_id` | Stable source/split/source-id identifier. |
| `source` | Source dataset name. |
| `split` | Source split. |
| `source_id` | Original source identifier. |
| `x_patient_context` | Patient context available at time zero. |
| `a_clinician` | Clinician final diagnosis/differential/management-plan action. |
| `e_action_repr` | Reserved for later embedding or cluster representation. |
| `y_score` | Rubric score in `[0,1]`; null until scoring is run. |
| `y_rubric` | Serialized rubric output or scoring status. |
| `y_source` | Scoring provider or `unscored`/`needs_scoring`. |
| `dialogue` | Raw normalized dialogue provenance. |
| `note` | Raw normalized note provenance. |
| `section_header` | Source section header when available. |
| `section_text` | Source section text when available. |
| `time_zero_policy` | Leakage rule used for `X`. |
| `extraction_method` | How `A_i^b` was extracted. |
| `inclusion_status` | `included`, `excluded`, or `auxiliary_only`. |
| `exclusion_reasons` | Reasons an example should not enter primary OPE. |
| `metadata` | Source-specific metadata. |

## Phase 2 Design

The pipeline intentionally separates three stages:

1. Download raw public data.
2. Extract auditable `(X, A_i^b)` pairs with leakage flags.
3. Score `Y_i` with a calibrated rubric judge.

This avoids mixing source-format parsing with causal assumptions. The generated dataset can be used immediately for support diagnostics and agent-action generation, while rubric scoring can be rerun as the judge prompt improves.

See [docs/phase2_data_construction_review.md](docs/phase2_data_construction_review.md) for the data-source and literature review behind the design.
