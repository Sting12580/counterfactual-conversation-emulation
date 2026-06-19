# Phase 7 v2 Judge Weighting Plan

生成日期：2026-06-17

Phase 7 v2 更新 judge calibration / judge weighting pipeline，但不改动现有 OPE
和 CCE estimator。主目标是构建更干净的 answer-level human/expert calibration
matrix，并用多个 donor judge 的加权组合拟合 human/expert score。

## v2 数据默认行为

Ayers AskDocs 默认只使用 `composite`。旧版本同时纳入 `composite`、`quality`、
`empathy`，三者复用同一组 question-answer pair，但目标标签不同，会让 Ayers 在
主矩阵中过度重复。v2 主实验只把 composite 作为 Ayers 主标签。

`quality` 和 `empathy` 仍保留为诊断维度。需要诊断实验时显式传入：

```bash
PYTHONPATH=src python scripts/build_judge_calibration_dataset.py \
  --repo-root . \
  --ayers-dimensions composite,quality,empathy
```

默认 v2 canonical 输出：

```text
data/judge_calibration/canonical_examples_phase7_v2.jsonl
outputs/judge_calibration/data_summary_phase7_v2.json
```

## MEDIQA / LiveQA 下载与转换

MEDIQA-QA 2019 和 TREC LiveQA Medical 2017 需要先从官方 raw 文件转换成
answer-level JSONL。运行：

```bash
PYTHONPATH=src python scripts/download_medical_qa_calibration_sources.py
```

输出：

```text
data/mediqa_qa_2019/answers.jsonl
data/liveqa_medical_2017/answers.jsonl
data/raw/mediqa_qa_2019/*.xml
data/raw/liveqa_medical_2017/*.xml
data/raw/liveqa_medical_2017/*.txt
data/medical_qa_calibration_sources_manifest.json
```

然后重新构建 calibration examples：

```bash
PYTHONPATH=src python scripts/build_judge_calibration_dataset.py \
  --repo-root . \
  --include-mediqa-qa-2019 \
  --include-liveqa-medical-2017
```

LiveQA 的 qrels 文件中存在完全重复的 `(question, grade, answer)` 行；下载脚本
只做精确去重，不改变评分。

默认过滤规则：

- MEDIQA 中的 `train_liveqa_med` split 来自 TREC LiveQA Medical 问题集；
  同时构建两个数据集时会从 MEDIQA 输出中移除该 split，避免 question-level overlap。
- 同一数据集内完全相同的 `(question, answer)` 如果评分相同，只保留第一条。
- 同一数据集内完全相同的 `(question, answer)` 如果评分冲突，整组移除。
- LiveQA 的 `raw_grade=-2` 行没有可归一化人工评分，会从 JSONL 输出中移除。

如需保留未过滤转换结果，可传入 `--no-filter`。

## domain_dimension_v2 prompt

v2 donor judge 默认使用 `domain_dimension_v2`。Ayers、CounselBench、
MEDIQA-QA 2019 和 TREC LiveQA Medical 2017 共用一套通用 human-aligned rubric。
`domain`、`task_type` 和 `score_dimension` 仍会出现在 prompt 中作为上下文，但
rubric 本身不再为每个数据集写过细的专用规则。

通用 rubric 要求 judge 估计 careful human expert 会给这个 answer 的分数。它奖励
correct、safe、relevant、useful、sufficiently complete、clearly communicated 的回答；
惩罚 incorrect、unsafe、unsupported、irrelevant、incomplete、misleading 或明显不匹配用户问题的回答。

Prompt 包含：

- rubric id
- domain
- task type
- score dimension
- universal human-aligned evaluation rule
- 0.0、0.25、0.50、0.75、1.0 score anchors
- question
- optional context
- answer to rate
- JSON-only return instruction

Prompt 不包含 `human_score`、`human_score_raw`、`human_score_scale`、
`human_score_type`、`human_rubric`、原始 expert score 字段或任何 answer-level
gold label。

## MEDIQA-QA 2019 canonicalization

MEDIQA-QA 2019 是可选数据源。默认查找：

```text
data/mediqa_qa_2019/
data/raw/mediqa_qa_2019/
```

如果本地文件不存在，默认跳过；传入 `--fail-on-missing` 时失败。v2 不自动下载
MEDIQA-QA 2019。路径可以是目录或文件；目录下会读取 `.jsonl`、`.json`、`.csv`、`.tsv`。

映射规则：

- `dataset_id`: `mediqa_qa_2019`
- `domain`: `medical`
- `task_type`: `medical_qa`
- `score_dimension`: `medical_answer_quality`
- `question_id`: 优先使用源数据稳定 id，否则使用 question hash
- `answer`: answer/response/candidate answer 等字段
- `answer_source`: model key 或 answer source 字段

Score normalization:

```text
raw 1-4 medical expert rating -> (raw_score - 1) / 3
```

没有可用 raw 1-4 medical expert rating 的 answer 会跳过；不会从模型身份或答案文本推断分数。

## TREC LiveQA Medical 2017 canonicalization

TREC LiveQA Medical 2017 是可选数据源。默认查找：

```text
data/liveqa_medical_2017/
data/raw/liveqa_medical_2017/
```

如果本地文件不存在，默认跳过；传入 `--fail-on-missing` 时失败。v2 不自动下载
LiveQA Medical 2017。路径可以是目录或文件；目录下会读取 `.jsonl`、`.json`、`.csv`、`.tsv`。

映射规则：

- `dataset_id`: `liveqa_medical_2017`
- `domain`: `medical`
- `task_type`: `medical_qa`
- `score_dimension`: `medical_answer_quality`
- `question_id`: 优先使用源数据稳定 id，否则使用 question hash
- `answer`: answer/response/candidate answer 等字段
- `answer_source`: team/system/model/source 字段

Score normalization:

```text
official 0-3 score -> score / 3
raw 1-4 grade -> (grade - 1) / 3
```

如果 official 0-3 score 和 raw 1-4 grade 都不可用，该 answer 会跳过。

## v2 donor judge set

v2 默认 donor judge set：

```text
openai_gpt55
anthropic_sonnet46
google_gemini25flash_think1024
cohere_command_a_plus
mistral_large
```

DeepSeek 被 Cohere 替换。DeepSeek client 保留，便于旧实验复现，但不在 v2 默认
required judge list 中。

Gemini 沿用旧 full-run 设置 `google_gemini25flash_think1024`，对应
`gemini-2.5-flash` 和 Flash-specific `thinkingBudget=1024`。这样避免把 v2 主流程切到
更容易触发限额的 Gemini 2.5 Pro。

Cohere 使用 OpenAI-compatible chat endpoint：

```text
provider: cohere
model: command-a-plus-05-2026
api key env var: COHERE_API_KEY
base URL: https://api.cohere.ai/compatibility/v1
```

## v2 run commands

Build canonical v2 dataset:

```bash
PYTHONPATH=src python scripts/build_judge_calibration_dataset.py \
  --repo-root . \
  --out data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --summary-out outputs/judge_calibration/data_summary_phase7_v2.json \
  --ayers-dimensions composite \
  --include-ayers \
  --include-counselbench \
  --include-mediqa-qa-2019 \
  --include-liveqa-medical-2017
```

Run donor judge scoring for OpenAI:

```bash
PYTHONPATH=src python scripts/run_donor_judge_scoring.py \
  --examples data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --out data/judge_calibration/judge_scores/openai_gpt55_phase7_v2.jsonl \
  --judge-id openai_gpt55 \
  --provider openai \
  --model gpt-5.5 \
  --rubric-id domain_dimension_v2 \
  --execute
```

Run donor judge scoring for Anthropic:

```bash
PYTHONPATH=src python scripts/run_donor_judge_scoring.py \
  --examples data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --out data/judge_calibration/judge_scores/anthropic_sonnet46_phase7_v2.jsonl \
  --judge-id anthropic_sonnet46 \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --rubric-id domain_dimension_v2 \
  --execute
```

Run donor judge scoring for Gemini Flash thinking1024:

```bash
PYTHONPATH=src python scripts/run_donor_judge_scoring.py \
  --examples data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --out data/judge_calibration/judge_scores/gemini25flash_think1024_phase7_v2.jsonl \
  --judge-id google_gemini25flash_think1024 \
  --provider gemini \
  --model gemini-2.5-flash \
  --rubric-id domain_dimension_v2 \
  --execute
```

Run donor judge scoring for Cohere:

```bash
PYTHONPATH=src python scripts/run_donor_judge_scoring.py \
  --examples data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --out data/judge_calibration/judge_scores/cohere_command_a_plus_phase7_v2.jsonl \
  --judge-id cohere_command_a_plus \
  --provider cohere \
  --model command-a-plus-05-2026 \
  --rubric-id domain_dimension_v2 \
  --execute
```

Run donor judge scoring for Mistral:

```bash
PYTHONPATH=src python scripts/run_donor_judge_scoring.py \
  --examples data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --out data/judge_calibration/judge_scores/mistral_large_phase7_v2.jsonl \
  --judge-id mistral_large \
  --provider mistral \
  --model mistral-large-latest \
  --rubric-id domain_dimension_v2 \
  --execute
```

Build v2 score matrix:

```bash
PYTHONPATH=src python scripts/build_judge_score_matrix.py \
  --examples data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --judge-scores data/judge_calibration/judge_scores/openai_gpt55_phase7_v2.jsonl \
  --judge-scores data/judge_calibration/judge_scores/anthropic_sonnet46_phase7_v2.jsonl \
  --judge-scores data/judge_calibration/judge_scores/gemini25flash_think1024_phase7_v2.jsonl \
  --judge-scores data/judge_calibration/judge_scores/cohere_command_a_plus_phase7_v2.jsonl \
  --judge-scores data/judge_calibration/judge_scores/mistral_large_phase7_v2.jsonl \
  --out-csv data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv \
  --missingness-out outputs/judge_calibration/phase7_v2_missingness_report.json \
  --required-judges openai_gpt55,anthropic_sonnet46,google_gemini25flash_think1024,cohere_command_a_plus,mistral_large
```

Run grouped random experiment:

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv \
  --out-dir outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_grouped_random_balanced_dataset_dimension \
  --split-protocol grouped_random \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

Run leave-one-dataset-out experiment:

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv \
  --out-dir outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_lodo_balanced_dataset_dimension \
  --split-protocol leave_one_dataset_out \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

Run leave-one-answer-source-out experiment:

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv \
  --out-dir outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_loaso_balanced_dataset_dimension \
  --split-protocol leave_one_answer_source_out \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

Run bootstrap weight stability:

```bash
PYTHONPATH=src python scripts/run_judge_weight_bootstrap.py \
  --score-matrix data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv \
  --out-dir outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_bootstrap_weights \
  --n-bootstrap 100 \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

## Tests

Recommended local verification:

```bash
PYTHONPATH=src pytest -q
ruff check .
```

Key coverage:

- Ayers composite-only default.
- Ayers all-dimension diagnostic mode.
- MEDIQA-QA 2019 raw 1-4 medical expert rating normalization.
- TREC LiveQA Medical 2017 official 0-3 and raw 1-4 grade normalization.
- `domain_dimension_v2` prompt content and no gold-label leakage.
- Cohere provider factory behavior.
- v2 required judge list with Gemini Flash thinking1024 and Cohere.
- End-to-end mock pipeline through score matrix and weighted judge combination.

## Known limitations

- MEDIQA-QA 2019 rows without a usable raw 1-4 medical expert rating are skipped.
- TREC LiveQA Medical 2017 rows without official 0-3 score or raw 1-4 grade are skipped.
- The loaders accept common JSON/JSONL/CSV/TSV field names, but unusual local exports may need an
  explicit path or small field-name adapter.
- Real OpenAI, Anthropic, Gemini, Cohere, and Mistral scoring requires valid API credentials and
  may be rate-limited.
- This document does not claim real v2 API scoring has been completed. That claim requires the
  generated judge score files and score matrix to exist from actual `--execute` runs.
