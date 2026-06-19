# Phase 7 v2 Judge Weighting 完整实验报告

生成日期：2026-06-18
实验对象：Phase 7 v2 synthetic human-aligned weighted judge combination
最终 canonical 数据：`data/judge_calibration/canonical_examples_phase7_v2.jsonl`
最终 score matrix：`data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv`
最终主实验目录：`outputs/judge_calibration/phase7_v2_5judge_gemini25flash_think1024_cohere_grouped_random_balanced_dataset_dimension/`

## 1. 实验目标

Phase 7 v2 延续 Phase 7 v1 的 judge calibration layer 设计：在 CCE/OPE 之前，先把多个 LLM judge 的 answer-level 评分组合成一个更接近 human/expert score 的 synthetic human-aligned judge。

v2 相比 v1 的核心变化是：

1. calibration 数据从 2 个来源扩展到 4 个来源，新增 MEDIQA-QA 2019 和 TREC LiveQA Medical 2017；
2. donor judge 从 DeepSeek 替换为 Cohere，最终使用 OpenAI、Anthropic、Gemini、Cohere、Mistral 五个公开 API judge；
3. judge prompt 从 v1 的较细 rubric 改为更通用的 `domain_dimension_v2`，目标是学习一个跨数据集、跨来源、跨领域评分维度都更接近人类评分的统一规则；
4. Google judge 使用 `gemini-2.5-flash` with `thinkingBudget=1024`，不使用容易触发配额限制的 `gemini-2.5-pro`。

核心问题仍然是：

1. 多个 LLM judge 的 convex weighted combination 是否比简单平均或单个 judge 更接近 human/expert score？
2. 权重是否稳定，而不是退化到单一 judge？
3. 新增医疗问答数据后，模型是否仍能跨 dataset 和 answer source 泛化？
4. 哪些数据集、answer source 或 judge 是主要误差来源？

主模型仍采用 constrained weighted ensemble：

```text
pred_i = clip(beta_0 + sum_j w_j * score_ij, 0, 1)

约束：
w_j >= 0
sum_j w_j = 1
```

其中 `score_ij` 是第 `j` 个 donor LLM judge 对第 `i` 个 answer-level example 的评分，`pred_i` 是 synthetic human-aligned judge score。

## 2. 数据设计

### 2.1 Calibration 数据单位

v2 仍使用 answer-level calibration examples。每一行表示一个被评分回答，并带有归一化到 `[0, 1]` 的 human/expert gold score。

canonical 数据文件：

```text
data/judge_calibration/canonical_examples_phase7_v2.jsonl
```

关键字段：

| 字段 | 含义 |
|---|---|
| `calib_example_id` | 稳定唯一 ID |
| `dataset_id` | 数据集来源 |
| `domain` | 领域，例如 `medical` 或 `counseling` |
| `task_type` | 任务类型 |
| `score_dimension` | 评分维度 |
| `question_id` | 问题 ID，用于 grouped split |
| `question` / `context` | 问题与可选上下文 |
| `answer` | 被评分回答 |
| `answer_source` | 回答来源 |
| `human_score` | 归一化到 `[0, 1]` 的 human/expert score |
| `split_group` | 防止同题样本跨 train/val/test 的分组键 |

所有 donor judge prompt 都不包含 `human_score`、`human_score_raw`、`y_score` 或 `y_agent_score` 等 gold label 字段，避免 reward leakage。

### 2.2 数据来源

v2 使用 4 个有 human/expert score 的来源。

| 数据集 | 样本数 | 说明 |
|---|---:|---|
| Ayers AskDocs | 390 | 医疗问答；v2 使用 composite 维度 |
| CounselBench | 400 | 咨询场景；human counselor、GPT-4、LLaMA-3、Gemini answers |
| MEDIQA-QA 2019 | 2179 | 医疗问答；raw 1-4 expert rating 归一化 |
| TREC LiveQA Medical 2017 | 564 | 医疗问答；official 0-3 或 raw 1-4 grade 归一化 |

总样本：

| 项目 | 数值 |
|---|---:|
| canonical answer-level examples | 3533 |
| human_score 最小值 | 0.0 |
| human_score 最大值 | 1.0 |
| duplicates removed | 200 |
| skipped examples | 0 |
| missing files | 0 |

按数据集分布：

| dataset_id | n |
|---|---:|
| `ayers_askdocs` | 390 |
| `counselbench` | 400 |
| `liveqa_medical_2017` | 564 |
| `mediqa_qa_2019` | 2179 |

按评分维度分布：

| score_dimension | n |
|---|---:|
| `composite` | 390 |
| `counseling_quality` | 400 |
| `medical_answer_quality` | 2743 |

按回答来源分布：

| answer_source | n |
|---|---:|
| `chatgpt` | 195 |
| `chiqa` | 2179 |
| `gemini` | 100 |
| `gpt4` | 100 |
| `human_clinician` | 195 |
| `human_counselor` | 100 |
| `llama3` | 100 |
| `trec_liveqa_judged_answer` | 564 |

### 2.3 新增医疗 QA 数据处理

MEDIQA-QA 2019：

| 项目 | 值 |
|---|---|
| source paths | `data/mediqa_qa_2019/`, `data/raw/mediqa_qa_2019/` |
| canonical output | `data/mediqa_qa_2019/answers.jsonl` |
| included examples | 2179 |
| score mapping | `raw 1-4 medical expert rating -> (raw_score - 1) / 3` |
| score_dimension | `medical_answer_quality` |

MEDIQA 过滤记录：

| 项目 | 数值 |
|---|---:|
| input rows | 3042 |
| output rows | 2179 |
| train LiveQA Med overlap rows removed | 839 |
| exact duplicate rows removed | 14 |
| conflicting duplicate rows removed | 10 |

TREC LiveQA Medical 2017：

| 项目 | 值 |
|---|---|
| source paths | `data/liveqa_medical_2017/`, `data/raw/liveqa_medical_2017/` |
| canonical output | `data/liveqa_medical_2017/answers.jsonl` |
| included examples | 564 |
| score mapping | `official 0-3 score -> score / 3` or `raw 1-4 grade -> (grade - 1) / 3` |
| score_dimension | `medical_answer_quality` |

LiveQA 过滤记录：

| 项目 | 数值 |
|---|---:|
| qrels exact duplicate rows removed | 111 |
| input rows | 581 |
| output rows | 564 |
| unusable raw grade rows removed | 17 |

## 3. v2 Prompt 与 Donor LLM Judges

### 3.1 `domain_dimension_v2` prompt

v2 使用 `domain_dimension_v2` prompt。该 prompt 不再为每个数据集写非常细的专门说明，而是使用统一的人类专家评分标准：

```text
Estimate the score that a careful human expert would assign for this benchmark row.
Use the domain, task type, score dimension, question, context, and answer as context,
but apply one general standard across datasets. Reward answers that are correct, safe,
relevant, useful, sufficiently complete, and clearly communicated. Penalize answers
that are incorrect, unsafe, unsupported, irrelevant, incomplete, misleading, or poorly
matched to the user's question. Do not require perfection; assign a calibrated score
reflecting overall answer quality relative to the prompt and context.
```

score anchors：

| score | anchor |
|---:|---|
| 0.0 | dangerous, irrelevant, or severely misleading |
| 0.25 | mostly poor, with major omissions or unsafe framing |
| 0.50 | partially helpful but incomplete or mixed quality |
| 0.75 | good, mostly correct, safe, and helpful with minor issues |
| 1.0 | excellent, safe, accurate, complete, and well-communicated |

该 prompt 会给 judge 提供 `domain`、`task_type`、`score_dimension`、`question`、`context` 和 `answer`，并要求返回：

```json
{"score": <number between 0 and 1>, "rationale": "<brief explanation>"}
```

### 3.2 Donor judges

v2 full run 使用 5 个 donor judge：

| judge_id | provider | model_version | rows | rubric |
|---|---|---|---:|---|
| `openai_gpt55` | OpenAI | `gpt-5.5` | 3533 | `domain_dimension_v2` |
| `anthropic_sonnet46` | Anthropic | `claude-sonnet-4-6` | 3533 | `domain_dimension_v2` |
| `google_gemini25flash_think1024` | Google Gemini | `gemini-2.5-flash` | 3533 | `domain_dimension_v2` |
| `cohere_command_a_plus` | Cohere | `command-a-03-2025` | 3533 | `domain_dimension_v2` |
| `mistral_large` | Mistral | `mistral-large-latest` | 3533 | `domain_dimension_v2` |

说明：

- v1 使用 DeepSeek；v2 改为 Cohere，以增加 cross-provider diversity。
- Google judge 使用 `gemini-2.5-flash`，并使用 thinking budget 1024；没有使用更容易触发配额限制的 `gemini-2.5-pro`。
- OpenAI 在中途曾因账户额度不足停止，充值后续跑成功；最终 OpenAI 覆盖率为 100%。
- Mistral 在早期曾遇到 429 rate limit，因此最终采用保守的 `sleep_seconds=12` 逐分片完成。

### 3.3 Donor score coverage

最终 score matrix 覆盖率：

| judge_id | n_scored | coverage |
|---|---:|---:|
| `openai_gpt55` | 3533 | 1.000 |
| `anthropic_sonnet46` | 3533 | 1.000 |
| `google_gemini25flash_think1024` | 3533 | 1.000 |
| `cohere_command_a_plus` | 3533 | 1.000 |
| `mistral_large` | 3533 | 1.000 |

完整覆盖样本数为 3533，因 judge coverage 过滤删除的样本数为 0。

## 4. Score Matrix 设计

wide score matrix 文件：

```text
data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv
```

矩阵结构：

| 列类型 | 说明 |
|---|---|
| canonical metadata | `dataset_id`、`score_dimension`、`answer_source`、`question_id` 等 |
| gold label | `human_score` |
| donor judge columns | `judge::<judge_id>` |

最终 judge 列：

```text
judge::openai_gpt55
judge::anthropic_sonnet46
judge::google_gemini25flash_think1024
judge::cohere_command_a_plus
judge::mistral_large
```

矩阵行数为 3533，judge 缺失单元格数为 0。

## 5. 模型与超参数

### 5.1 主模型：Simplex Weighted Judge

主模型是带 simplex constraint 的加权组合：

```text
min_{w, beta_0} sum_i v_i * (y_i - beta_0 - S_i w)^2
                 + lambda * ||w - u||_2^2

subject to:
w_j >= 0
sum_j w_j = 1
```

其中：

- `y_i` 是 human/expert gold score；
- `S_i` 是第 `i` 个样本的 donor judge score vector；
- `w` 是 donor judge 权重；
- `u` 是 uniform prior；
- `lambda` 控制向 uniform prior 收缩的强度；
- `v_i` 是 group-balanced sample weight。

### 5.2 Group-balanced loss

v2 数据强烈不平衡：MEDIQA-QA 2019 有 2179 条，而 CounselBench 只有 400 条。因此主实验启用：

```text
--group-balance
```

默认平衡分组为：

```text
dataset_id + "::" + score_dimension
```

这样新增的大规模医疗问答不会完全主导权重学习。

### 5.3 Hyperparameter grid

| 超参 | 设置 |
|---|---|
| `seed` | 0 |
| `train_frac` | 0.6 |
| `val_frac` | 0.2 |
| `test_frac` | 0.2 |
| split group | `dataset_id::question_id::score_dimension` |
| `lambda_grid` | `[0, 0.0001, 0.001, 0.01, 0.1, 1.0]` |
| `fit_intercept_options` | `[true, false]` |
| `clip_predictions` | true |
| judge score normalization | none |
| hyperparameter selection metric | validation RMSE |

### 5.4 Baselines

| baseline | 说明 |
|---|---|
| unweighted mean | 5 个 judge 简单平均 |
| median | 5 个 judge 的中位数 |
| best single judge | 在 validation set 上选择 RMSE 最低的单个 judge |
| ridge stacking | 无约束 ridge regression，作为性能上界参考 |

## 6. 验证协议

v2 已跑以下 protocol：

| protocol | 目的 | 输出目录 |
|---|---|---|
| grouped random | 主实验，验证整体拟合与 baseline 对比 | `phase7_v2_5judge_gemini25flash_think1024_cohere_grouped_random_balanced_dataset_dimension` |
| leave-one-dataset-out | 检查跨数据集泛化 | `phase7_v2_5judge_gemini25flash_think1024_cohere_lodo_balanced_dataset_dimension` |
| leave-one-answer-source-out | 检查回答来源泛化与 source bias | `phase7_v2_5judge_gemini25flash_think1024_cohere_loaso_balanced_dataset_dimension` |
| leave-one-judge sensitivity | 每次移除一个 donor judge 后重训 | 主实验目录内 `leave_one_judge_sensitivity.csv` |
| bootstrap weight stability | 对 train groups 做 100 次 bootstrap | `phase7_v2_5judge_gemini25flash_think1024_cohere_bootstrap_weights` |

说明：v1 报告中包含 leave-one-score-dimension-out。v2 当前 pipeline 未跑该 protocol；本报告只记录已实际完成的实验。

## 7. 主实验结果

### 7.1 最终可用模型

最终主模型：

```text
model_type: simplex_weighted_judge
lambda_reg: 0.0
fit_intercept: false
intercept: 0.0
clip_predictions: true
validation_rmse: 0.205204
max_weight: 0.334587
effective_num_judges: 3.839654
```

最终权重：

| judge_id | weight |
|---|---:|
| `anthropic_sonnet46` | 0.334587 |
| `mistral_large` | 0.244031 |
| `cohere_command_a_plus` | 0.219608 |
| `openai_gpt55` | 0.201774 |
| `google_gemini25flash_think1024` | 0.000000 |

解释：

- 权重没有退化到单一 judge；最大权重为 0.335，effective number of judges 为 3.84。
- Claude、Mistral、Cohere、OpenAI 是主要贡献者。
- Gemini Flash 在当前 v2 组合中被压到近 0，说明它与其他 judge 高相关且没有额外带来稳定校准收益。
- v2 比 v1 更平均地使用 4 个主要 judge；v1 的前三个主力是 Mistral、Claude、OpenAI，v2 中 Cohere 也成为主要 donor。

### 7.2 Hyperparameter selection candidates

| lambda | fit_intercept | validation RMSE |
|---:|---|---:|
| 0.0 | true | 0.206254 |
| 0.0 | false | 0.205204 |
| 0.0001 | true | 0.206271 |
| 0.0001 | false | 0.205214 |
| 0.001 | true | 0.206403 |
| 0.001 | false | 0.205277 |
| 0.01 | true | 0.207039 |
| 0.01 | false | 0.205264 |
| 0.1 | true | 0.212943 |
| 0.1 | false | 0.208867 |
| 1.0 | true | 0.215711 |
| 1.0 | false | 0.210806 |

最终选择 `lambda=0.0` 且 `fit_intercept=false`，因为它的 validation RMSE 最低。

### 7.3 Overall metrics

主实验 test set 为 721 条。

| model | n | RMSE | MAE | mean bias | Pearson | Spearman | calibration slope | calibration intercept | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic judge | 721 | 0.190414 | 0.145058 | -0.016128 | 0.811214 | 0.811802 | 0.919714 | 0.055895 | 0.031409 |
| unweighted mean | 721 | 0.192252 | 0.147146 | -0.014744 | 0.811439 | 0.814154 | 0.878598 | 0.075045 | 0.042611 |
| median | 721 | 0.196494 | 0.147056 | -0.020988 | 0.806190 | 0.806928 | 0.857932 | 0.090667 | 0.050604 |
| best single judge | 721 | 0.222018 | 0.173804 | 0.037861 | 0.740006 | 0.735144 | 0.887067 | 0.024174 | 0.043836 |
| ridge stacking | 721 | 0.189210 | 0.146924 | 0.017542 | 0.812391 | 0.812851 | 0.947082 | 0.010451 | 0.035426 |

结论：

- synthetic judge 是满足 simplex constraint 的主要可用模型。
- synthetic judge 相比 unweighted mean 的 RMSE 从 0.192252 降到 0.190414，约降低 1.0%。
- synthetic judge 相比 best single judge 的 RMSE 从 0.222018 降到 0.190414，约降低 14.2%。
- ridge stacking 的 RMSE 略低，为 0.189210，但它是无约束模型，不作为最终 calibrated reward。
- v2 的 Pearson/Spearman 约 0.81，高于 v1 的约 0.69/0.66，但 v2 引入了大量 MEDIQA/LiveQA medical rows，不能直接解释为同一分布上的提升。

## 8. 分组误差诊断

### 8.1 按数据集

| dataset_id | n | human mean | pred mean | bias | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| `ayers_askdocs` | 74 | 0.603604 | 0.599381 | -0.004222 | 0.158940 | 0.122999 |
| `counselbench` | 84 | 0.691498 | 0.762841 | 0.071343 | 0.156557 | 0.121268 |
| `liveqa_medical_2017` | 106 | 0.242138 | 0.204105 | -0.038033 | 0.233420 | 0.159982 |
| `mediqa_qa_2019` | 457 | 0.525894 | 0.496840 | -0.029053 | 0.189656 | 0.149541 |

观察：

- grouped-random test 中 CounselBench 和 Ayers 的 RMSE 最低，约 0.157-0.159。
- LiveQA Medical 2017 的 RMSE 最高，为 0.233，说明该来源的人工评分尺度或 answer distribution 与其他数据不同。
- MEDIQA-QA 2019 的 RMSE 为 0.190，接近整体 synthetic judge RMSE。

### 8.2 按评分维度

| score_dimension | n | human mean | pred mean | bias | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| `composite` | 74 | 0.603604 | 0.599381 | -0.004222 | 0.158940 | 0.122999 |
| `counseling_quality` | 84 | 0.691498 | 0.762841 | 0.071343 | 0.156557 | 0.121268 |
| `medical_answer_quality` | 563 | 0.472469 | 0.441725 | -0.030744 | 0.198634 | 0.151506 |

观察：

- `medical_answer_quality` 汇总了 MEDIQA 和 LiveQA，是 v2 中最大的维度，也是误差最高的维度。
- `counseling_quality` 平均 bias 为正，说明模型对 CounselBench test rows 有高估倾向。

### 8.3 按回答来源

| answer_source | n | human mean | pred mean | bias | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| `chatgpt` | 37 | 0.727477 | 0.683640 | -0.043838 | 0.176002 | 0.133676 |
| `chiqa` | 457 | 0.525894 | 0.496840 | -0.029053 | 0.189656 | 0.149541 |
| `gemini` | 21 | 0.649762 | 0.758984 | 0.109222 | 0.145345 | 0.123689 |
| `gpt4` | 21 | 0.748333 | 0.705033 | -0.043300 | 0.113401 | 0.086922 |
| `human_clinician` | 37 | 0.479730 | 0.515123 | 0.035393 | 0.139811 | 0.112323 |
| `human_counselor` | 21 | 0.523214 | 0.725163 | 0.201949 | 0.236983 | 0.201949 |
| `llama3` | 21 | 0.844683 | 0.862185 | 0.017502 | 0.088846 | 0.072510 |
| `trec_liveqa_judged_answer` | 106 | 0.242138 | 0.204105 | -0.038033 | 0.233420 | 0.159982 |

观察：

- `human_counselor` 的 bias 和 RMSE 最高，但 test n 只有 21。
- `trec_liveqa_judged_answer` 的 RMSE 也很高，说明 LiveQA 的评分尺度或 answer construction 是主要误差来源之一。
- `llama3` 和 `gpt4` 在 grouped-random test 中较容易预测。

### 8.4 最差分组

| group | n | bias | RMSE | MAE | 说明 |
|---|---:|---:|---:|---:|---|
| `human_counselor` answer source | 21 | 0.201949 | 0.236983 | 0.201949 | 小样本且明显高估 |
| `liveqa_medical_2017` | 106 | -0.038033 | 0.233420 | 0.159982 | 最大医疗 QA 难点之一 |
| `trec_liveqa_judged_answer` | 106 | -0.038033 | 0.233420 | 0.159982 | 与 LiveQA dataset 相同 |
| `medical_answer_quality` | 563 | -0.030744 | 0.198634 | 0.151506 | 大规模医疗 answer quality 汇总维度 |
| `mediqa_qa_2019` | 457 | -0.029053 | 0.189656 | 0.149541 | 大样本来源，决定整体表现 |

## 9. Leave-One-Dataset-Out 结果

| held-out dataset | test n | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `ayers_askdocs` | 390 | 0.166691 | 0.130584 | 0.014562 | 0.0 | true | 0.321760 | 3.735777 |
| `counselbench` | 400 | 0.169751 | 0.132202 | 0.093985 | 0.0 | true | 0.401594 | 3.502603 |
| `liveqa_medical_2017` | 564 | 0.217610 | 0.150009 | -0.002668 | 0.01 | false | 0.291431 | 3.920193 |
| `mediqa_qa_2019` | 2179 | 0.214966 | 0.169456 | -0.097476 | 0.001 | true | 0.345764 | 3.802535 |

解释：

- 留出 Ayers 和 CounselBench 时 RMSE 约 0.167-0.170，说明从医疗 QA 与另一部分数据迁移到这两个旧数据集并不困难。
- 留出 LiveQA 或 MEDIQA 时 RMSE 升到约 0.215-0.218，说明新增医疗 QA 来源之间仍有明显 dataset shift。
- 留出 MEDIQA 时 bias 为 -0.097，方向性低估明显。

## 10. Leave-One-Answer-Source-Out 结果

| held-out answer_source | test n | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `chatgpt` | 195 | 0.166580 | 0.127079 | -0.051622 | 0.01 | false | 0.349323 | 3.768276 |
| `chiqa` | 2179 | 0.214966 | 0.169456 | -0.097476 | 0.001 | true | 0.345764 | 3.802535 |
| `gemini` | 100 | 0.188126 | 0.159265 | 0.146966 | 0.01 | false | 0.259863 | 4.203570 |
| `gpt4` | 100 | 0.109649 | 0.086863 | -0.036786 | 0.001 | false | 0.392099 | 3.531576 |
| `human_clinician` | 195 | 0.174972 | 0.140298 | 0.086995 | 0.01 | true | 0.278249 | 4.178159 |
| `human_counselor` | 100 | 0.234227 | 0.205719 | 0.200607 | 0.1 | false | 0.222242 | 4.897253 |
| `llama3` | 100 | 0.077241 | 0.062834 | -0.003019 | 0.001 | false | 0.345940 | 3.760846 |
| `trec_liveqa_judged_answer` | 564 | 0.217610 | 0.150009 | -0.002668 | 0.01 | false | 0.291431 | 3.920193 |

解释：

- `human_counselor` 是最难泛化的 answer source，RMSE 0.234，bias 0.201。
- `chiqa` 与 MEDIQA 绑定，留出后 RMSE 0.215，bias -0.097。
- `trec_liveqa_judged_answer` 与 LiveQA 绑定，留出后 RMSE 0.218。
- `llama3` 留出结果最好，RMSE 0.077。

## 11. Leave-One-Judge Sensitivity

| dropped judge | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---|---:|---:|
| `openai_gpt55` | 0.190903 | 0.145925 | -0.009297 | 0.01 | false | 0.398518 | 3.162375 |
| `anthropic_sonnet46` | 0.194271 | 0.147984 | 0.002883 | 0.01 | false | 0.434073 | 3.031431 |
| `google_gemini25flash_think1024` | 0.190835 | 0.145235 | -0.009821 | 1.0 | false | 0.251185 | 3.999965 |
| `cohere_command_a_plus` | 0.192407 | 0.144318 | -0.022430 | 0.0 | false | 0.366645 | 2.970860 |
| `mistral_large` | 0.191113 | 0.147358 | -0.017846 | 0.0 | true | 0.378199 | 2.972932 |

解释：

- 移除 Anthropic 后 RMSE 上升最多，说明 Claude 是 v2 中最重要的 single donor。
- 移除 Gemini 后 RMSE 几乎不变，符合主模型和 bootstrap 中 Gemini 权重接近 0 的现象。
- 移除 OpenAI、Cohere、Mistral 后 RMSE 小幅上升，说明三者有一定互补贡献。

## 12. Bootstrap Weight Stability

对 train groups 做 100 次 bootstrap，每次重新拟合权重。

| judge_id | weight mean | std | p2.5 | median | p97.5 | zero weight rate |
|---|---:|---:|---:|---:|---:|---:|
| `anthropic_sonnet46` | 0.320284 | 0.054850 | 0.225469 | 0.311183 | 0.426244 | 0.00 |
| `mistral_large` | 0.244457 | 0.037957 | 0.179036 | 0.240689 | 0.324328 | 0.00 |
| `cohere_command_a_plus` | 0.233229 | 0.034199 | 0.185067 | 0.228349 | 0.320431 | 0.00 |
| `openai_gpt55` | 0.199247 | 0.065579 | 0.064593 | 0.215882 | 0.282399 | 0.00 |
| `google_gemini25flash_think1024` | 0.002783 | 0.006412 | 0.000000 | 0.000000 | 0.023357 | 0.79 |

解释：

- Anthropic、Mistral、Cohere、OpenAI 的 bootstrap zero weight rate 都是 0，说明它们是稳定贡献者。
- Gemini 的 zero weight rate 为 0.79，说明它在多数 bootstrap replicate 中被置 0。
- v2 最稳定的排序是 Claude > Mistral > Cohere > OpenAI >> Gemini。

## 13. Judge Correlation

donor judge score 之间的 Pearson correlation：

|  | OpenAI | Claude | Gemini Flash | Cohere | Mistral |
|---|---:|---:|---:|---:|---:|
| OpenAI | 1.000 | 0.947 | 0.910 | 0.885 | 0.913 |
| Claude | 0.947 | 1.000 | 0.897 | 0.877 | 0.902 |
| Gemini Flash | 0.910 | 0.897 | 1.000 | 0.828 | 0.858 |
| Cohere | 0.885 | 0.877 | 0.828 | 1.000 | 0.887 |
| Mistral | 0.913 | 0.902 | 0.858 | 0.887 | 1.000 |

观察：

- v2 donor judges 之间整体高度相关，最高是 OpenAI-Claude 的 0.947。
- Gemini 与其他 judge 也高度相关，但在 human calibration 上没有提供独立收益，因此主权重接近 0。
- Cohere 与 Mistral、OpenAI、Claude 均有高相关，但 bootstrap 中仍保留稳定正权重，说明它贡献的是校准方向而不是简单 decorrelation。

## 14. 最好的结果如何理解

从“可作为 calibrated reward 的主模型”角度，最佳结果是 constrained simplex synthetic judge：

```text
RMSE: 0.190414
MAE: 0.145058
mean bias: -0.016128
Pearson: 0.811214
Spearman: 0.811802
effective_num_judges: 3.840
```

它优于：

| 对比对象 | RMSE | 相比 synthetic judge |
|---|---:|---:|
| synthetic judge | 0.190414 | baseline |
| unweighted mean | 0.192252 | synthetic judge 更低约 1.0% |
| best single judge | 0.222018 | synthetic judge 更低约 14.2% |
| median | 0.196494 | synthetic judge 更低约 3.1% |

从“纯预测上界”角度，ridge stacking 的 RMSE 为 0.189210，略低于 constrained synthetic judge。但 ridge stacking 是无约束模型，包含自由截距，不满足本阶段 synthetic-control-inspired convex combination 的约束，因此不作为最终 reward source。

ridge stacking 系数：

| judge | coefficient |
|---|---:|
| OpenAI | 0.237571 |
| Claude | 0.329136 |
| Gemini Flash | 0.002435 |
| Cohere | 0.150965 |
| Mistral | 0.244254 |
| intercept | 0.055856 |

## 15. v2 与 v1 的主要差异

| 项目 | v1 | v2 |
|---|---|---|
| canonical examples | 1570 | 3533 |
| 数据集 | Ayers、CounselBench | Ayers、CounselBench、MEDIQA-QA 2019、LiveQA Medical 2017 |
| Ayers 维度 | composite、quality、empathy | composite |
| 新增医学 QA | 无 | MEDIQA-QA 2019、TREC LiveQA Medical 2017 |
| donor judge 第 4 个 | DeepSeek | Cohere |
| prompt | `generic_v1` | `domain_dimension_v2` |
| grouped-random test n | 320 | 721 |
| synthetic RMSE | 0.178066 | 0.190414 |
| synthetic Pearson | 0.689677 | 0.811214 |
| effective judges | 3.498 | 3.840 |

注意：v1 与 v2 的数据分布不同，不能把 RMSE 变化直接解释为模型退步或进步。v2 的目标是扩大 calibration set 并提高跨数据源代表性。

## 16. Calibrated Reward 写回状态

v1 已完成 answer-level synthetic judge predictions 和 paired calibrated JSONL 写回。

v2 当前已完成：

1. v2 canonical dataset；
2. 5 个真实 donor judge 的完整 scoring；
3. v2 score matrix；
4. grouped random / LODO / LOASO / bootstrap weighting experiments。

v2 尚未执行：

1. `apply_synthetic_judge_scores.py` 写出 v2 synthetic judge predictions；
2. paired calibrated JSONL 写回；
3. 后续接入 CCE/OPE 的 reward-source 对比。

建议下一步使用 v2 的 score matrix 和 grouped-random weights 生成：

```text
data/judge_calibration/synthetic_judge_predictions_phase7_v2_5judge_gemini25flash_think1024_cohere.jsonl
```

## 17. 可复现实验命令

### 17.1 构建 v2 canonical calibration dataset

```bash
PYTHONPATH=src python scripts/build_judge_calibration_dataset.py \
  --repo-root . \
  --out data/judge_calibration/canonical_examples_phase7_v2.jsonl \
  --summary-out outputs/judge_calibration/data_summary_phase7_v2.json \
  --include-mediqa-qa-2019 \
  --include-liveqa-medical-2017
```

### 17.2 运行 donor judge scoring pipeline

```bash
PYTHONPATH=src python scripts/run_phase7_v2_pipeline.py \
  --poll-seconds 30
```

如果只需补跑某个 judge，可以使用：

```bash
PYTHONPATH=src python scripts/run_phase7_v2_pipeline.py \
  --poll-seconds 30 \
  --score-only \
  --only-judges openai_gpt55
```

### 17.3 跳过 scoring，直接合并分片、构建矩阵并跑 weighting experiments

```bash
PYTHONPATH=src python scripts/run_phase7_v2_pipeline.py \
  --skip-scoring
```

该命令会生成：

```text
data/judge_calibration/judge_scores/openai_gpt55_phase7_v2.jsonl
data/judge_calibration/judge_scores/anthropic_sonnet46_phase7_v2.jsonl
data/judge_calibration/judge_scores/gemini25flash_think1024_phase7_v2.jsonl
data/judge_calibration/judge_scores/cohere_command_a_plus_phase7_v2.jsonl
data/judge_calibration/judge_scores/mistral_large_phase7_v2.jsonl
data/judge_calibration/score_matrix_phase7_v2_5judge_gemini25flash_think1024_cohere.csv
outputs/judge_calibration/phase7_v2_missingness_report.json
```

### 17.4 主 grouped-random 实验

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

### 17.5 LODO

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

### 17.6 LOASO

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

### 17.7 Bootstrap weight stability

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

## 18. 代码与测试状态

v2 相关新增或更新模块：

| 文件 | 作用 |
|---|---|
| `scripts/download_medical_qa_calibration_sources.py` | 下载并转换 MEDIQA-QA 2019 与 LiveQA Medical 2017 |
| `scripts/build_judge_calibration_dataset.py` | 构建 v2 canonical examples |
| `scripts/run_phase7_v2_pipeline.py` | v2 scoring、合并、矩阵构建和实验 orchestration |
| `scripts/run_donor_judge_scoring.py` | donor judge API scoring 与 cache |
| `src/cce_data/judge_calibration/canonicalize.py` | 新增 MEDIQA/LiveQA canonicalization 与 score normalization |
| `src/cce_data/judge_calibration/judge_prompting.py` | `domain_dimension_v2` prompt |
| `src/cce_data/judge_calibration/judge_clients.py` | OpenAI/Anthropic/Gemini/Cohere/Mistral clients |
| `src/cce_data/judge_calibration/weighted_ensemble.py` | simplex weighted judge 与 baselines |
| `src/cce_data/judge_calibration/stability.py` | bootstrap weight stability |

本报告生成时，已完成的 pipeline 验证包括：

```text
PYTHONPATH=src python scripts/run_phase7_v2_pipeline.py --skip-scoring
```

该命令成功完成：

1. 五个 judge score 文件合并；
2. 3533 行 v2 score matrix 构建；
3. missingness report；
4. grouped random；
5. LODO；
6. LOASO；
7. 100 次 bootstrap。

## 19. 结论

Phase 7 v2 完成了四数据源、真实 5-judge 的 human-aligned judge weighting 实验闭环。主要结论：

1. **五个 donor judge 全量覆盖成功**：3533 个 examples 全部有 OpenAI、Claude、Gemini、Cohere、Mistral 分数，coverage drop 为 0。
2. **weighted judge combination 仍有收益**：constrained synthetic judge 的 RMSE 为 0.1904，优于 unweighted mean 的 0.1923、median 的 0.1965 和 best single judge 的 0.2220。
3. **主模型没有退化成单 judge**：最大权重 0.335，effective number of judges 3.84。
4. **v2 主要贡献来自 Claude、Mistral、Cohere、OpenAI**：四者在 bootstrap 中 zero weight rate 均为 0。
5. **Gemini Flash 在 v2 中边际贡献接近 0**：主模型权重约 0，bootstrap zero weight rate 0.79。
6. **新增医疗 QA 带来明显 domain/source shift**：LiveQA 和 MEDIQA 的 LODO RMSE 约 0.215-0.218，高于 Ayers/CounselBench 的 0.167-0.170。
7. **source bias 仍需关注**：`human_counselor`、`chiqa`、`trec_liveqa_judged_answer` 是主要 held-out 风险来源。
8. **下一步应写出 v2 calibrated reward**：当前已完成 score matrix 和 weighting experiments，但 v2 answer-level synthetic predictions 与 paired calibrated JSONL 尚未写回。

因此，v2 的主要价值是把 Phase 7 judge calibration 从两个小规模数据源扩展到四个更丰富的数据源，并验证在更大、更不均衡的 medical/counseling mixed calibration set 上，constrained weighted judge 仍然能形成稳定、可解释、接近 human/expert score 的组合评分器。
