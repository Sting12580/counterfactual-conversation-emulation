# Phase 7 Judge Weighting 完整实验报告

生成日期：2026-06-17
实验对象：Synthetic human-aligned weighted judge combination
最终数据矩阵：`data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv`
最终主实验目录：`outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_grouped_random_balanced_dataset_dimension/`

## 1. 实验目标

本阶段的目标不是直接替换 Phase 5/Phase 6 的 OPE estimator，而是在 OPE 之前新增一个独立的 judge calibration layer。该层把多个 LLM judge 的打分视为 donor judge scores，把已有 human/expert score 视为 gold label，学习一个更接近人类评分的 synthetic human-aligned judge。

核心问题是：

1. 多个 LLM judge 的 convex weighted combination 是否能比单个 judge 更接近 human/expert score？
2. 加权组合是否能降低不同数据集、评分维度和回答来源上的系统偏差？
3. 学到的 judge 权重是否稳定，而不是退化成单一 judge？
4. 校准后的 synthetic judge score 是否能作为后续 CCE/OPE 的 reward source 写回数据文件？

最终主模型采用 constrained weighted ensemble，即：

```text
pred_i = clip(beta_0 + sum_j w_j * score_ij, 0, 1)

约束：
w_j >= 0
sum_j w_j = 1
```

其中 `score_ij` 是第 `j` 个 LLM judge 对第 `i` 个 answer-level example 的评分，`pred_i` 是 synthetic judge score。

## 2. 数据设计

### 2.1 Calibration 数据单位

实验数据被整理为 answer-level calibration examples，而不是 paired OPE rows。每一行代表“一个问题/上下文下的一个回答”，并带有 human/expert gold score。

canonical 数据文件：

```text
data/judge_calibration/canonical_examples.jsonl
```

关键字段包括：

| 字段 | 含义 |
|---|---|
| `calib_example_id` | 稳定唯一 ID |
| `dataset_id` | 数据集来源 |
| `domain` | 领域，例如 medical 或 counseling |
| `task_type` | 任务类型 |
| `score_dimension` | 评分维度 |
| `question_id` | 问题 ID，用于 grouped split |
| `question` / `context` | 问题与可选上下文 |
| `answer` | 被评分回答 |
| `answer_source` | 回答来源，例如 human、ChatGPT、GPT-4 |
| `human_score` | 归一化到 `[0, 1]` 的 human/expert score |
| `split_group` | 防止同题样本跨 train/val/test 的分组键 |

### 2.2 数据来源

第一轮 full experiment 只使用已有 human/expert-scored 数据，不使用只有 LLM judge score 的数据作为 gold label。

| 数据集 | 样本数 | 说明 |
|---|---:|---|
| Ayers AskDocs | 1170 | 医疗问答，包含 clinician answer 和 ChatGPT answer；按 composite、quality、empathy 三个评分维度拆分 |
| CounselBench | 400 | 咨询场景，包含 human counselor baseline 和 GPT-4 / LLaMA-3 / Gemini responder answer |

总样本：

| 项目 | 数值 |
|---|---:|
| canonical answer-level examples | 1570 |
| human_score 最小值 | 0.0 |
| human_score 最大值 | 1.0 |
| CounselBench human baseline 去重数 | 200 |
| 缺失源文件 | 0 |

按数据集分布：

| dataset_id | n |
|---|---:|
| `ayers_askdocs` | 1170 |
| `counselbench` | 400 |

按评分维度分布：

| score_dimension | n |
|---|---:|
| `composite` | 390 |
| `quality` | 390 |
| `empathy` | 390 |
| `counseling_quality` | 400 |

按回答来源分布：

| answer_source | n |
|---|---:|
| `human_clinician` | 585 |
| `chatgpt` | 585 |
| `human_counselor` | 100 |
| `gpt4` | 100 |
| `llama3` | 100 |
| `gemini` | 100 |

## 3. Donor LLM Judges

最终 full run 使用 5 个公开 API judge。每个 judge 对 1570 个 canonical examples 都给出了一个 `[0, 1]` 分数，并返回 JSON。

| judge_id | provider | model_version | rows | rubric |
|---|---|---|---:|---|
| `openai_gpt55` | OpenAI | `gpt-5.5` | 1570 | `generic_v1` |
| `anthropic_sonnet46` | Anthropic | `claude-sonnet-4-6` | 1570 | `generic_v1` |
| `google_gemini25flash_think1024` | Google Gemini | `gemini-2.5-flash` | 1570 | `generic_v1` |
| `deepseek_v4flash` | DeepSeek | `deepseek-v4-flash` | 1570 | `generic_v1` |
| `mistral_large` | Mistral | `mistral-large-latest` | 1570 | `generic_v1` |

说明：

- 原计划中曾尝试使用 Google `gemini-2.5-pro`，但触发每日请求配额限制，未能完成 1570 条 full run。
- 后续测试过 Google `gemini-3.5-flash`，模型可连通，但在实验当天存在 high-demand/长尾响应问题，不适合批量跑完整 1570 条。
- 最终采用 Google `gemini-2.5-flash`，并设置 `thinkingBudget=1024`，在质量和吞吐之间做折中。
- DeepSeek 调用时关闭额外 thinking 输出，以保证 judge 返回结构化 JSON 分数。
- 所有 judge prompt 都不包含 human_score，避免 reward leakage。

### 3.1 Donor score coverage

最终 score matrix 覆盖率：

| judge_id | n_scored | coverage |
|---|---:|---:|
| `openai_gpt55` | 1570 | 1.000 |
| `anthropic_sonnet46` | 1570 | 1.000 |
| `google_gemini25flash_think1024` | 1570 | 1.000 |
| `deepseek_v4flash` | 1570 | 1.000 |
| `mistral_large` | 1570 | 1.000 |

完整覆盖样本数为 1570，因 judge coverage 过滤删除的样本数为 0。

## 4. Score Matrix 设计

wide score matrix 文件：

```text
data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv
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
judge::deepseek_v4flash
judge::mistral_large
```

矩阵行数为 1570，judge 缺失单元格数为 0。

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
- `u` 是 uniform prior，即每个 judge 初始先验权重为 `1 / M`；
- `lambda` 控制向 uniform prior 收缩的强度；
- `v_i` 是 group-balanced sample weight。

### 5.2 Group-balanced loss

主实验使用 plan 推荐的 group-balanced loss。因为 Ayers AskDocs 有 1170 条，CounselBench 有 400 条，如果直接用普通 MSE，Ayers 会主导训练。因此主实验启用：

```text
--group-balance
```

默认平衡分组为：

```text
dataset_id + "::" + score_dimension
```

这样 composite、quality、empathy、counseling_quality 不会因为样本量差异而完全主导权重。

### 5.3 Hyperparameter grid

主实验使用固定 seed 和 validation RMSE 选择超参。

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

每个 split 同时比较以下 baseline：

| baseline | 说明 |
|---|---|
| unweighted mean | 5 个 judge 简单平均 |
| median | 5 个 judge 的中位数 |
| best single judge | 在 validation set 上选择 RMSE 最低的单个 judge |
| ridge stacking | 无约束 ridge regression，允许负权重和自由系数，仅作为性能上界参考 |

注意：ridge stacking 数值上可能更低，但它不满足 `w_j >= 0` 和 `sum_j w_j = 1`，并且本次学到了 DeepSeek 的负系数。因此它不作为最终 synthetic judge reward，只作为 upper baseline。

## 6. 验证协议

完整实验跑了以下 protocol：

| protocol | 目的 | 输出目录 |
|---|---|---|
| grouped random | 主实验，验证整体拟合和 baseline 对比 | `real_5judge_gemini25flash_think1024_full_grouped_random_balanced_dataset_dimension` |
| leave-one-dataset-out | 检查跨数据集泛化 | `real_5judge_gemini25flash_think1024_full_lodo_balanced_dataset_dimension` |
| leave-one-score-dimension-out | 检查对 held-out rubric/dimension 的泛化 | `real_5judge_gemini25flash_think1024_full_losdo_balanced_dataset_dimension` |
| leave-one-answer-source-out | 检查对回答来源的 source bias | `real_5judge_gemini25flash_think1024_full_loaso_balanced_dataset_dimension` |
| leave-one-judge sensitivity | 每次移除一个 donor judge 后重训 | 主实验目录内 `leave_one_judge_sensitivity.csv` |
| bootstrap weight stability | 对 train groups 做 100 次 bootstrap | `real_5judge_gemini25flash_think1024_full_bootstrap_weights` |

## 7. 主实验结果

### 7.1 最终可用模型

最终主模型为：

```text
model_type: simplex_weighted_judge
lambda_reg: 0.01
fit_intercept: false
intercept: 0.0
clip_predictions: true
validation_rmse: 0.180055
max_weight: 0.341205
effective_num_judges: 3.498274
```

最终权重：

| judge_id | weight |
|---|---:|
| `mistral_large` | 0.341205 |
| `anthropic_sonnet46` | 0.307647 |
| `openai_gpt55` | 0.263679 |
| `google_gemini25flash_think1024` | 0.070526 |
| `deepseek_v4flash` | 0.016943 |

解释：

- 权重没有退化到单一 judge；最大权重为 0.341，effective number of judges 为 3.50。
- Mistral、Claude、OpenAI 是主要贡献者。
- Google Gemini Flash 获得稳定的小正权重。
- DeepSeek 权重接近 0，说明在当前数据与约束下贡献有限。

### 7.2 Overall metrics

主实验 test set 为 320 条。

| model | n | RMSE | MAE | mean bias | Pearson | Spearman | calibration slope | calibration intercept | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| synthetic judge | 320 | 0.178066 | 0.141085 | 0.004257 | 0.689677 | 0.663158 | 0.693247 | 0.180974 | 0.066710 |
| unweighted mean | 320 | 0.191250 | 0.152042 | 0.028502 | 0.657550 | 0.632164 | 0.646521 | 0.193514 | 0.070378 |
| median | 320 | 0.211434 | 0.167294 | 0.029883 | 0.645697 | 0.603042 | 0.553822 | 0.250972 | 0.104701 |
| best single judge | 320 | 0.195830 | 0.153794 | -0.040711 | 0.651521 | 0.618419 | 0.635676 | 0.244322 | 0.090898 |
| ridge stacking | 320 | 0.161834 | 0.131731 | 0.004276 | 0.700945 | 0.673638 | 0.951244 | 0.025166 | 0.015048 |

结论：

- 在满足 simplex constraint 的可用模型中，synthetic judge 是最佳结果。
- synthetic judge 相比 unweighted mean 的 RMSE 从 0.191250 降到 0.178066，约降低 6.9%。
- synthetic judge 相比 best single judge 的 RMSE 从 0.195830 降到 0.178066，约降低 9.1%。
- ridge stacking 的 RMSE 更低，但它是无约束上界，不作为最终 reward source。

### 7.3 Hyperparameter selection candidates

validation RMSE 选择结果：

| lambda | fit_intercept | validation RMSE |
|---:|---|---:|
| 0.0 | true | 0.181091 |
| 0.0 | false | 0.181187 |
| 0.0001 | true | 0.181071 |
| 0.0001 | false | 0.181167 |
| 0.001 | true | 0.180906 |
| 0.001 | false | 0.180998 |
| 0.01 | true | 0.180075 |
| 0.01 | false | 0.180055 |
| 0.1 | true | 0.182762 |
| 0.1 | false | 0.182265 |
| 1.0 | true | 0.185268 |
| 1.0 | false | 0.185991 |

最终选择 `lambda=0.01` 且 `fit_intercept=false`，因为它的 validation RMSE 最低。

## 8. 分组误差诊断

### 8.1 按数据集

| dataset_id | n | human mean | pred mean | bias | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| `ayers_askdocs` | 228 | 0.574013 | 0.554290 | -0.019723 | 0.188766 | 0.149723 |
| `counselbench` | 92 | 0.662962 | 0.726649 | 0.063687 | 0.148258 | 0.119679 |

CounselBench 的 test RMSE 更低，但 bias 为正，说明 synthetic judge 在该 split 上倾向于给 CounselBench 稍高分。Ayers 的 RMSE 更高，说明医疗问答中的评分维度更难拟合，尤其 quality 维度。

### 8.2 按评分维度

| score_dimension | n | human mean | pred mean | bias | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| `composite` | 74 | 0.578266 | 0.608536 | 0.030270 | 0.171763 | 0.130667 |
| `counseling_quality` | 92 | 0.662962 | 0.726649 | 0.063687 | 0.148258 | 0.119679 |
| `empathy` | 82 | 0.485772 | 0.497829 | 0.012056 | 0.157620 | 0.135772 |
| `quality` | 72 | 0.670139 | 0.562840 | -0.107299 | 0.232851 | 0.185197 |

`quality` 是主实验中最困难的评分维度：RMSE 最高，bias 为 -0.107，表示模型在该维度上明显低估 human/expert score。

### 8.3 按回答来源

| answer_source | n | human mean | pred mean | bias | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| `chatgpt` | 114 | 0.715643 | 0.683144 | -0.032499 | 0.168130 | 0.135846 |
| `gemini` | 23 | 0.669130 | 0.800907 | 0.131776 | 0.179751 | 0.158132 |
| `gpt4` | 23 | 0.620761 | 0.575508 | -0.045253 | 0.125083 | 0.101397 |
| `human_clinician` | 114 | 0.432383 | 0.425435 | -0.006948 | 0.207358 | 0.163600 |
| `human_counselor` | 23 | 0.502500 | 0.665987 | 0.163487 | 0.188098 | 0.163695 |
| `llama3` | 23 | 0.859457 | 0.864194 | 0.004737 | 0.067713 | 0.055491 |

回答来源维度上，`llama3` 最容易预测，`human_clinician` RMSE 较高，`human_counselor` 和 `gemini` 有较明显正 bias。

### 8.4 最差分组

| group | n | bias | RMSE | MAE | 说明 |
|---|---:|---:|---:|---:|---|
| `quality` | 72 | -0.107299 | 0.232851 | 0.185197 | 最难评分维度，明显低估 |
| `ayers_askdocs::quality` | 72 | -0.107299 | 0.232851 | 0.185197 | 与上面相同，集中在 Ayers quality |
| `human_clinician` | 114 | -0.006948 | 0.207358 | 0.163600 | RMSE 高但平均 bias 小 |
| `human_counselor` | 23 | 0.163487 | 0.188098 | 0.163695 | 小样本且正 bias 明显 |
| `gemini` answer source | 23 | 0.131776 | 0.179751 | 0.158132 | 小样本且正 bias 明显 |

最主要的 error-analysis 入口是 Ayers 的 `quality` 维度。它不仅 RMSE 高，而且方向性偏差明显。

## 9. Leave-One-Dataset-Out 结果

LODO 用于检查跨数据集泛化能力。每次完整留出一个 dataset 作为 test，其余 dataset 用于 train/validation。

| held-out dataset | test n | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `ayers_askdocs` | 1170 | 0.217596 | 0.172272 | -0.115409 | 0.0 | true | 0.351753 | 3.782473 |
| `counselbench` | 400 | 0.177599 | 0.141355 | 0.100869 | 0.0 | true | 0.346077 | 3.186784 |

解释：

- 留出 Ayers 时 RMSE 更高，说明从 CounselBench 迁移到医疗问答更难。
- 两个 LODO split 的 bias 方向相反：留出 Ayers 时低估，留出 CounselBench 时高估。
- 这说明跨数据集 calibration 仍存在 domain shift，不能只看 grouped-random 的总体 RMSE。

## 10. Leave-One-Score-Dimension-Out 结果

该 protocol 检查模型是否依赖某个具体评分 rubric。

| held-out score_dimension | test n | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `composite` | 390 | 0.177847 | 0.136922 | 0.011159 | 0.01 | false | 0.304442 | 3.787675 |
| `counseling_quality` | 400 | 0.177599 | 0.141355 | 0.100869 | 0.0 | true | 0.346077 | 3.186784 |
| `empathy` | 390 | 0.186196 | 0.146827 | 0.037222 | 0.0 | false | 0.445477 | 2.846917 |
| `quality` | 390 | 0.210955 | 0.171121 | -0.131473 | 0.0 | true | 0.365025 | 3.092058 |

结论：

- `quality` 维度仍然是最困难的 held-out dimension，RMSE 最高且低估明显。
- `composite` 和 `counseling_quality` 的 RMSE 接近 0.178。
- `empathy` 中等难度。

## 11. Leave-One-Answer-Source-Out 结果

该 protocol 检查模型对回答来源是否存在 source bias。

| held-out answer_source | test n | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `chatgpt` | 585 | 0.199409 | 0.160040 | -0.101127 | 0.0 | true | 0.437980 | 3.032503 |
| `gemini` | 100 | 0.210810 | 0.182561 | 0.172456 | 0.0 | true | 0.386042 | 3.136614 |
| `gpt4` | 100 | 0.141861 | 0.110098 | -0.046334 | 0.0 | true | 0.352375 | 3.043041 |
| `human_clinician` | 585 | 0.190887 | 0.149769 | 0.014706 | 0.0 | true | 0.457876 | 2.847073 |
| `human_counselor` | 100 | 0.220960 | 0.186681 | 0.178003 | 0.0 | true | 0.391208 | 3.164910 |
| `llama3` | 100 | 0.080639 | 0.063052 | 0.004365 | 0.0 | true | 0.335458 | 3.226002 |

结论：

- 留出 `human_counselor` 和 `gemini` answer source 时，RMSE 和正 bias 较高。
- 留出 `llama3` 时效果最好，RMSE 仅 0.080639。
- `chatgpt` 和 `human_clinician` 作为大样本 answer source，留出后 RMSE 分别为 0.199 和 0.191，说明模型对回答来源仍存在一定分布依赖。

## 12. Leave-One-Judge Sensitivity

每次移除一个 donor judge 后，在同一 grouped-random protocol 下重新训练 constrained model。

| dropped judge | RMSE | MAE | bias | selected lambda | fit intercept | max weight | effective judges |
|---|---:|---:|---:|---:|---|---:|---:|
| `openai_gpt55` | 0.183516 | 0.143516 | 0.005939 | 0.01 | false | 0.456823 | 2.759017 |
| `anthropic_sonnet46` | 0.182350 | 0.145675 | 0.029289 | 0.001 | false | 0.621737 | 1.926831 |
| `google_gemini25flash_think1024` | 0.177640 | 0.140793 | 0.003666 | 0.01 | true | 0.358970 | 3.104678 |
| `deepseek_v4flash` | 0.177713 | 0.140947 | 0.004263 | 0.01 | true | 0.347575 | 3.384190 |
| `mistral_large` | 0.183155 | 0.147437 | 0.002380 | 0.01 | true | 0.447354 | 2.909676 |

解释：

- 移除 Google 或 DeepSeek 后 RMSE 几乎不变，符合它们在最终模型中权重较小的现象。
- 移除 OpenAI、Claude 或 Mistral 后 RMSE 上升，说明主要性能来自这三个 judge 的互补。
- 移除 Claude 后 effective number of judges 降到 1.93，组合更接近单 judge，因此 Claude 对 ensemble diversity 有贡献。

## 13. Bootstrap Weight Stability

对 train groups 做 100 次 bootstrap，每次重新拟合权重。bootstrap 使用：

| 设置 | 值 |
|---|---|
| n_bootstrap | 100 |
| seed | 0 |
| train/val/test | 0.6 / 0.2 / 0.2 |
| group_col | `split_group` |
| group_balance | true |
| lambda_grid | `[0, 0.0001, 0.001, 0.01, 0.1, 1.0]` |
| fit_intercept_options | `[true, false]` |

结果：

| judge_id | weight mean | std | p2.5 | median | p97.5 | zero weight rate |
|---|---:|---:|---:|---:|---:|---:|
| `mistral_large` | 0.334353 | 0.024149 | 0.291200 | 0.337873 | 0.370280 | 0.00 |
| `anthropic_sonnet46` | 0.314849 | 0.025395 | 0.267202 | 0.313298 | 0.359647 | 0.00 |
| `openai_gpt55` | 0.262238 | 0.028846 | 0.215477 | 0.259849 | 0.321500 | 0.00 |
| `google_gemini25flash_think1024` | 0.068004 | 0.024043 | 0.011905 | 0.069043 | 0.109560 | 0.00 |
| `deepseek_v4flash` | 0.020556 | 0.021418 | 0.000000 | 0.017288 | 0.060796 | 0.25 |

解释：

- Mistral、Claude、OpenAI 的权重非常稳定，是主要 donor judges。
- Google 权重较小但稳定为正，100 次 bootstrap 中 zero weight rate 为 0。
- DeepSeek 权重低且不稳定，25% bootstrap replicate 中权重为 0。
- 这支持“最终组合没有退化成单 judge，但主要依赖三个强 judge”的结论。

## 14. Judge Correlation

donor judge score 之间的 Pearson correlation：

|  | OpenAI | Claude | Gemini Flash | DeepSeek | Mistral |
|---|---:|---:|---:|---:|---:|
| OpenAI | 1.000 | 0.857 | 0.751 | 0.658 | 0.788 |
| Claude | 0.857 | 1.000 | 0.717 | 0.629 | 0.821 |
| Gemini Flash | 0.751 | 0.717 | 1.000 | 0.538 | 0.681 |
| DeepSeek | 0.658 | 0.629 | 0.538 | 1.000 | 0.660 |
| Mistral | 0.788 | 0.821 | 0.681 | 0.660 | 1.000 |

观察：

- OpenAI 与 Claude 高度相关，相关系数 0.857。
- Claude 与 Mistral 也高度相关，相关系数 0.821。
- Gemini Flash 与 DeepSeek 的相关性最低，为 0.538，说明它们的评分模式与主流 judge 有差异。
- 低相关不一定代表高权重。DeepSeek 虽然较独立，但与 human score 的校准贡献有限，因此最终权重低。

## 15. 最好的结果如何理解

从“可作为 calibrated reward 的主模型”角度，最佳结果是 constrained simplex synthetic judge：

```text
RMSE: 0.178066
MAE: 0.141085
mean bias: 0.004257
Pearson: 0.689677
Spearman: 0.663158
effective_num_judges: 3.498
```

它优于：

| 对比对象 | RMSE | 相比 synthetic judge |
|---|---:|---:|
| synthetic judge | 0.178066 | baseline |
| unweighted mean | 0.191250 | synthetic judge 更低约 6.9% |
| best single judge | 0.195830 | synthetic judge 更低约 9.1% |
| median | 0.211434 | synthetic judge 更低约 15.8% |

从“纯预测上界”角度，ridge stacking 的 RMSE 为 0.161834，数值最好。但它是无约束模型，可以使用负权重和自由截距。当前 ridge stacking 系数为：

| judge | coefficient |
|---|---:|
| OpenAI | 0.207525 |
| Claude | 0.194392 |
| Gemini Flash | 0.082569 |
| DeepSeek | -0.119717 |
| Mistral | 0.344553 |
| intercept | 0.165567 |

由于包含负系数，不满足本阶段 synthetic-control-inspired convex combination 的约束，因此不能作为最终 reward source。它的作用是提示：如果未来允许更复杂的 calibration model，例如 affine calibration、isotonic calibration、nonnegative but non-simplex regression，可能还能进一步降低误差。

## 16. Calibrated Reward 写回结果

本阶段已经生成 answer-level synthetic judge predictions：

```text
data/judge_calibration/synthetic_judge_predictions_real_5judge_gemini25flash_think1024_full.jsonl
```

行数为 1570。

同时生成了 paired calibrated JSONL 文件，保留原始 `y_score` 和 `y_agent_score`，新增：

```text
y_score_calibrated
y_agent_score_calibrated
y_score_calibration_model_id
y_agent_score_calibration_model_id
calibration_metadata
```

paired 输出目录：

```text
data/judge_calibration/paired_calibrated_real_5judge_gemini25flash_think1024_unique/
```

输出文件：

| file | rows |
|---|---:|
| `ayers_askdocs__phase3_chatgpt_expert_scored_calibrated.jsonl` | 195 |
| `ayers_askdocs_quality__phase3_chatgpt_expert_scored_calibrated.jsonl` | 195 |
| `ayers_askdocs_empathy__phase3_chatgpt_expert_scored_calibrated.jsonl` | 195 |
| `phase3_gpt4_expert_scored_calibrated.jsonl` | 100 |
| `phase3_llama3_expert_scored_calibrated.jsonl` | 100 |
| `phase3_gemini_expert_scored_calibrated.jsonl` | 100 |

注意：最初 apply 脚本只用 input stem 生成输出名，三个 Ayers 文件同名时会发生覆盖。已修复为同名 stem 时自动加父目录前缀，因此现在 6 个输出文件都能正确保留。

## 17. 可复现实验命令

### 17.1 构建 canonical calibration dataset

```bash
PYTHONPATH=src python scripts/build_judge_calibration_dataset.py \
  --repo-root . \
  --out data/judge_calibration/canonical_examples.jsonl \
  --summary-out outputs/judge_calibration/data_summary.json
```

### 17.2 构建 5-judge full score matrix

```bash
PYTHONPATH=src python scripts/build_judge_score_matrix.py \
  --examples data/judge_calibration/canonical_examples.jsonl \
  --judge-scores data/judge_calibration/judge_scores/openai_gpt55_full.jsonl \
  --judge-scores data/judge_calibration/judge_scores/anthropic_sonnet46_full.jsonl \
  --judge-scores data/judge_calibration/judge_scores/gemini25flash_think1024_full.jsonl \
  --judge-scores data/judge_calibration/judge_scores/deepseek_v4flash_full.jsonl \
  --judge-scores data/judge_calibration/judge_scores/mistral_large_full.jsonl \
  --out-csv data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --missingness-out outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_missingness_report.json \
  --required-judges openai_gpt55,anthropic_sonnet46,google_gemini25flash_think1024,deepseek_v4flash,mistral_large
```

### 17.3 主 grouped-random 实验

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --out-dir outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_grouped_random_balanced_dataset_dimension \
  --split-protocol grouped_random \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

### 17.4 LODO

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --out-dir outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_lodo_balanced_dataset_dimension \
  --split-protocol leave_one_dataset_out \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

### 17.5 Leave-one-score-dimension-out

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --out-dir outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_losdo_balanced_dataset_dimension \
  --split-protocol leave_one_score_dimension_out \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

### 17.6 Leave-one-answer-source-out

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --out-dir outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_loaso_balanced_dataset_dimension \
  --split-protocol leave_one_answer_source_out \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

### 17.7 Bootstrap weight stability

```bash
PYTHONPATH=src python scripts/run_judge_weight_bootstrap.py \
  --score-matrix data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --out-dir outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_bootstrap_weights \
  --n-bootstrap 100 \
  --seed 0 \
  --group-balance \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

### 17.8 写出 calibrated predictions

```bash
PYTHONPATH=src python scripts/apply_synthetic_judge_scores.py \
  --canonical-examples data/judge_calibration/canonical_examples.jsonl \
  --score-matrix data/judge_calibration/score_matrix_real_5judge_gemini25flash_think1024_full.csv \
  --weights-json outputs/judge_calibration/real_5judge_gemini25flash_think1024_full_grouped_random_balanced_dataset_dimension/weights.json \
  --out-predictions data/judge_calibration/synthetic_judge_predictions_real_5judge_gemini25flash_think1024_full.jsonl
```

## 18. 代码与测试状态

新增或更新的核心模块：

| 文件 | 作用 |
|---|---|
| `src/cce_data/judge_calibration/schema.py` | calibration schema 与 JSONL I/O |
| `src/cce_data/judge_calibration/canonicalize.py` | Ayers 与 CounselBench canonicalization |
| `src/cce_data/judge_calibration/score_matrix.py` | donor judge score matrix 构建 |
| `src/cce_data/judge_calibration/weighted_ensemble.py` | simplex weighted judge 与 baselines |
| `src/cce_data/judge_calibration/splits.py` | grouped random、LODO、leave-one-column split |
| `src/cce_data/judge_calibration/metrics.py` | RMSE、MAE、bias、correlation、calibration metrics |
| `src/cce_data/judge_calibration/runner.py` | 实验 runner |
| `src/cce_data/judge_calibration/stability.py` | bootstrap weight stability |
| `src/cce_data/judge_calibration/apply.py` | calibrated score 写回 |
| `scripts/run_donor_judge_scoring.py` | donor judge API scoring |
| `scripts/run_weighted_judge_combination.py` | 主实验命令 |
| `scripts/run_judge_weight_bootstrap.py` | bootstrap 稳定性命令 |
| `scripts/apply_synthetic_judge_scores.py` | 写出 calibrated predictions |

最终测试：

```text
89 passed in 4.86s
```

## 19. 结论

本次 judge weighting 实验证明，在当前 Ayers AskDocs 与 CounselBench 的 human/expert-scored calibration set 上，constrained simplex weighted judge 能稳定优于简单平均、中位数和 best single judge。

最重要的结论是：

1. **weighted judge combination 有实际收益**：主模型 RMSE 为 0.1781，优于 unweighted mean 的 0.1912 和 best single judge 的 0.1958。
2. **权重没有退化成单一 judge**：最大权重约 0.341，effective number of judges 为 3.50。
3. **主要贡献来自 Mistral、Claude、OpenAI**：三者在 bootstrap 中权重稳定。
4. **Google Gemini Flash 有小但稳定的正贡献**：权重约 0.07，bootstrap zero weight rate 为 0。
5. **DeepSeek 贡献较弱且不稳定**：主权重约 0.017，bootstrap 中 25% 为 0。
6. **跨数据集与跨维度仍有明显 domain/rubric shift**：Ayers quality 维度和部分 answer source 是主要误差来源。
7. **calibrated reward 已可写回 OPE 输入文件**：本阶段已经生成 answer-level 和 paired-level calibrated score 文件，但后续接入 OPE 时仍需单独评估 support/positivity 问题。

因此，本阶段完成了 judge calibration layer 的第一版闭环：从 human/expert-scored 数据整理、真实 5-judge scoring、score matrix 构建、weighted synthetic judge 训练、完整验证协议、稳定性诊断，到 calibrated score 写回，均已跑通并通过测试。
