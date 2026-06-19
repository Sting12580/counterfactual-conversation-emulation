# Current Project Overview / 当前项目总览

本文档总结当前的研究设定、已经完成的进展，以及诊断分析。

## 1. Current Setting / 当前设定

### 1.1 Research Question / 研究问题

项目的核心问题是：在历史医生-患者对话中，如果把医生最终写出的诊断和治疗计划替换成一个固定 medical AI agent 的回答，平均质量会变成什么样？


当前版本把每个 encounter 建模为一个 single-step contextual bandit，而不是完整多轮对话 MDP：

```text
Observe patient context X_i
final clinical action A_i
Observe or assign reward Y_i in [0, 1]
```

主要 estimand 是 agent policy value 和相对医生行为策略的 deployment effect：

```text
V(pi_agent) = E_X E_{A ~ pi_agent(. | X)} [q(X, A)]
V(pi_b)     = E_X E_{A ~ pi_b(. | X)}     [q(X, A)]
Delta       = V(pi_agent) - V(pi_b)
```
policy value：先从真实数据分布里抽一个 context X，再让 policy pi 在这个 X 下选 action，最后看这个 action 的期望 reward。

### 1.2 Dataset Construction / 数据集构建

每条样本被整理成 paired evaluation 结构：


```text
{
  X_i: patient context available at time zero,
  A_i^b: clinician final diagnosis / assessment / management plan,
  A_i^agent: target agent answer generated from the same X_i,
  E_i: embedding or learned representation of the action,
  Y_i^b: reward for the clinician action,
  Y_i^agent: reward for the agent action, used for evaluation
}
```

其中 `X_i` 必须只包含 time-zero information，也就是在最终诊断或治疗计划之前可用的信息。`A_i^b` 是从医生笔记中抽取出来的最终 clinical action。`A_i^agent` 是把相同 `X_i` 输入固定 agent policy 后生成的回答。


实验中 `Y_i^agent` 是为了评估 estimator 的 bias 和 RMSE 而保留的 ground-truth target score；estimator 本身主要从 logged clinician tuples 学习和估计。


### 1.3 Dataset Types / 数据集类型

当前数据集分为两类。

**With expert scored / 有专家评分的数据集**

这类数据集本身包含人类专家或专业标注者对回答质量的评分，因此 reward 来自 expert annotation，而不是 LLM judge。当前主要包括：

- Ayers AskDocs: physician response vs ChatGPT response, with healthcare-professional quality and empathy ratings.
- CounselBench-Eval: therapist/human baseline vs logged LLM responders, with mental-health professional annotations.

**Without expert scored / 没有专家评分的数据集**

这类数据集提供 medical dialogue 和 clinician note，但没有现成的人类质量评分。因此 pipeline 会先构建 `X_i` 和 `A_i^b`，再让固定 agent 对同一个 `X_i` 生成 `A_i^agent`，最后用 LLM judge 对医生回答和 agent 回答分别打分。


LLM judge 使用 rubric 维度对 `(X_i, A_i)` 打分，例如诊断/计划质量、reasoning、communication、safety 等，并把结果归一化到 `[0, 1]`。


### 1.4 Estimator Inputs, Outputs, and Goal / 估计器输入、输出和目标

Estimator 的输入包括：

- patient context `X_i`
- logged clinician action `A_i^b`
- generated target-agent action `A_i^agent`
- clinician reward `Y_i^b`
- text embeddings or learned features for context/action pairs

Estimator 的输出是：

```text
V_hat(pi_agent)
```

也就是目标 agent policy 的平均 reward 估计值。然后可以和医生行为策略的 empirical value 对比：


```text
V_hat(pi_b) = (1 / n) sum_i Y_i^b
Delta_hat   = V_hat(pi_agent) - V_hat(pi_b)
```

Estimator 的目标不是直接预测每个样本的 agent score，而是在 partial-feedback / off-policy setting 下尽可能准确地估计整体 policy value。


### 1.5 Evaluation Metrics / 评估指标

因为实验数据中我们也有 `Y_i^agent`，所以可以计算 empirical ground truth：


```text
V_true(pi_agent) = (1 / n) sum_i Y_i^agent
```

Estimator 效果主要用以下指标判断：

- Bias: `V_hat(pi_agent) - V_true(pi_agent)`
- Relative bias: `Bias / V_true(pi_agent)`
- RMSE: bootstrap estimates against `V_true(pi_agent)`
- Direction agreement: whether `V_hat(pi_agent) - V_hat(pi_b)` has the same sign as the true effect
- CI coverage: whether the confidence interval covers `V_true(pi_agent)`
- CI width: whether the interval is informative rather than too wide to use

## 2. Estimators / 估计器

### 2.1 Shared Features and Density Ratio / 共同特征和密度比

当前 real-data runner 会先分别 embed context 和 action：

```text
phi_x              = embed(X_i)
phi_a_clinician    = embed(A_i^b)
phi_a_agent        = embed(A_i^agent)
```

然后构建交互特征：

```text
z_i(a) = [phi_x ; phi_a ; phi_x * phi_a]
```

Density ratio `w_i` 用来衡量在当前上下文 i 下，clinician 实际做出的 action feature `z_i(A_i^b)`在 agent policy 下有多可能出现，相比在 clinician behavior policy 下有多可能出现。

`w_i` 在 embedding feature space 中估计，实现上使用classifier 区分 target-agent features `z_i(A_i^agent)` 和 behavior-clinician features `z_i(A_i^b)`，再把 classifier probability 转换成 target/behavior ratio。

```text
w_i ~= p(z_i(A_i^b) under pi_agent) / p(z_i(A_i^b) under pi_b)
```

### 2.2 Direct Method (DM)

DM 先用 logged clinician tuples 训练 reward model：

```text
q_hat(X_i, A_i^b) -> Y_i^b
```

然后直接在 agent action 上预测 reward：

```text
V_hat_DM = (1 / n) sum_i q_hat(X_i, A_i^agent)
```

优点是稳定、低方差；缺点是如果 `q_hat` extrapolate 到 agent actions 时有系统性偏差，DM 会系统性低估或高估。

### 2.3 Marginalized IPS / Self-Normalized MIPS

MIPS 用 embedding-space density ratio 给 logged clinician reward 加权。当前 headline 实现使用 self-normalized version，也就是 SNIPS：

```text
V_hat_MIPS = sum_i w_i Y_i^b / sum_i w_i
```

其中 `w_i` 衡量 clinician action feature 在 target agent distribution 下相对于 behavior clinician distribution 的密度比。


为了得到 `w_i`，首先需要 训练一个二分类器区分 `D = 1: sample 来自 agent policy` 或者是 `D = 0: sample 来自 clinician / behavior policy`

然后再使用
```text
p_i = P_hat(D = 1 | Z_i^b)
```
P_i 就是对于这个真实 clinician action 的 embedding feature，分类器认为它有多像 agent policy 产生的 action，最后得到 `w_i` importance weight
```text
w_i = p_i / (1 - p_i)
```

MIPS 的优点是可以不训练 reward model；缺点是高度依赖 embedding support 和 density-ratio 估计，weight tail 会导致 CI 很宽。


### 2.4 OffCEM

OffCEM 结合 reward model 和 density-ratio correction。当前 real-data 实现对 residual correction 使用 self-normalization：

```text
residual_i = Y_i^b - q_hat(X_i, A_i^b)

V_hat_OffCEM =
  (1 / n) sum_i q_hat(X_i, A_i^agent)
  + sum_i w_i residual_i / sum_i w_i
```

直觉上，DM term 给出 agent action 的 reward prediction，weighted residual correction 用 clinician logged data 修正 reward model 的偏差。


### 2.5 Marginalized Doubly Robust (MDR)

MDR 是最近加入的探索性 estimator。它使用标准 unnormalized doubly robust correction：


```text
V_hat_MDR =
  (1 / n) sum_i [
    q_hat(X_i, A_i^agent)
    + w_i * (Y_i^b - q_hat(X_i, A_i^b))
  ]
```

它和当前 OffCEM 的关键区别是 residual correction 不做 self-normalization：

```text
OffCEM correction = sum_i w_i residual_i / sum_i w_i
MDR correction    = (1 / n) sum_i w_i residual_i
```

最新五数据集对比显示，MDR 没有超过之前最优的 DM/MIPS/OffCEM cell，因此目前更适合作为 negative/exploratory result 和 DR correction 形式诊断。

## 3. Current Best Embedding Setup / 当前最强嵌入设定

当前最强设定是：

```text
BGE-M3 + learned concat-PCA + MIPS
```

流程如下：

```text
1. Use BGE-M3 to embed X_i, A_i^b, and A_i^agent.
2. Train a reward-informed learned projection on clinician tuples only:
   (phi_x, phi_a_clinician) -> Y_i^b.
3. Use concat-PCA merge:
   z = [PCA(phi_base) ; learned_projection(phi_base)].
4. Build estimator features [z_x ; z_a ; z_x * z_a].
```

这个 learned projection 只使用 clinician reward `Y_i^b` 训练，不使用 agent reward `Y_i^agent`。`Y_i^agent` 只在最终 evaluation 时用于计算 true value、bias 和 RMSE。

这个设定的优势是：PCA block 保留 BGE-M3 的原始语义几何，learned block 增加 reward-informed correction，因此比单纯 frozen embedding 更适合当前 reward-estimation/OPE 任务。

## 4. Progress / 当前进展

### 4.1 Completed Work / 已完成工作

目前已经完成：

Completed work includes:

- Dataset construction pipeline for medical dialogue datasets.
- Expert-scored dataset conversion for Ayers AskDocs and CounselBench-Eval.
- LLM-judge scoring workflow for datasets without expert scores.
- Main estimator experiments for DM, MIPS/SNIPS, OffCEM, and exploratory MDR.
- Embedding experiments over SBERT, OpenAI text embeddings, MedCPT, BGE-M3, MedCPT+BGE, and BGE-M3 learned concat-PCA.

当前主要结论是：

```text
BGE-M3 + learned concat-PCA + MIPS
is the strongest point-estimation setup.
```

同时，plain bootstrap CI 普遍不能覆盖 `V_true(pi_agent)`。


### 4.2 Core Results Under the Best Setup / 最强设定下的核心结果

下表只保留当前最关键的结果，不复制完整 ablation 表。

The table keeps only the key results instead of copying the full ablation
tables.

| Dataset / Judge | n | Best setup | V_b | V_agent | Effect | V_hat | RelBias | RMSE | Dir% | CI coverage |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| CounselBench / GPT-4 responder | 100 | BGE-M3 + learned concat-PCA + MIPS | 0.5191 | 0.6664 | +0.1473 | 0.6008 | -9.84% | 0.0632 | 100% | no |
| CounselBench / LLaMA-3 responder | 100 | BGE-M3 + learned concat-PCA + MIPS | 0.5191 | 0.8580 | +0.3389 | 0.6476 | -24.52% | 0.2091 | 100% | no |
| Ayers AskDocs / composite expert reward | 195 | BGE-M3 + learned concat-PCA + MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6464 | -10.63% | 0.0857 | 100% | no |
| Ayers AskDocs / quality-only expert reward | 195 | BGE-M3 + learned concat-PCA + MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7587 | -3.09% | 0.0363 | 100% | yes |

Interpretation / 解读：

- Learned concat-PCA consistently improves MIPS point accuracy in the strongest result cells.
- MIPS is the best point estimator in most current best-setup comparisons.
- Plain bootstrap CI still often misses truth, so low RMSE does not automatically imply calibrated uncertainty.
- Ayers quality-only is the cleanest current result: low relative bias, low RMSE, correct direction, and plain CI coverage.

## 5. Diagnostic Results / 诊断结果

### 5.1 验证一：support / positivity

诊断分析显示，当前 MIPS/SNIPS 低估的主要原因不是简单的 normalization 问题，而是 support / positivity 问题。

直观地说，只要 target policy 可能产生某类 action，behavior policy 也必须以非零概率产生足够相似的 action。否则 density ratio 会变得极端，甚至不可估。

在有限样本下，这会表现为：

- effective sample size 很低 / low effective sample size
- 权重集中在极少数 behavior samples 上 / weights concentrated on a few behavior samples
- 最近邻 behavior reward 低于 agent reward / nearest-neighbor behavior rewards below agent rewards
- MIPS/SNIPS point estimate 被拉低 / MIPS/SNIPS point estimates pulled downward
- bootstrap CI 只围绕偏低的 estimator 波动，因此 CI 也偏低 / bootstrap CIs fluctuate around a biased-low estimator, so the intervals also sit too low

最关键的 support 诊断是：在 `BGE-M3 + learned concat-PCA` feature space 里，很多 agent responses 的真实专家分数高于相近 clinician responses 的分数。

| Dataset | n | Support severity | required_topk_fraction | knn10_gap_to_agent_mean | frac_agent_reward_above_knn10_max | Interpretation |
|---|---:|---|---:|---:|---:|---|
| Ayers composite | 195 | severe | 0.128 | 0.149 | 0.210 | Agent mean lies near the high end of the behavior reward distribution. |
| Ayers empathy | 195 | severe | 0.128 | 0.211 | 0.200 | Empathy reward is especially out-of-support. |
| CounselBench LLaMA-3 | 100 | severe | 0.030 | 0.285 | 0.740 | Strongest evidence: 74% of agent responses score above their 10 nearest behavior neighbors' maximum. |

`required_topk_fraction = 0.128` 表示：为了让 clinician reward 的平均值达到 agent mean，只能集中在 clinician reward 分布的前约 12.8%。这说明 agent 的真实均值处在 behavior reward distribution 的高端区域。

`knn10_gap_to_agent_mean = 0.149`  表示：在 BGE-M3 + learned concat-PCA feature space 里，每个 agent response 找 10 个最近的 clinician responses，这些 clinician neighbors 的平均 reward 仍然比 agent mean 低约 0.149


CounselBench LLaMA-3 是最强证据：`required_topk_fraction = 0.030`，并且 `frac_agent_reward_above_knn10_max = 0.740`。也就是说，74% 的 agent responses 的专家分数甚至高于其 10 个最近 clinician neighbors 的最高分。

总体诊断结论是：

```text
The bottleneck is not just estimator choice.
The bottleneck is target-action support in the logged behavior data.
```

当前最好的 `BGE-M3 + learned concat-PCA + MIPS` setup 已经是最强 point estimator，但它仍然会低估 `V(pi_agent)`，因为 agent responses 经常落在 behavior data 的高分稀疏区域或 out-of-support 区域。

### 5.2 验证二：reward-relevant features

诊断中加入了 20 个 regex-based surface features，例如长度、标点、段落、列表、第一/第二人称、empathy phrases、validation phrases、specificity markers、safety-referral phrases 和 hedging phrases。

结果显示：surface features 让 classifier 更容易区分 agent vs behavior，但这种区分不一定等价于 reward-relevant overlap。

例如 density-ratio classifier 可能学到：


```text
agent responses are longer,
agent responses use more commas,
agent responses have more complex formatting.
```

但如果 behavior data 里相同长度/格式的 clinician responses 很少，overlap 会进一步变差，MIPS 权重会更不稳定或更偏。

在 Ayers 上，surface features 对 MIPS 没有帮助，反而增加 bias：

| Dataset | Embedding-only MIPS abs bias | Surface-only MIPS abs bias |
|---|---:|---:|
| Ayers composite | 0.077 | 0.140 |
| Ayers empathy | 0.121 | 0.210 |
| Ayers quality | 0.024 | 0.071 |

但 surface features 对 DM 有帮助，说明它们更适合作为 reward model 的 signal，而不是 density-ratio overlap 的直接修复。


| Dataset | Embedding-only DM `V_hat` | Surface-only DM `V_hat` | True agent mean |
|---|---:|---:|---:|
| Ayers composite | 0.559 | 0.631 | 0.723 |
| Ayers empathy | 0.442 | 0.529 | 0.664 |

因此，surface features 更像 reward-relevant features：它们对 `q_hat` / DM 有帮助，但不适合作为 density-ratio overlap 的直接修复。

### 5.3 验证三：density ratio 估计和 self-normalization

Ayers 的 MIPS/SNIPS 权重高度集中。以 Ayers composite 为例：

```text
n = 195
ess_fraction = 0.062
effective samples ~= 195 * 0.062 = 12
top10_weight_mass = 0.875
```

这表示最高权重的 10% 样本吃掉了约 87.5% 的总权重，MIPS 估计高度依赖少量 clinician rows。

类似现象也出现在 Ayers empathy 和 Ayers quality：

| Dataset | n | ess_fraction | Approx effective samples | top10_weight_mass | Baseline SNIPS `V_hat` | Abs bias |
|---|---:|---:|---:|---:|---:|---:|
| Ayers composite | 195 | 0.062 | 12.2 | 0.875 | 0.646 | 0.077 |
| Ayers empathy | 195 | 0.064 | 12.5 | 0.868 | 0.542 | 0.121 |
| Ayers quality | 195 | 0.069 | 13.5 | 0.838 | 0.759 | 0.024 |

这种 estimator 会有两个问题：一是 variance 大；二是更关键的，如果这些少量 high-weight clinician rows 的 reward 仍不足以代表 agent 高分区域，point estimate 会继续偏低。

理论上，ideal density ratio 应该满足：

```text
E_behavior[w] = 1
```

但实际估计出来的 mean weight 经常小于 1，所以 unnormalized MIPS 会被整体 scale 压低。SNIPS 通过除以 `sum_i w_i` 把 scale 拉回来，因此反而显著提高估计值。

| Dataset | Unnormalized MIPS | SNIPS | True agent mean |
|---|---:|---:|---:|
| Ayers composite | 0.301 | 0.646 | 0.723 |
| Ayers empathy | 0.252 | 0.542 | 0.664 |
| Ayers quality | 0.366 | 0.759 | 0.783 |

因此，SNIPS 本身不是低估的主要来源。它修正了 scale，但无法解决 target action 高分区域缺少相似 behavior support 的问题。

因此，下一步优化 estimator 时，重点应该放在：

Therefore, the next estimator-improvement work should focus on:

- diagnosing high-reward target regions before changing estimator defaults
- avoiding high-variance clipping fixes as the main solution
- improving reward models with useful surface / semantic features
- considering restricted-scope evaluation when support is weak
- collecting or constructing more behavior-like high-quality examples when possible
- treating MDR and other DR variants as diagnostics for correction design rather than immediate headline replacements

### 5.4 验证四：策略互换诊断测试

这个小测试把 policy evaluation 的方向反过来，用来检查 estimator 的偏差是否和 behavior / target 的支持集不对称有关。

原始方向：

```text
behavior = doctor
target   = agent
```

反向方向：

```text
behavior = agent
target   = doctor
```

如果原始方向中 estimator 系统性低估 agent value，而反向方向中 estimator 又系统性高估 doctor value，那么这说明偏差不是随机噪声，而是会随着 policy direction 翻转。这进一步支持 support / positivity 是关键问题：两个 policy 的 action distribution 在 embedding space 中并不对称重叠。

| Dataset | 原始方向 bias | 反向方向 bias | 结论 |
|---|---:|---:|---|
| Ayers composite | -0.0769 | +0.2000 | 偏差翻转 |
| Ayers quality | -0.0242 | +0.1249 | 偏差翻转 |
| Ayers empathy | -0.1214 | +0.2860 | 偏差翻转 |
| CounselBench GPT-4 | -0.0654 | +0.2753 | 偏差翻转 |
| CounselBench LLaMA-3 | -0.2108 | +0.3195 | 偏差翻转 |
| CounselBench Gemini | -0.1293 | +0.1790 | 偏差翻转 |

这个结果说明：当 doctor 是 behavior、agent 是 target 时，MIPS/SNIPS 倾向于低估 agent；当方向交换后，MIPS/SNIPS 又倾向于高估 doctor。也就是说，偏差方向和 policy swap 一起翻转，进一步说明当前估计误差主要来自两个 policy action support 的不对称，而不是单纯来自 reward scale、bootstrap variance 或某个固定方向的评分偏差。
