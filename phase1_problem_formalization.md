# Phase 1 Problem Formalization

## 1. Core Question

This project asks the following causal policy-evaluation question:

> For historical clinician-patient encounters where a clinician wrote a final assessment and plan, what would the expected quality of care have been if a fixed medical AI agent had generated the final diagnosis and management plan for the same patient context?

The phase-1 decision is to formalize this as **agent replacement of the final clinical decision**, not as a full multi-turn dialogue policy. Each encounter is treated as one contextual-bandit decision:

1. Observe patient context `X` at decision time.
2. Choose a free-form natural-language clinical action `A`.
3. Observe or assign a scalar quality reward `Y in [0, 1]`.

This keeps the target trial emulation precise while making off-policy evaluation feasible for the first version of the project.

## 2. Literature-Derived Decisions

| Literature thread | What it contributes to the formalization |
| --- | --- |
| Target trial emulation | Use a target-trial protocol to make the causal question explicit: eligibility, treatment strategies, assignment/time zero, follow-up, outcome, causal contrast, and analysis plan. |
| Contextual-bandit OPE and doubly robust policy evaluation | Treat each encounter as partial-feedback data: we observe the reward only for the clinician action, while the target agent action is counterfactual. |
| MIPS for large action spaces | Replace exact language-action support with embedding-space support, because exact IPS over free-form text is not usable. |
| OffCEM | Treat no-direct-effect through embeddings as fragile and introduce a cluster/residual reward decomposition as the more defensible estimator-level formalization. |
| Dialog OPE | Avoid full multi-turn MDP OPE in v1 because historical clinician conversations do not provide reliable coverage for target agent trajectories. |
| Conversational diagnostic AI evaluation | Define the action as a clinically meaningful final output, and define the reward over diagnostic quality, management reasoning, communication, and safety dimensions. |

## 3. Target Trial Emulation Protocol

| Target-trial component | Specification for this project |
| --- | --- |
| Eligibility criteria | Historical encounters with enough pre-decision patient information to define `X`, and with a clinician final note from which final diagnosis/differential and management plan can be extracted. |
| Target population | Encounters on the natural support of available clinician-written notes. This avoids claiming validity for cases unlike the logged data. |
| Time zero | The moment after patient presentation/history is available but before using the clinician final assessment, diagnosis, or management plan. |
| Treatment strategies | `pi_b`: clinician behavior policy that produced the logged final output. `pi_agent`: fixed medical agent policy that receives the same `X` and produces a final diagnosis plus management plan. |
| Assignment | In the target trial, eligible encounters would be randomized to clinician final output vs. agent final output. In the observational emulation, only clinician actions are logged, so OPE is required to estimate the agent arm. |
| Follow-up | Immediate evaluation of the final clinical output under a rubric. This is a surrogate endpoint, not real downstream patient outcome follow-up. |
| Outcome | `Y in [0, 1]`, a continuous rubric-based quality score for `(X, A)`, ideally from an LLM judge calibrated against human review. |
| Causal contrast | Average deployment effect `Delta = V(pi_agent) - V(pi_b)` over the eligible historical encounter population. |
| Analysis plan | Estimate `V(pi_agent)` from logged clinician data using OPE adapted to free-form language actions, with embedding or cluster support diagnostics. |

## 4. Observed Data and Notation

The logged dataset is

```text
D = {(X_i, A_i^b, E_i, Y_i)}_{i=1}^n.
```

| Symbol | Meaning | Operational definition |
| --- | --- | --- |
| `i` | Encounter index | One historical clinician-patient encounter. |
| `X_i` | Patient context | Information available at time zero: symptoms, history, relevant exam/lab context if available before the final decision. Must exclude clinician assessment/plan and post-decision information. |
| `A_i^b` | Logged behavior action | Clinician final diagnosis, differential, and management plan extracted from the note. |
| `pi_b(A | X)` | Behavior policy | Unknown clinician policy that generated the logged final action. It is not assumed to be known or computable. |
| `pi_agent(A | X)` | Target policy | Fixed agent configuration: model, system prompt, input template, decoding settings, and any tool policy. |
| `phi(A)` | Action representation map | Embedding or clustering function mapping free-form language action into a lower-dimensional representation. |
| `E_i = phi(A_i^b)` | Logged action representation | Embedding or cluster for the clinician action. |
| `Y_i` | Reward | Rubric score for `(X_i, A_i^b)`, normalized to `[0, 1]`. |
| `q(x, a)` | Conditional reward function | `E[Y(a) | X = x]`, the expected rubric score if action `a` were used for context `x`. |

The target agent can also be run on each logged context to produce

```text
A_i^agent ~ pi_agent(. | X_i),    E_i^agent = phi(A_i^agent).
```

These generated actions are used for direct-method and validation workflows, but the core OPE estimand is defined independently of any finite generated sample.

## 5. Primary Estimand

The main estimand is the target policy value:

```text
V(pi_agent)
  = E_X [ E_{A ~ pi_agent(.|X)} [ Y(A) | X ] ]
  = E_X [ E_{A ~ pi_agent(.|X)} [ q(X, A) ] ].
```

The behavior-policy value is

```text
V(pi_b)
  = E_X [ E_{A ~ pi_b(.|X)} [ q(X, A) ] ].
```

The deployment causal contrast is

```text
Delta = V(pi_agent) - V(pi_b).
```

Interpretation: `Delta` is the average change in rubric-defined final clinical-output quality if the fixed agent replaced the clinician final output for encounters drawn from the same eligible historical population.

For the observed behavior policy, a plug-in empirical estimate is

```text
V_hat(pi_b) = (1/n) * sum_i Y_i.
```

Estimating `V(pi_agent)` is the hard part because the logged data contain rewards for clinician actions, not rewards for all possible agent actions.

## 6. Why Standard IPS Is Not the Right Formalization

If actions were small and discrete, standard contextual-bandit IPS would use

```text
V_hat_IPS(pi_agent)
  = (1/n) * sum_i [ pi_agent(A_i^b | X_i) / pi_b(A_i^b | X_i) ] * Y_i.
```

This is not viable here:

1. `A` is free-form natural language, so exact sequence-level overlap between clinician and agent outputs is essentially zero.
2. `pi_b` is a human clinician policy and is not observed as a tractable probability distribution.
3. Exact action-space positivity is too strong: the agent may phrase the same clinical plan differently from any logged clinician note.

Therefore, the OPE problem should be formalized in an **embedding or cluster action space** rather than exact language-action space.

## 7. Embedding-Space OPE Formulation

Let `E = phi(A)` be an embedding or cluster representation of the clinical action. The MIPS-style target value can be rewritten as an embedding-mediated value when the representation is sufficient:

```text
V(pi_agent)
  = E_X [ E_{E ~ p(.|X, pi_agent)} [ E[Y | X, E] ] ].
```

The marginal density ratio is

```text
w(X, E)
  = p(E | X, pi_agent) / p(E | X, pi_b).
```

A MIPS-style estimator has the form

```text
V_hat_MIPS(pi_agent)
  = (1/n) * sum_i w(X_i, E_i) * Y_i.
```

In practice, `w(X, E)` can be estimated with a classifier that distinguishes clinician-action embeddings from agent-action embeddings conditional on `X`. This treats density-ratio estimation as a supervised classification problem rather than requiring explicit probabilities for human text generation.

## 8. OffCEM-Style Refinement

The strongest MIPS assumption is that `A` has no effect on `Y` after conditioning on `X` and `E`. This is fragile in clinical language because two actions can share the same diagnosis/plan semantics but differ in clarity, safety wording, uncertainty communication, or patient-specific caveats.

To make the formalization robust, treat OffCEM as the main conceptual refinement. Let `C = phi(A)` be a cluster or low-dimensional action representation. Decompose the reward as

```text
q(X, A) = g(X, C) + h(X, A),
```

where:

- `g(X, C)` is the cluster or embedding-mediated effect.
- `h(X, A)` is the residual effect of the exact action within the cluster.

An OffCEM-style estimator uses importance weighting only over the cluster space and uses a model-based term for within-cluster residuals:

```text
V_hat_OffCEM(pi_agent)
  = (1/n) * sum_i [
      w(X_i, C_i) * (Y_i - f_hat(X_i, A_i^b))
      + E_{A ~ pi_agent(.|X_i)} f_hat(X_i, A)
    ].
```

where

```text
w(X, C) = p(C | X, pi_agent) / p(C | X, pi_b).
```

This should be presented as the preferred way to relax the MIPS no-direct-effect assumption, not as proof that all residual language effects are solved.

## 9. Identification Assumptions

| Assumption | Role in this project |
| --- | --- |
| Consistency | The observed clinician score is the potential score under the observed clinician action: `Y_i = Y_i(A_i^b)`. |
| Time-zero validity | `X` must contain only pre-decision information. Leakage from the final note or post-decision data invalidates the causal question. |
| Fixed target policy | `pi_agent` must be frozen: model version, prompt, input template, decoding, and any tool behavior are part of the policy. |
| No interference | The action for one encounter does not affect another encounter's potential reward. |
| Conditional exchangeability | Given `X`, the reward function for an action is comparable between the logged clinician setting and the counterfactual agent-action setting. |
| Common embedding or cluster support | Agent-generated embeddings/clusters must lie in regions covered by clinician actions for comparable `X`. |
| MIPS no-direct-effect | For pure MIPS, action effects on reward must be fully mediated by `E = phi(A)`. This is a diagnostic assumption, not the main claim. |
| OffCEM local correctness | The residual model should preserve relative reward differences among actions within each cluster. |
| Rubric validity | The LLM-judge reward is a surrogate measure of output quality, not a validated patient outcome unless separately calibrated. |

## 10. Scope Decision

The project should not start with a full multi-turn MDP formalization. Dialog OPE literature frames conversation as sequential interaction, but that requires state-action coverage of target-policy trajectories. Historical clinician-patient data generally cover human clinician trajectories, not the trajectories induced by a new medical agent. This creates a severe support problem.

The first version should instead define a single decision point: final clinical output after patient context. This still captures the core novelty, because the action remains a free-form natural-language medical decision and the behavior policy remains an unknown human clinician policy.

## 11. Dataset Construction Checks

Before moving to Phase 2, each candidate dataset must pass these checks:

1. `X` can be extracted without using clinician final diagnosis, assessment, or management plan.
2. `A^b` can be extracted as final diagnosis/differential plus management plan.
3. `Y` can be assigned by a stable rubric to both clinician and agent actions.
4. The same `X` can be passed to the fixed agent without requiring hidden information from the final note.
5. Agent-action embeddings or clusters can be compared to clinician-action embeddings or clusters.
6. Support diagnostics can identify cases where `pi_agent` produces actions outside the clinician embedding/cluster support.

## 12. Acceptance Tests for Phase 1

- The estimand reduces to standard contextual-bandit OPE when `A` is finite and discrete.
- Exact action-space IPS is explicitly rejected as infeasible for free-form clinical language.
- The formalization includes both a target-trial protocol table and an OPE notation table.
- The assumptions distinguish causal-identification assumptions from estimator-specific assumptions.
- The limitations explicitly state that rubric reward is a surrogate endpoint, not real patient outcome.

## References

- Saito, Y. and Joachims, T. (2022). Off-Policy Evaluation for Large Action Spaces via Embeddings. ICML. https://proceedings.mlr.press/v162/saito22a.html
- Saito, Y., Ren, Q., and Joachims, T. (2023). Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling. ICML. https://usait0.com/en/publication/2023/icml2023/
- Hernan, M. A. and Robins, J. M. (2016). Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. American Journal of Epidemiology. https://academic.oup.com/aje/article-pdf/183/8/758/6652570/kwv254.pdf
- Cashin, A. G. et al. (2025). Transparent Reporting of Observational Studies Emulating a Target Trial: The TARGET Statement. JAMA. https://jamanetwork.com/journals/jama/fullarticle/2837724
- Dudik, M., Langford, J., and Li, L. (2011). Doubly Robust Policy Evaluation and Learning. ICML. https://arxiv.org/abs/1103.4601
- Jiang, H. et al. (2021). Towards Automatic Evaluation of Dialog Systems: A Model-Free Off-Policy Evaluation Approach. EMNLP. https://aclanthology.org/2021.emnlp-main.589/
- Tu, T. et al. (2024). Towards Conversational Diagnostic AI. https://arxiv.org/abs/2401.05654
