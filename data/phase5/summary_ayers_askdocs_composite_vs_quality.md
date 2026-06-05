# Ayers AskDocs Composite vs Quality-Only Phase 5 Comparison

Input datasets:

- Composite: `data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl`
- Quality-only: `data/ayers_askdocs_quality/phase3_chatgpt_expert_scored.jsonl`

Main table:

| Reward | Config | Estimator | V_b | V_agent | Effect | V_hat | Bias | RMSE | 95% CI | Cov |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Composite | BGE-M3 | DM | 0.4254 | 0.7233 | +0.2979 | 0.5537 | -0.1696 | 0.1832 | [0.502, 0.593] | no |
| Composite | BGE-M3 | MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6193 | -0.1040 | 0.1134 | [0.555, 0.696] | no |
| Composite | BGE-M3 | OffCEM | 0.4254 | 0.7233 | +0.2979 | 0.5551 | -0.1682 | 0.1826 | [0.503, 0.593] | no |
| Composite | BGE-M3 + learned concat-PCA | DM | 0.4254 | 0.7233 | +0.2979 | 0.5593 | -0.1640 | 0.1611 | [0.531, 0.597] | no |
| Composite | BGE-M3 + learned concat-PCA | MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6464 | -0.0769 | 0.0857 | [0.588, 0.716] | no |
| Composite | BGE-M3 + learned concat-PCA | OffCEM | 0.4254 | 0.7233 | +0.2979 | 0.5607 | -0.1626 | 0.1603 | [0.532, 0.597] | no |
| Quality-only | BGE-M3 | DM | 0.5641 | 0.7829 | +0.2188 | 0.6914 | -0.0915 | 0.1102 | [0.635, 0.720] | no |
| Quality-only | BGE-M3 | MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7304 | -0.0526 | 0.0612 | [0.679, 0.776] | no |
| Quality-only | BGE-M3 | OffCEM | 0.5641 | 0.7829 | +0.2188 | 0.6926 | -0.0903 | 0.1098 | [0.635, 0.720] | no |
| Quality-only | BGE-M3 + learned concat-PCA | DM | 0.5641 | 0.7829 | +0.2188 | 0.6983 | -0.0846 | 0.0857 | [0.668, 0.727] | no |
| Quality-only | BGE-M3 + learned concat-PCA | MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7587 | -0.0242 | 0.0363 | [0.706, 0.801] | yes |
| Quality-only | BGE-M3 + learned concat-PCA | OffCEM | 0.5641 | 0.7829 | +0.2188 | 0.6998 | -0.0831 | 0.0848 | [0.669, 0.727] | no |

MIPS direct comparison:

| Reward | Config | V_agent | MIPS V_hat | Bias | RMSE | Cov |
|---|---|---:|---:|---:|---:|---|
| Composite | BGE-M3 | 0.7233 | 0.6193 | -0.1040 | 0.1134 | no |
| Composite | BGE-M3 + learned concat-PCA | 0.7233 | 0.6464 | -0.0769 | 0.0857 | no |
| Quality-only | BGE-M3 | 0.7829 | 0.7304 | -0.0526 | 0.0612 | no |
| Quality-only | BGE-M3 + learned concat-PCA | 0.7829 | 0.7587 | -0.0242 | 0.0363 | yes |

Takeaway:

- Quality-only reward raises both `V_b` and `V_agent`, but the true effect is smaller than composite.
- MIPS is best in both reward definitions.
- Learned concat-PCA improves MIPS in both cases; under quality-only it also covers `V_agent`.
