# Ayers AskDocs All Reward Definitions Phase 5 Comparison

One-page comparison across composite, quality-only, and empathy-only rewards.

Ground truth by reward:

| Reward | V_b | V_agent | Effect |
|---|---:|---:|---:|
| Composite | 0.4254 | 0.7233 | +0.2979 |
| Quality-only | 0.5641 | 0.7829 | +0.2188 |
| Empathy-only | 0.2868 | 0.6637 | +0.3769 |

Full estimator table:

| Reward | Config | Estimator | V_b | V_agent | Effect | V_hat | Bias | RMSE | 95% CI | Cov |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| Composite | BGE-M3 | DM | 0.4254 | 0.7233 | +0.2979 | 0.5537 | -0.1696 | 0.1832 | [0.502, 0.593] | no |
| Composite | BGE-M3 | MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6193 | -0.1040 | 0.1134 | [0.555, 0.696] | no |
| Composite | BGE-M3 | OffCEM | 0.4254 | 0.7233 | +0.2979 | 0.5551 | -0.1682 | 0.1826 | [0.503, 0.593] | no |
| Composite | BGE-M3 + concat-PCA | DM | 0.4254 | 0.7233 | +0.2979 | 0.5593 | -0.1640 | 0.1611 | [0.531, 0.597] | no |
| Composite | BGE-M3 + concat-PCA | MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6464 | -0.0769 | 0.0857 | [0.588, 0.716] | no |
| Composite | BGE-M3 + concat-PCA | OffCEM | 0.4254 | 0.7233 | +0.2979 | 0.5607 | -0.1626 | 0.1603 | [0.532, 0.597] | no |
| Quality-only | BGE-M3 | DM | 0.5641 | 0.7829 | +0.2188 | 0.6914 | -0.0915 | 0.1102 | [0.635, 0.720] | no |
| Quality-only | BGE-M3 | MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7304 | -0.0526 | 0.0612 | [0.679, 0.776] | no |
| Quality-only | BGE-M3 | OffCEM | 0.5641 | 0.7829 | +0.2188 | 0.6926 | -0.0903 | 0.1098 | [0.635, 0.720] | no |
| Quality-only | BGE-M3 + concat-PCA | DM | 0.5641 | 0.7829 | +0.2188 | 0.6983 | -0.0846 | 0.0857 | [0.668, 0.727] | no |
| Quality-only | BGE-M3 + concat-PCA | MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7587 | -0.0242 | 0.0363 | [0.706, 0.801] | yes |
| Quality-only | BGE-M3 + concat-PCA | OffCEM | 0.5641 | 0.7829 | +0.2188 | 0.6998 | -0.0831 | 0.0848 | [0.669, 0.727] | no |
| Empathy-only | BGE-M3 | DM | 0.2868 | 0.6637 | +0.3769 | 0.3928 | -0.2709 | 0.2655 | [0.358, 0.459] | no |
| Empathy-only | BGE-M3 | MIPS | 0.2868 | 0.6637 | +0.3769 | 0.5082 | -0.1555 | 0.1674 | [0.420, 0.619] | no |
| Empathy-only | BGE-M3 | OffCEM | 0.2868 | 0.6637 | +0.3769 | 0.3964 | -0.2673 | 0.2647 | [0.359, 0.460] | no |
| Empathy-only | BGE-M3 + concat-PCA | DM | 0.2868 | 0.6637 | +0.3769 | 0.4423 | -0.2214 | 0.2159 | [0.402, 0.502] | no |
| Empathy-only | BGE-M3 + concat-PCA | MIPS | 0.2868 | 0.6637 | +0.3769 | 0.5423 | -0.1214 | 0.1303 | [0.466, 0.652] | no |
| Empathy-only | BGE-M3 + concat-PCA | OffCEM | 0.2868 | 0.6637 | +0.3769 | 0.4429 | -0.2208 | 0.2153 | [0.404, 0.502] | no |

MIPS summary:

| Reward | Config | V_hat | Bias | RMSE | Cov |
|---|---|---:|---:|---:|---|
| Composite | BGE-M3 | 0.6193 | -0.1040 | 0.1134 | no |
| Composite | BGE-M3 + learned concat-PCA | 0.6464 | -0.0769 | 0.0857 | no |
| Quality-only | BGE-M3 | 0.7304 | -0.0526 | 0.0612 | no |
| Quality-only | BGE-M3 + learned concat-PCA | 0.7587 | -0.0242 | 0.0363 | yes |
| Empathy-only | BGE-M3 | 0.5082 | -0.1555 | 0.1674 | no |
| Empathy-only | BGE-M3 + learned concat-PCA | 0.5423 | -0.1214 | 0.1303 | no |

Takeaways:

- MIPS is the best estimator under every reward definition.
- Learned concat-PCA improves MIPS RMSE under composite, quality-only, and empathy-only.
- Only quality-only + learned concat-PCA + MIPS covers `V_agent` with the plain bootstrap CI.
