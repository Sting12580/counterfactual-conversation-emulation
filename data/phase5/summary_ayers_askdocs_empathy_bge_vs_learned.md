# Ayers AskDocs Empathy-Only Phase 5 Summary Table

Input: `data/ayers_askdocs_empathy/phase3_chatgpt_expert_scored.jsonl`

Dataset: Ayers et al. 2023 AskDocs physician vs ChatGPT responses, 195 paired rows.

Reward: normalized healthcare-professional `empathy` score only. Quality is excluded from `Y`.

Run configuration:

- Reward source: human healthcare-professional mean, not an LLM judge
- Target policy: logged ChatGPT response
- Behavior policy: logged physician response
- Bootstrap: 100
- Seed: 0

| Source | n | Config | Estimator | V_b | V_agent | Effect | V_hat | RelBias | RMSE | 95% CI | Dir% | Cov |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| ayers_askdocs_2023 | 195 | BGE-M3 | DM | 0.2868 | 0.6637 | +0.3769 | 0.3928 | -40.82% | 0.2655 | [0.358, 0.459] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 | MIPS | 0.2868 | 0.6637 | +0.3769 | 0.5082 | -23.43% | 0.1674 | [0.420, 0.619] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 | OffCEM | 0.2868 | 0.6637 | +0.3769 | 0.3964 | -40.27% | 0.2647 | [0.359, 0.460] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | DM | 0.2868 | 0.6637 | +0.3769 | 0.4423 | -33.36% | 0.2159 | [0.402, 0.502] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | MIPS | 0.2868 | 0.6637 | +0.3769 | 0.5423 | -18.29% | 0.1303 | [0.466, 0.652] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | OffCEM | 0.2868 | 0.6637 | +0.3769 | 0.4429 | -33.26% | 0.2153 | [0.404, 0.502] | 100.00% | no |

Best by RMSE:

- Overall: `BGE-M3 + learned concat-PCA` + `MIPS`, RMSE 0.1303, V_hat 0.5423.
- `BGE-M3`: `MIPS`, RMSE 0.1674, V_hat 0.5082.
- `BGE-M3 + learned concat-PCA`: `MIPS`, RMSE 0.1303, V_hat 0.5423.

Notes:

- Empathy-only ground truth is `V_b=0.2868`, `V_agent=0.6637`, effect `+0.3769`.
- Both configurations recover the positive direction on every bootstrap resample.
- Learned concat-PCA improves MIPS point accuracy: bias changes from -0.1555 to -0.1214 and RMSE from 0.1674 to 0.1303.
- Under empathy-only reward, no plain bootstrap CI covers `V_agent`.

Source result files:

- `BGE-M3`: `data/phase5/headline_ayers_askdocs_empathy_chatgpt_bge_boot100.json`
- `BGE-M3 + learned concat-PCA`: `data/phase5/headline_ayers_askdocs_empathy_chatgpt_bge_pca128_learn64.json`
