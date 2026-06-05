# Ayers AskDocs Phase 5 Summary Table

Input: `data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl`

Dataset: Ayers et al. 2023 AskDocs physician vs ChatGPT responses, 195 paired rows with healthcare-professional quality/empathy ratings.

Run configuration:

- Reward source: human healthcare-professional mean, not an LLM judge
- Target policy: logged ChatGPT response
- Behavior policy: logged physician response
- Bootstrap: 100
- Seed: 0

| Source | n | Config | Estimator | V_b | V_agent | Effect | V_hat | RelBias | RMSE | 95% CI | Dir% | Cov |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| ayers_askdocs_2023 | 195 | BGE-M3 | DM | 0.4254 | 0.7233 | +0.2979 | 0.5537 | -23.45% | 0.1832 | [0.502, 0.593] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 | MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6193 | -14.38% | 0.1134 | [0.555, 0.696] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 | OffCEM | 0.4254 | 0.7233 | +0.2979 | 0.5551 | -23.25% | 0.1826 | [0.503, 0.593] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | DM | 0.4254 | 0.7233 | +0.2979 | 0.5593 | -22.67% | 0.1611 | [0.531, 0.597] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | MIPS | 0.4254 | 0.7233 | +0.2979 | 0.6464 | -10.63% | 0.0857 | [0.588, 0.716] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | OffCEM | 0.4254 | 0.7233 | +0.2979 | 0.5607 | -22.48% | 0.1603 | [0.532, 0.597] | 100.00% | no |

Best by RMSE:

- Overall: `BGE-M3 + learned concat-PCA` + `MIPS`, RMSE 0.0857, V_hat 0.6464.
- `BGE-M3`: `MIPS`, RMSE 0.1134, V_hat 0.6193.
- `BGE-M3 + learned concat-PCA`: `MIPS`, RMSE 0.0857, V_hat 0.6464.

Notes:

- Both configurations recover the positive direction on every bootstrap resample.
- No plain bootstrap CI covers the human-rated ChatGPT target mean.
- Learned concat-PCA improves MIPS point accuracy on this dataset: bias changes from -0.1040 to -0.0769 and RMSE from 0.1134 to 0.0857.

Source result files:

- `BGE-M3`: `data/phase5/headline_ayers_askdocs_chatgpt_bge_boot100.json`
- `BGE-M3 + learned concat-PCA`: `data/phase5/headline_ayers_askdocs_chatgpt_bge_pca128_learn64.json`
