# Ayers AskDocs Quality-Only Phase 5 Summary Table

Input: `data/ayers_askdocs_quality/phase3_chatgpt_expert_scored.jsonl`

Dataset: Ayers et al. 2023 AskDocs physician vs ChatGPT responses, 195 paired rows.

Reward: normalized healthcare-professional `quality` score only. Empathy is excluded from `Y`.

Run configuration:

- Reward source: human healthcare-professional mean, not an LLM judge
- Target policy: logged ChatGPT response
- Behavior policy: logged physician response
- Bootstrap: 100
- Seed: 0

| Source | n | Config | Estimator | V_b | V_agent | Effect | V_hat | RelBias | RMSE | 95% CI | Dir% | Cov |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| ayers_askdocs_2023 | 195 | BGE-M3 | DM | 0.5641 | 0.7829 | +0.2188 | 0.6914 | -11.68% | 0.1102 | [0.635, 0.720] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 | MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7304 | -6.71% | 0.0612 | [0.679, 0.776] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 | OffCEM | 0.5641 | 0.7829 | +0.2188 | 0.6926 | -11.53% | 0.1098 | [0.635, 0.720] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | DM | 0.5641 | 0.7829 | +0.2188 | 0.6983 | -10.80% | 0.0857 | [0.668, 0.727] | 100.00% | no |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | MIPS | 0.5641 | 0.7829 | +0.2188 | 0.7587 | -3.09% | 0.0363 | [0.706, 0.801] | 100.00% | yes |
| ayers_askdocs_2023 | 195 | BGE-M3 + learned concat-PCA | OffCEM | 0.5641 | 0.7829 | +0.2188 | 0.6998 | -10.61% | 0.0848 | [0.669, 0.727] | 100.00% | no |

Best by RMSE:

- Overall: `BGE-M3 + learned concat-PCA` + `MIPS`, RMSE 0.0363, V_hat 0.7587.
- `BGE-M3`: `MIPS`, RMSE 0.0612, V_hat 0.7304.
- `BGE-M3 + learned concat-PCA`: `MIPS`, RMSE 0.0363, V_hat 0.7587.

Notes:

- Quality-only ground truth is `V_b=0.5641`, `V_agent=0.7829`, effect `+0.2188`.
- Both configurations recover the positive direction on every bootstrap resample.
- Learned concat-PCA improves MIPS point accuracy: bias changes from -0.0526 to -0.0242 and RMSE from 0.0612 to 0.0363.
- Under quality-only reward, learned concat-PCA MIPS is the only cell whose plain bootstrap CI covers `V_agent`.

Source result files:

- `BGE-M3`: `data/phase5/headline_ayers_askdocs_quality_chatgpt_bge_boot100.json`
- `BGE-M3 + learned concat-PCA`: `data/phase5/headline_ayers_askdocs_quality_chatgpt_bge_pca128_learn64.json`
