# MDR Five-Dataset Experiment

Estimator: standard marginalized doubly robust (MDR) with unnormalized residual correction.

- Bootstrap iterations: 0
- CounselBench mode: `pooled`
- Embeddings: `BGE-M3`; `BGE-M3 + learned concat-PCA` with PCA=128 and learned dim=64

Notes:

- This is a point-estimate exploratory run. A 100-bootstrap MDR run was attempted first, but the calibrated density-ratio refits were too slow for an interactive pass.
- CounselBench is treated as one pooled dataset by concatenating the GPT-4, LLaMA-3, and Gemini target-responder files, giving n=300.
- Ayers AskDocs uses the composite quality/empathy reward, not the quality-only ablation.

| Dataset | Embedding | n | V_b | V_agent | Effect | MDR V_hat | Bias | RelBias | RMSE | 95% CI | Dir% | Cov |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| ACI-Bench | BGE-M3 | 207 | 0.7423 | 0.8727 | +0.1303 | 0.7583 | -0.1144 | -13.11% |  |  | yes |  |
| ACI-Bench | BGE-M3 + learned concat-PCA | 207 | 0.7423 | 0.8727 | +0.1303 | 0.7731 | -0.0996 | -11.41% |  |  | yes |  |
| MTS-Dialog | BGE-M3 | 142 | 0.4397 | 0.7874 | +0.3477 | 0.5327 | -0.2547 | -32.35% |  |  | yes |  |
| MTS-Dialog | BGE-M3 + learned concat-PCA | 142 | 0.4397 | 0.7874 | +0.3477 | 0.5175 | -0.2699 | -34.28% |  |  | yes |  |
| PriMock57 | BGE-M3 | 48 | 0.6073 | 0.8656 | +0.2583 | 0.5566 | -0.3090 | -35.70% |  |  | no |  |
| PriMock57 | BGE-M3 + learned concat-PCA | 48 | 0.6073 | 0.8656 | +0.2583 | 0.6086 | -0.2570 | -29.69% |  |  | yes |  |
| CounselBench-Eval pooled | BGE-M3 | 300 | 0.5191 | 0.7208 | +0.2018 | 0.5496 | -0.1712 | -23.75% |  |  | yes |  |
| CounselBench-Eval pooled | BGE-M3 + learned concat-PCA | 300 | 0.5191 | 0.7208 | +0.2018 | 0.5556 | -0.1653 | -22.93% |  |  | yes |  |
| Ayers AskDocs | BGE-M3 | 195 | 0.4254 | 0.7233 | +0.2979 | 0.5543 | -0.1690 | -23.37% |  |  | yes |  |
| Ayers AskDocs | BGE-M3 + learned concat-PCA | 195 | 0.4254 | 0.7233 | +0.2979 | 0.5600 | -0.1633 | -22.58% |  |  | yes |  |
