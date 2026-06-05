# Phase 5 Embedding / Estimator Results Summary (Updated)

**Bold rows mark the current best result within each dataset/judge block, using RMSE as the primary criterion and relative bias as a secondary check.**

Key headline: **BGE-M3 + learned concat-PCA + MIPS** is the strongest point-estimation setup. It gives the best mixed/Sonnet result (RMSE 0.0674) and the best CounselBench-GPT4 result (RMSE 0.0632).

## Mixed medical dialogue dataset (ACI 207 + MTS 142 + PriMock 48, n=397) | Judge: GPT-4o | V_b=0.6690, V_agent=0.8585, effect=+0.1895

| Embedding | Estimator | V_hat | RelBias | RMSE | Dir% | 95% CI | Cov | Note |
|---|---:|---:|---:|---:|---:|---|---:|---|
| SBERT | DM | 0.7330 | -14.62% | 0.1357 | 100% | [0.697, 0.751] | no |  |
| SBERT | MIPS | 0.6636 | -22.71% | 0.2142 | 32% | [0.574, 0.719] | no |  |
| SBERT | OffCEM | 0.7256 | -15.48% | 0.1401 | 100% | [0.695, 0.747] | no |  |
| OpenAI text-emb-3-small | DM | 0.6884 | -19.81% | 0.1789 | 96% | [0.660, 0.700] | no |  |
| **OpenAI text-emb-3-small** | **MIPS** | **0.7494** | **-12.71%** | **0.1068** | **100%** | **[0.706, 0.793]** | **no** | **best GPT-4o** |
| OpenAI text-emb-3-small | OffCEM | 0.6921 | -19.39% | 0.1772 | 96% | [0.663, 0.703] | no |  |

## Mixed medical dialogue dataset (n=397) | Judge: Claude Sonnet 4.6 | V_b=0.6178, V_agent=0.8413, effect=+0.2236

| Embedding | Estimator | V_hat | RelBias | RMSE | Dir% | 95% CI | Cov | Note |
|---|---:|---:|---:|---:|---:|---|---:|---|
| OpenAI text-emb-3-small | DM | 0.6364 | -24.35% | 0.2111 | 91% | [0.609, 0.653] | no |  |
| OpenAI text-emb-3-small | MIPS | 0.6850 | -18.58% | 0.1526 | 99% | [0.635, 0.734] | no |  |
| OpenAI text-emb-3-small | OffCEM | 0.6427 | -23.60% | 0.2088 | 92% | [0.611, 0.656] | no |  |
| MedCPT-Article | DM | 0.6951 | -17.38% | 0.1577 | 100% | [0.665, 0.705] | no |  |
| MedCPT-Article | MIPS | 0.7525 | -10.56% | 0.0846 | 100% | [0.726, 0.789] | no |  |
| MedCPT-Article | OffCEM | 0.7000 | -16.79% | 0.1545 | 100% | [0.668, 0.709] | no |  |
| BGE-M3 | DM | 0.7106 | -15.53% | 0.1352 | 100% | [0.673, 0.732] | no |  |
| BGE-M3 | MIPS | 0.7659 | -8.97% | 0.0705 | 100% | [0.750, 0.793] | no |  |
| BGE-M3 | OffCEM | 0.7123 | -15.33% | 0.1341 | 100% | [0.674, 0.734] | no |  |
| MedCPT+BGE concat | DM | 0.7162 | -14.87% | 0.1344 | 100% | [0.679, 0.733] | no |  |
| MedCPT+BGE concat | MIPS | 0.7619 | -9.43% | 0.0719 | 100% | [0.746, 0.795] | no |  |
| MedCPT+BGE concat | OffCEM | 0.7183 | -14.62% | 0.1329 | 100% | [0.680, 0.735] | no |  |
| BGE-M3 + learned concat-PCA | DM | 0.7025 | -16.50% | 0.1373 | 100% | [0.679, 0.730] | no |  |
| **BGE-M3 + learned concat-PCA** | **MIPS** | **0.7711** | **-8.34%** | **0.0674** | **100%** | **[0.751, 0.799]** | **no** | **best mixed/Sonnet** |
| BGE-M3 + learned concat-PCA | OffCEM | 0.7089 | -15.74% | 0.1344 | 100% | [0.683, 0.733] | no |  |

## CounselBench | Judge: GPT-4 responder | V_b=0.5191, V_agent=0.6664, effect=+0.1473

| Embedding | Estimator | V_hat | RelBias | RMSE | Dir% | 95% CI | Cov | Note |
|---|---:|---:|---:|---:|---:|---|---:|---|
| BGE-M3 | DM | 0.5242 | -21.34% | 0.1382 | 71% | [0.488, 0.572] | no |  |
| BGE-M3 | MIPS | 0.5479 | -17.78% | 0.1161 | 92% | [0.514, 0.588] | no |  |
| BGE-M3 | OffCEM | 0.5243 | -21.33% | 0.1382 | 71% | [0.488, 0.572] | no |  |
| BGE-M3 + learned concat-PCA | DM | 0.5382 | -19.24% | 0.1297 | 88% | [0.504, 0.569] | no |  |
| **BGE-M3 + learned concat-PCA** | **MIPS** | **0.6008** | **-9.84%** | **0.0632** | **100%** | **[0.571, 0.639]** | **no** | **global best RMSE** |
| BGE-M3 + learned concat-PCA | OffCEM | 0.5384 | -19.20% | 0.1297 | 88% | [0.504, 0.569] | no |  |

## CounselBench | Judge: LLaMA-3 responder | V_b=0.5191, V_agent=0.8580, effect=+0.3389

| Embedding | Estimator | V_hat | RelBias | RMSE | Dir% | 95% CI | Cov | Note |
|---|---:|---:|---:|---:|---:|---|---:|---|
| BGE-M3 | DM | 0.5978 | -30.33% | 0.2979 | 100% | [0.519, 0.610] | no |  |
| BGE-M3 | MIPS | 0.6287 | -26.72% | 0.2307 | 100% | [0.571, 0.674] | no |  |
| BGE-M3 | OffCEM | 0.5979 | -30.31% | 0.2979 | 100% | [0.519, 0.610] | no |  |
| BGE-M3 + learned concat-PCA | DM | 0.6011 | -29.93% | 0.2614 | 100% | [0.549, 0.637] | no |  |
| **BGE-M3 + learned concat-PCA** | **MIPS** | **0.6476** | **-24.52%** | **0.2091** | **100%** | **[0.607, 0.684]** | **no** | **best in subset** |
| BGE-M3 + learned concat-PCA | OffCEM | 0.6014 | -29.90% | 0.2613 | 100% | [0.549, 0.637] | no |  |

## CounselBench | Judge: Gemini responder | V_b=0.5191, V_agent=0.6381, effect=+0.1191

| Embedding | Estimator | V_hat | RelBias | RMSE | Dir% | 95% CI | Cov | Note |
|---|---:|---:|---:|---:|---:|---|---:|---|
| BGE-M3 | DM | 0.5175 | -18.91% | 0.1197 | 53% | [0.488, 0.550] | no |  |
| BGE-M3 | MIPS | 0.5084 | -20.33% | 0.1357 | 32% | [0.445, 0.558] | no |  |
| BGE-M3 | OffCEM | 0.5176 | -18.89% | 0.1197 | 54% | [0.488, 0.550] | no |  |
| **BGE-M3 + learned concat-PCA** | **DM** | **0.5259** | **-17.60%** | **0.1167** | **54%** | **[0.497, 0.554]** | **no** | **best in subset** |
| BGE-M3 + learned concat-PCA | MIPS | 0.5086 | -20.30% | 0.1342 | 39% | [0.445, 0.566] | no |  |
| **BGE-M3 + learned concat-PCA** | **OffCEM** | **0.5258** | **-17.60%** | **0.1167** | **54%** | **[0.497, 0.554]** | **no** | **best in subset** |

## Notes

- All plain bootstrap CIs still miss V_true(pi_agent), so the main gain here is lower point-estimation error, not calibrated uncertainty.
- CounselBench-GPT4 is the best transferred CounselBench setting; LLaMA-3 and Gemini remain substantially harder.
- DM and OffCEM are nearly identical in most runs, suggesting the residual correction is very small under the current feature/density-ratio setup.