# Sonnet 4.6 By-Source Phase 5 Results

Input: `data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl`

Split sources:

- `aci_bench`: 207 records
- `mts_dialog`: 142 records
- `primock57`: 48 records

Run configuration:

- Embedder: BGE-M3
- Learned action embedding: concat-PCA
- PCA dim: 128
- Learned dim: 64
- Bootstrap: 100
- Seed: 0
- Score fields: `y_score_claude_sonnet46`, `y_agent_score_claude_sonnet46`

| Source | n | Estimator | V_b | V_agent | Effect | V_hat | RelBias | RMSE | 95% CI | Dir% | Cov |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| aci_bench | 207 | DM | 0.7423 | 0.8727 | +0.1303 | 0.7730 | -11.42% | 0.0974 | [0.761, 0.791] | 100.0% | no |
| aci_bench | 207 | MIPS | 0.7423 | 0.8727 | +0.1303 | 0.7882 | -9.68% | 0.0820 | [0.772, 0.817] | 100.0% | no |
| aci_bench | 207 | OffCEM | 0.7423 | 0.8727 | +0.1303 | 0.7734 | -11.37% | 0.0972 | [0.761, 0.792] | 100.0% | no |
| mts_dialog | 142 | DM | 0.4397 | 0.7874 | +0.3477 | 0.5174 | -34.29% | 0.2704 | [0.485, 0.541] | 100.0% | no |
| mts_dialog | 142 | MIPS | 0.4397 | 0.7874 | +0.3477 | 0.6535 | -17.00% | 0.1496 | [0.556, 0.698] | 100.0% | no |
| mts_dialog | 142 | OffCEM | 0.4397 | 0.7874 | +0.3477 | 0.5180 | -34.21% | 0.2702 | [0.485, 0.541] | 100.0% | no |
| primock57 | 48 | DM | 0.6073 | 0.8656 | +0.2583 | 0.6086 | -29.69% | 0.2709 | [0.550, 0.630] | 19.0% | no |
| primock57 | 48 | MIPS | 0.6073 | 0.8656 | +0.2583 | 0.5532 | -36.09% | 0.3047 | [0.516, 0.614] | 1.0% | no |
| primock57 | 48 | OffCEM | 0.6073 | 0.8656 | +0.2583 | 0.6086 | -29.69% | 0.2709 | [0.550, 0.630] | 19.0% | no |

Best by RMSE:

- `aci_bench`: MIPS, RMSE 0.0820
- `mts_dialog`: MIPS, RMSE 0.1496
- `primock57`: DM / OffCEM tie, RMSE 0.2709

Notes:

- ACI and MTS recover the positive direction on all bootstrap resamples.
- PriMock57 is unstable under this split, likely because n=48 is too small for the learned feature map plus bootstrap refits.
- No plain bootstrap CI covers `V_agent`; this matches the mixed-dataset pattern.
