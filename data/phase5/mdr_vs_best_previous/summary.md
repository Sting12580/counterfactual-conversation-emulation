# MDR vs Previous Best Estimator

Comparison scope: five datasets x two embeddings. The previous-best estimator is selected
from DM/MIPS/OffCEM by smallest absolute point-estimate bias in the same cell.

- Wall time for matched previous-estimator point rerun: 246.4s
- Previous best wins: 10 / 10
- MDR wins: 0 / 10

| Dataset | Embedding | Prev best | Prev abs bias | MDR abs bias | Delta MDR-prev | Winner |
|---|---|---|---:|---:|---:|---|
| ACI-Bench | BGE-M3 | MIPS | 0.0892 | 0.1144 | +0.0252 | Previous best |
| ACI-Bench | BGE-M3 + learned concat-PCA | MIPS | 0.0845 | 0.0996 | +0.0151 | Previous best |
| Ayers AskDocs | BGE-M3 | MIPS | 0.1040 | 0.1690 | +0.0650 | Previous best |
| Ayers AskDocs | BGE-M3 + learned concat-PCA | MIPS | 0.0769 | 0.1633 | +0.0864 | Previous best |
| CounselBench-Eval pooled | BGE-M3 | MIPS | 0.1359 | 0.1712 | +0.0353 | Previous best |
| CounselBench-Eval pooled | BGE-M3 + learned concat-PCA | MIPS | 0.0892 | 0.1653 | +0.0761 | Previous best |
| MTS-Dialog | BGE-M3 | MIPS | 0.1402 | 0.2547 | +0.1145 | Previous best |
| MTS-Dialog | BGE-M3 + learned concat-PCA | MIPS | 0.1338 | 0.2699 | +0.1361 | Previous best |
| PriMock57 | BGE-M3 | MIPS | 0.3004 | 0.3090 | +0.0086 | Previous best |
| PriMock57 | BGE-M3 + learned concat-PCA | DM | 0.2570 | 0.2570 | +0.0000 | Previous best |

Detailed point estimates:

| Dataset | Embedding | V_agent | Prev V_hat | Prev rel bias | MDR V_hat | MDR rel bias |
|---|---|---:|---:|---:|---:|---:|
| ACI-Bench | BGE-M3 | 0.8727 | 0.7835 | -10.22% | 0.7583 | -13.11% |
| ACI-Bench | BGE-M3 + learned concat-PCA | 0.8727 | 0.7882 | -9.68% | 0.7731 | -11.41% |
| Ayers AskDocs | BGE-M3 | 0.7233 | 0.6193 | -14.38% | 0.5543 | -23.37% |
| Ayers AskDocs | BGE-M3 + learned concat-PCA | 0.7233 | 0.6464 | -10.63% | 0.5600 | -22.58% |
| CounselBench-Eval pooled | BGE-M3 | 0.7208 | 0.5849 | -18.85% | 0.5496 | -23.75% |
| CounselBench-Eval pooled | BGE-M3 + learned concat-PCA | 0.7208 | 0.6317 | -12.37% | 0.5556 | -22.93% |
| MTS-Dialog | BGE-M3 | 0.7874 | 0.6472 | -17.80% | 0.5327 | -32.35% |
| MTS-Dialog | BGE-M3 + learned concat-PCA | 0.7874 | 0.6535 | -17.00% | 0.5175 | -34.28% |
| PriMock57 | BGE-M3 | 0.8656 | 0.5652 | -34.71% | 0.5566 | -35.70% |
| PriMock57 | BGE-M3 + learned concat-PCA | 0.8656 | 0.6086 | -29.69% | 0.6086 | -29.69% |
