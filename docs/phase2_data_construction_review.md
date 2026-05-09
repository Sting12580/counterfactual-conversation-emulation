# Phase 2 Data Construction Review

## Goal

Phase 2 converts public medical conversation datasets into the canonical logged-bandit dataset:

```text
D = {(X_i, A_i^b, E_i, Y_i)}_{i=1}^n
```

where `X_i` is patient context available at time zero, `A_i^b` is the clinician final output, `E_i` is a later embedding or cluster representation, and `Y_i` is a rubric score.

The most important construction rule from Phase 1 is leakage control: `X_i` cannot include the clinician final diagnosis, assessment, or management plan.

## Source Prioritization

| Source | Use | Rationale |
| --- | --- | --- |
| ACI-Bench | Primary | Full doctor-patient conversations paired with clinical notes; best match to final-output replacement. |
| MTS-Dialog | Primary/scale-up | 1.7k short doctor-patient conversations with section headers and section summaries; useful for action-section extraction. |
| PriMock57 | Smoke test | Small, clean mock primary-care consultations with transcripts and clinician notes; ideal for parser tests. |
| MedDialog | Auxiliary only | Large patient-doctor dialogues, but no clinician note; useful for language/domain coverage, not primary causal OPE. |
| NoteChat | Sensitivity only | Synthetic note-conditioned dialogues; useful for robustness checks, but should not define the main estimand. |

## Literature-Derived Construction Decisions

Target trial emulation argues that the causal question should be specified before analysis: eligibility, treatment strategies, time zero, outcome, causal contrast, and analysis plan. This directly motivates the dataset schema and the `time_zero_policy` column.

MIPS and OffCEM show why exact action-space matching is the wrong target for free-form language. Phase 2 therefore preserves raw clinician action text and reserves `e_action_repr` for embeddings/clusters rather than pretending exact text probabilities are available.

Dialog OPE work frames conversation as sequential interaction but also highlights the coverage requirement for target-policy trajectories. Historical clinician conversations do not reliably cover a new agent's trajectory, so Phase 2 uses the final clinical output as the action for the first benchmark.

Medical dialogue-to-note datasets such as ACI-Bench and MTS-Dialog were built for clinical note generation, not causal policy evaluation. The pipeline therefore records extraction method, inclusion status, and exclusion reasons instead of silently treating every row as valid OPE data.

## Dataset-Specific Rules

### ACI-Bench

Source: https://github.com/wyim/aci-bench

ACI-Bench provides full conversations and notes across train/valid/test splits. The pipeline uses the dialogue as `X_i` and extracts `A_i^b` from action-like note sections such as assessment, impression, diagnosis, and plan. If section labels cannot be found, the row is kept with `full_note_fallback` and flagged for review.

Known issue: ACI-Bench notes that some speaker tags are swapped due to ASR artifacts. The pipeline does not auto-correct speakers; downstream analysis should run source-specific QA before final experiments.

### MTS-Dialog

Source: https://github.com/abachaa/MTS-Dialog

MTS-Dialog rows contain a dialogue, a normalized section header, and a section summary. Only action-like headers are included in the primary OPE dataset: assessment, diagnosis, plan, disposition, and emergency-department course. Context/history sections remain in the output as excluded rows when `--included-only` is not set, which makes the filtering auditable.

### PriMock57

Source: https://github.com/babylonhealth/primock57

PriMock57 has TextGrid doctor/patient transcripts and clinician notes. The pipeline merges doctor and patient utterances by timestamp to form `X_i`, then extracts impression/plan-like text from the note for `A_i^b`. This source is small but useful for smoke testing because it has stable file names and human-written notes.

### MedDialog

Sources: https://aclanthology.org/2020.emnlp-main.743/ and Hugging Face mirrors such as `UCSD26/medical_dialog`.

MedDialog is not enabled by default. It is dialogue-only and lacks a clinician final note, so it cannot directly instantiate `A_i^b` as final diagnosis plus management plan. If included with `--include-dialog-only`, the pipeline treats the final doctor turn as an auxiliary action candidate and marks rows as `auxiliary_only`.

## Output Quality Checks

Before using the dataset for estimator development:

1. Check `qa_counts.csv` for included/excluded balance.
2. Sample `phase2_dataset.jsonl` and manually verify `X_i` excludes the final action.
3. Inspect all rows with `full_note_fallback` before including them in primary experiments.
4. Run judge scoring on a small batch and manually review agreement before scoring the full dataset.
5. Build embeddings/clusters only after action extraction quality is acceptable.

## References

- ACI-Bench: https://github.com/wyim/aci-bench
- MTS-Dialog: https://github.com/abachaa/MTS-Dialog
- MedDialog: https://aclanthology.org/2020.emnlp-main.743/
- PriMock57: https://github.com/babylonhealth/primock57
- Saito and Joachims 2022 MIPS: https://proceedings.mlr.press/v162/saito22a.html
- Saito et al. 2023 OffCEM: https://usait0.com/en/publication/2023/icml2023/
- Target trial emulation: https://academic.oup.com/aje/article-pdf/183/8/758/6652570/kwv254.pdf
- Dialog OPE: https://aclanthology.org/2021.emnlp-main.589/
