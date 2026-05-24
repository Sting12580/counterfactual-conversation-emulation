# Phase 6 Learned Action Embedding

This adds a FineTune-style learned action embedding on top of the current
fixed text embedders. The intended headline baseline is BGE-M3, because the
Phase 5 embedding ablation found BGE-M3 to be the strongest single embedder
for MIPS point accuracy.

## Method

The base text embedder is frozen. For each included record, Phase 5 first
computes:

```text
phi_x              = base_embed(x_patient_context)
phi_a_clinician    = base_embed(a_clinician)
phi_a_agent        = base_embed(a_agent)
```

The learned layer then trains a small two-tower reward model using only logged
clinician tuples:

```text
(phi_x, phi_a_clinician) -> y_score
```

It learns:

```text
z_x = context_projector(phi_x)
z_a = action_projector(phi_a)
```

Then the existing estimators reuse the same Phase 5 feature shape:

```text
[z_x ; z_a ; z_x * z_a]
```

Two merge modes are available:

```text
replace:
  z = learned_projection(phi)

concat-pca:
  z = [PCA(phi) ; learned_projection(phi)]
```

`concat-pca` is the safer follow-up when BGE-M3 is already strong: the PCA
block preserves the frozen BGE geometry, while the learned block adds a small
reward-informed correction.

The agent reward `y_agent_score` is not passed to the learned embedding
training step. It is only used afterward to evaluate bias/RMSE against the
ground-truth target value.

## Command

```bash
PYTHONPATH=src python scripts/run_phase5.py \
  --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
  --output data/phase5/headline_sonnet46_bge_learned_action_embedding.json \
  --embedder bge \
  --learned-action-embedding \
  --learned-dim 128 \
  --learned-hidden-dim 128 \
  --learned-epochs 500 \
  --n-boot 100 \
  --seed 0
```

Recommended PCA-preserving follow-up:

```bash
PYTHONPATH=src python scripts/run_phase5.py \
  --input data/phase3/agent_scored_all_judge_claude_sonnet46.renamed.jsonl \
  --output data/phase5/headline_sonnet46_bge_pca128_learn32.json \
  --embedder bge \
  --learned-action-embedding \
  --learned-merge concat-pca \
  --pca-dim 128 \
  --learned-dim 32 \
  --learned-hidden-dim 32 \
  --learned-epochs 100 \
  --learned-weight-decay 1e-2 \
  --n-boot 100 \
  --seed 0
```

For a quick smoke run:

```bash
PYTHONPATH=src python scripts/run_phase5.py \
  --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
  --output data/phase5/headline_sonnet46_bge_learned_smoke.json \
  --embedder bge \
  --learned-action-embedding \
  --learned-dim 64 \
  --learned-epochs 100 \
  --n-boot 0 \
  --seed 0
```

## Caveat

The learned projection is currently fit once on the logged clinician data and
then treated as a fitted feature map during bootstrap. Bootstrap resamples still
refit DM/MIPS/OffCEM, but they do not retrain the learned projection. This keeps
runtime practical for Phase 6 exploration. A stricter, slower follow-up would
retrain the learned projection inside every bootstrap resample.
