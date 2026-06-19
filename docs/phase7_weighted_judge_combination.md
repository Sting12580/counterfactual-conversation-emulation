# Phase 7: Weighted Judge Combination

This phase adds a separate judge calibration layer before downstream CCE/OPE
estimators. Human or expert scores are treated as gold labels. LLM judges are
donor judges. A convex weighted combination of donor judge scores becomes a
synthetic human-aligned judge.

## Data Schema

`CalibrationExample` is answer-level, not paired-record level. It stores the
question/context, one answer, the answer source, the normalized human score in
`[0, 1]`, score dimension, dataset metadata, and a leakage-safe `split_group`.

`JudgeScoreRecord` is long-format donor judge output keyed by
`calib_example_id` and `judge_id`. It stores a normalized score, raw output,
rubric id, prompt hash, model version, and metadata.

`score_matrix.csv` is the wide training matrix. It includes canonical metadata,
`human_score`, and one column per donor judge named `judge::<judge_id>`.

## No-Leakage Rules

Human score fields must never be included in donor judge prompts. Splits are
grouped by `dataset_id::question_id::score_dimension`, so answers for the same
question and scoring dimension do not cross train/validation/test boundaries.
Hyperparameters are selected only on validation data, never test data.

## Commands

Build canonical calibration examples:

```bash
PYTHONPATH=src python scripts/build_judge_calibration_dataset.py \
  --repo-root . \
  --out data/judge_calibration/canonical_examples.jsonl \
  --summary-out outputs/judge_calibration/data_summary.json
```

Build a mock donor score matrix for offline smoke testing:

```bash
PYTHONPATH=src python scripts/build_judge_score_matrix.py \
  --examples data/judge_calibration/canonical_examples.jsonl \
  --out-csv data/judge_calibration/score_matrix_mock.csv \
  --missingness-out outputs/judge_calibration/mock_missingness_report.json \
  --mock-judges judge_a,judge_b,judge_c,judge_d \
  --mock-seed 7 \
  --mock-noise-std 0.08
```

Run grouped random evaluation:

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_mock.csv \
  --out-dir outputs/judge_calibration/mock_grouped_random \
  --split-protocol grouped_random \
  --seed 7 \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

Run leave-one-dataset-out evaluation:

```bash
PYTHONPATH=src python scripts/run_weighted_judge_combination.py \
  --score-matrix data/judge_calibration/score_matrix_mock.csv \
  --out-dir outputs/judge_calibration/mock_lodo \
  --split-protocol leave_one_dataset_out \
  --seed 7 \
  --lambda-grid 0,0.0001,0.001,0.01,0.1,1.0 \
  --fit-intercept-options true,false
```

## Output Interpretation

`weights.json` records learned donor weights, intercept, selected lambda,
maximum weight, Herfindahl index, and effective number of judges. High
`max_weight` or effective judge count below two indicates that the ensemble is
close to a single judge.

`metrics_overall.json` compares the synthetic judge against unweighted mean,
median, best single judge selected on validation, and ridge stacking. Group
metric CSVs report bias, RMSE, and MAE by dataset, score dimension, answer
source, and dataset-score dimension.

Mock-score runs are only pipeline tests. Real claims require precomputed real
donor judge scores or an explicitly executed scoring run with cached prompts and
model versions.

## Limitations And Next Steps

This layer improves reward calibration but does not solve OPE support or
positivity issues. Next work should add real cached donor judge scoring, pairwise
preference support, per-dimension or multi-task calibration, weight stability
bootstrap diagnostics, and downstream Phase 5 reruns with calibrated reward
fields.
