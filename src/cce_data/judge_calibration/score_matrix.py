from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from cce_data.judge_calibration.schema import (
    CalibrationExample,
    JudgeScoreRecord,
    calibration_example_from_dict,
    dataclass_to_json_dict,
    judge_score_record_from_dict,
    read_jsonl,
    validate_score_01,
    write_jsonl,
)


METADATA_COLUMNS = [
    "calib_example_id",
    "dataset_id",
    "domain",
    "task_type",
    "score_dimension",
    "question_id",
    "question",
    "context",
    "answer",
    "answer_source",
    "human_score",
    "human_score_raw",
    "human_score_scale",
    "human_score_type",
    "human_rubric",
    "source_file",
    "split_group",
]


def load_calibration_examples(path: str | Path) -> list[CalibrationExample]:
    examples = [calibration_example_from_dict(record) for record in read_jsonl(path)]
    ids = [example.calib_example_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("calib_example_id values must be unique")
    return examples


def load_judge_scores(paths: Iterable[str | Path]) -> list[JudgeScoreRecord]:
    records: list[JudgeScoreRecord] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        for raw in read_jsonl(path):
            record = judge_score_record_from_dict(raw)
            key = (record.calib_example_id, record.judge_id)
            if key in seen:
                raise ValueError(f"Duplicate judge score for {key}")
            seen.add(key)
            records.append(record)
    return records


def _example_to_row(example: CalibrationExample) -> dict[str, Any]:
    data = dataclass_to_json_dict(example)
    row = {column: data.get(column) for column in METADATA_COLUMNS}
    row["metadata"] = json.dumps(data.get("metadata", {}), sort_keys=True, ensure_ascii=False)
    for nested in ["human_score_raw", "human_score_scale"]:
        if isinstance(row.get(nested), (dict, list)):
            row[nested] = json.dumps(row[nested], sort_keys=True, ensure_ascii=False)
    return row


def build_score_matrix(
    examples: list[CalibrationExample],
    judge_scores: list[JudgeScoreRecord],
    required_judges: list[str] | None = None,
    min_judge_coverage: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if min_judge_coverage is not None and not 0.0 <= min_judge_coverage <= 1.0:
        raise ValueError("min_judge_coverage must be in [0, 1]")

    example_ids = {example.calib_example_id for example in examples}
    scores_by_example: dict[str, dict[str, float]] = defaultdict(dict)
    judge_ids: list[str] = []
    seen_judges: set[str] = set()
    for record in judge_scores:
        if record.calib_example_id not in example_ids:
            continue
        if record.judge_id in scores_by_example[record.calib_example_id]:
            raise ValueError(
                f"Duplicate judge score for {(record.calib_example_id, record.judge_id)}"
            )
        scores_by_example[record.calib_example_id][record.judge_id] = validate_score_01(
            record.score, "score"
        )
        if record.judge_id not in seen_judges:
            seen_judges.add(record.judge_id)
            judge_ids.append(record.judge_id)

    if required_judges:
        for judge_id in required_judges:
            if judge_id not in seen_judges:
                judge_ids.append(judge_id)
        ordered_judges = list(dict.fromkeys(required_judges))
    else:
        ordered_judges = sorted(judge_ids)

    rows: list[dict[str, Any]] = []
    dropped = 0
    for example in examples:
        row = _example_to_row(example)
        scores = scores_by_example.get(example.calib_example_id, {})
        present = sum(1 for judge_id in ordered_judges if judge_id in scores)
        coverage = present / len(ordered_judges) if ordered_judges else 0.0
        keep = True
        if required_judges and any(judge_id not in scores for judge_id in required_judges):
            keep = False
        if min_judge_coverage is not None and coverage < min_judge_coverage:
            keep = False
        if not keep:
            dropped += 1
            continue
        for judge_id in ordered_judges:
            row[f"judge::{judge_id}"] = scores.get(judge_id, math.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    coverage_per_judge = {}
    total = len(examples)
    for judge_id in ordered_judges:
        coverage_per_judge[judge_id] = {
            "n_scored": sum(1 for example in examples if judge_id in scores_by_example.get(example.calib_example_id, {})),
            "coverage": (
                sum(1 for example in examples if judge_id in scores_by_example.get(example.calib_example_id, {}))
                / total
                if total
                else 0.0
            ),
        }
    report = {
        "total_examples": total,
        "retained_examples": int(len(df)),
        "donor_judge_ids": ordered_judges,
        "coverage_per_judge": coverage_per_judge,
        "examples_with_complete_coverage": sum(
            1
            for example in examples
            if ordered_judges
            and all(judge_id in scores_by_example.get(example.calib_example_id, {}) for judge_id in ordered_judges)
        ),
        "examples_dropped_by_coverage_filtering": dropped,
    }
    return df, report


def write_score_matrix_csv(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_missingness_report(report: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_mock_judge_scores(
    examples: list[CalibrationExample],
    judge_ids: list[str],
    seed: int,
    noise_std: float,
    bias_by_judge: dict[str, float] | None = None,
) -> list[JudgeScoreRecord]:
    import random

    rng = random.Random(seed)
    bias_by_judge = bias_by_judge or {}
    records: list[JudgeScoreRecord] = []
    for example in examples:
        for judge_id in judge_ids:
            noisy = example.human_score + bias_by_judge.get(judge_id, 0.0) + rng.gauss(0.0, noise_std)
            score = min(1.0, max(0.0, noisy))
            records.append(
                JudgeScoreRecord(
                    calib_example_id=example.calib_example_id,
                    judge_id=judge_id,
                    score=score,
                    score_raw=score,
                    score_scale={"min": 0, "max": 1},
                    rubric_id="mock",
                    prompt_hash=None,
                    model_version="mock",
                    created_at=None,
                    metadata={"mock": True, "seed": seed, "noise_std": noise_std},
                )
            )
    return records


def write_judge_scores_long(path: str | Path, records: Iterable[JudgeScoreRecord]) -> None:
    write_jsonl(path, records)
