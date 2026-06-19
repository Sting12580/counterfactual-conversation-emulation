from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cce_data.judge_calibration.schema import calibration_example_from_dict, read_jsonl, write_jsonl


def load_weights(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply_weights_to_score_matrix(
    score_matrix: str | Path,
    weights_json: str | Path,
) -> pd.DataFrame:
    df = pd.read_csv(score_matrix)
    weights_payload = load_weights(weights_json)
    weights = weights_payload.get("weights")
    if not weights:
        raise ValueError("weights_json must contain a non-empty weights object")
    intercept = float(weights_payload.get("intercept", 0.0))
    clip_predictions = bool(weights_payload.get("clip_predictions", True))
    pred = np.zeros(len(df), dtype=float) + intercept
    missing_columns: list[str] = []
    for judge_id, weight in weights.items():
        column = f"judge::{judge_id}"
        if column not in df.columns:
            missing_columns.append(column)
            continue
        pred += float(weight) * pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if missing_columns:
        raise ValueError(f"Score matrix is missing judge columns required by weights: {missing_columns}")
    if not np.all(np.isfinite(pred)):
        raise ValueError("Synthetic predictions contain non-finite values")
    if clip_predictions:
        pred = np.clip(pred, 0.0, 1.0)
    out = df.copy()
    out["synthetic_judge_score"] = pred
    out["calibration_model_id"] = weights_payload.get("model_id") or Path(weights_json).stem
    return out


def write_answer_level_predictions(
    canonical_examples: str | Path,
    score_matrix: str | Path,
    weights_json: str | Path,
    out_predictions: str | Path,
) -> list[dict[str, Any]]:
    examples = {
        record["calib_example_id"]: calibration_example_from_dict(record)
        for record in read_jsonl(canonical_examples)
    }
    pred_df = apply_weights_to_score_matrix(score_matrix, weights_json)
    predictions: list[dict[str, Any]] = []
    weights_payload = load_weights(weights_json)
    timestamp = datetime.now(timezone.utc).isoformat()
    for row in pred_df.to_dict(orient="records"):
        example = examples.get(row["calib_example_id"])
        if example is None:
            continue
        predictions.append(
            {
                "calib_example_id": example.calib_example_id,
                "dataset_id": example.dataset_id,
                "question_id": example.question_id,
                "score_dimension": example.score_dimension,
                "answer_source": example.answer_source,
                "synthetic_judge_score": float(row["synthetic_judge_score"]),
                "human_score": example.human_score,
                "calibration_model_id": row["calibration_model_id"],
                "metadata": {
                    "weights_file": str(weights_json),
                    "score_matrix_file": str(score_matrix),
                    "created_at": timestamp,
                    "weights": weights_payload.get("weights"),
                },
            }
        )
    write_jsonl(out_predictions, predictions)
    return predictions


def _prediction_lookup(
    canonical_examples: str | Path,
    predictions: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    examples = {
        record["calib_example_id"]: calibration_example_from_dict(record)
        for record in read_jsonl(canonical_examples)
    }
    by_id = {record["calib_example_id"]: record for record in predictions}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for calib_id, example in examples.items():
        pred = by_id.get(calib_id)
        if pred is None:
            continue
        metadata = example.metadata or {}
        source_example_id = metadata.get("source_example_id")
        answer_field = metadata.get("answer_field")
        if source_example_id and answer_field:
            lookup[(str(source_example_id), str(answer_field))] = pred
    return lookup


def apply_predictions_to_paired_files(
    canonical_examples: str | Path,
    predictions: list[dict[str, Any]],
    paired_inputs: list[str | Path],
    paired_output_dir: str | Path,
    overwrite_reward_fields: bool = False,
    on_missing: str = "fail",
) -> list[Path]:
    if on_missing not in {"fail", "skip"}:
        raise ValueError("on_missing must be 'fail' or 'skip'")
    lookup = _prediction_lookup(canonical_examples, predictions)
    out_dir = Path(paired_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    input_paths = [Path(input_path) for input_path in paired_inputs]
    stem_counts = Counter(input_path.stem for input_path in input_paths)
    used_output_names: set[str] = set()
    for input_path in input_paths:
        rows = read_jsonl(input_path)
        output_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for row in rows:
            source_example_id = str(row.get("example_id"))
            clinician_pred = lookup.get((source_example_id, "a_clinician"))
            agent_pred = lookup.get((source_example_id, "a_agent"))
            if clinician_pred is None or agent_pred is None:
                missing.append(source_example_id)
                if on_missing == "skip":
                    continue
                continue
            updated = dict(row)
            if overwrite_reward_fields:
                updated.setdefault("y_score_raw_existing", row.get("y_score"))
                updated.setdefault("y_agent_score_raw_existing", row.get("y_agent_score"))
                updated["y_score"] = clinician_pred["synthetic_judge_score"]
                updated["y_agent_score"] = agent_pred["synthetic_judge_score"]
            updated["y_score_calibrated"] = clinician_pred["synthetic_judge_score"]
            updated["y_agent_score_calibrated"] = agent_pred["synthetic_judge_score"]
            updated["y_score_calibration_model_id"] = clinician_pred["calibration_model_id"]
            updated["y_agent_score_calibration_model_id"] = agent_pred["calibration_model_id"]
            updated["calibration_metadata"] = {
                "clinician_calib_example_id": clinician_pred["calib_example_id"],
                "agent_calib_example_id": agent_pred["calib_example_id"],
                "overwrite_reward_fields": overwrite_reward_fields,
            }
            output_rows.append(updated)
        if missing and on_missing == "fail":
            raise ValueError(f"Missing calibrated predictions for {len(missing)} rows in {input_path}")
        output_name = f"{input_path.stem}_calibrated.jsonl"
        if stem_counts[input_path.stem] > 1:
            output_name = f"{input_path.parent.name}__{output_name}"
        if output_name in used_output_names:
            suffix = 2
            base_name = output_name.removesuffix(".jsonl")
            while f"{base_name}_{suffix}.jsonl" in used_output_names:
                suffix += 1
            output_name = f"{base_name}_{suffix}.jsonl"
        used_output_names.add(output_name)
        output_path = out_dir / output_name
        write_jsonl(output_path, output_rows)
        written.append(output_path)
    return written
