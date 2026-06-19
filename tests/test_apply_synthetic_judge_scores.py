import json

import pandas as pd

from cce_data.judge_calibration.apply import (
    apply_predictions_to_paired_files,
    write_answer_level_predictions,
)
from cce_data.judge_calibration.schema import CalibrationExample, make_split_group, write_jsonl


def _examples() -> list[CalibrationExample]:
    split_group = make_split_group("ayers_askdocs", "q1", "quality")
    return [
        CalibrationExample(
            calib_example_id="human_ex",
            dataset_id="ayers_askdocs",
            domain="medical",
            task_type="qa",
            score_dimension="quality",
            question_id="q1",
            question="q",
            context="Patient question:\nq",
            answer="human",
            answer_source="human_clinician",
            human_score=0.2,
            human_score_raw=0.2,
            human_score_scale={"min": 0, "max": 1},
            human_score_type="expert_scalar",
            human_rubric=None,
            source_file=None,
            split_group=split_group,
            metadata={"source_example_id": "paired1", "answer_field": "a_clinician"},
        ),
        CalibrationExample(
            calib_example_id="agent_ex",
            dataset_id="ayers_askdocs",
            domain="medical",
            task_type="qa",
            score_dimension="quality",
            question_id="q1",
            question="q",
            context="Patient question:\nq",
            answer="agent",
            answer_source="chatgpt",
            human_score=0.8,
            human_score_raw=0.8,
            human_score_scale={"min": 0, "max": 1},
            human_score_type="expert_scalar",
            human_rubric=None,
            source_file=None,
            split_group=split_group,
            metadata={"source_example_id": "paired1", "answer_field": "a_agent"},
        ),
    ]


def test_apply_weights_and_preserve_original_fields(tmp_path) -> None:
    examples_path = tmp_path / "canonical.jsonl"
    write_jsonl(examples_path, _examples())
    matrix = tmp_path / "score_matrix.csv"
    pd.DataFrame(
        [
            {"calib_example_id": "human_ex", "judge::a": 0.3, "judge::b": 0.5},
            {"calib_example_id": "agent_ex", "judge::a": 0.7, "judge::b": 0.9},
        ]
    ).to_csv(matrix, index=False)
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps({"weights": {"a": 0.25, "b": 0.75}, "intercept": 0.0, "clip_predictions": True})
    )
    out_predictions = tmp_path / "predictions.jsonl"
    predictions = write_answer_level_predictions(examples_path, matrix, weights, out_predictions)
    assert len(predictions) == 2
    assert all(0.0 <= pred["synthetic_judge_score"] <= 1.0 for pred in predictions)

    paired = tmp_path / "paired.jsonl"
    paired.write_text(
        json.dumps({"example_id": "paired1", "y_score": 0.2, "y_agent_score": 0.8}) + "\n",
        encoding="utf-8",
    )
    written = apply_predictions_to_paired_files(
        examples_path, predictions, [paired], tmp_path / "paired_out", overwrite_reward_fields=False
    )
    row = json.loads(written[0].read_text().splitlines()[0])
    assert row["y_score"] == 0.2
    assert row["y_agent_score"] == 0.8
    assert "y_score_calibrated" in row
    assert "y_agent_score_calibrated" in row


def test_apply_paired_files_avoids_duplicate_stem_overwrite(tmp_path) -> None:
    examples_path = tmp_path / "canonical.jsonl"
    write_jsonl(examples_path, _examples())
    predictions = [
        {"calib_example_id": "human_ex", "synthetic_judge_score": 0.25, "calibration_model_id": "m"},
        {"calib_example_id": "agent_ex", "synthetic_judge_score": 0.75, "calibration_model_id": "m"},
    ]
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "paired.jsonl"
    second = second_dir / "paired.jsonl"
    payload = json.dumps({"example_id": "paired1", "y_score": 0.2, "y_agent_score": 0.8}) + "\n"
    first.write_text(payload, encoding="utf-8")
    second.write_text(payload, encoding="utf-8")

    written = apply_predictions_to_paired_files(
        examples_path,
        predictions,
        [first, second],
        tmp_path / "paired_out",
        overwrite_reward_fields=False,
    )

    assert len(written) == 2
    assert written[0].name == "first__paired_calibrated.jsonl"
    assert written[1].name == "second__paired_calibrated.jsonl"
    assert written[0].exists()
    assert written[1].exists()
