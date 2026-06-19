import json

import pytest

from cce_data.judge_calibration.schema import (
    CalibrationExample,
    JudgeScoreRecord,
    calibration_example_from_dict,
    dataclass_to_json_dict,
    judge_score_record_from_dict,
    make_split_group,
    normalize_score,
    stable_text_hash,
    validate_score_01,
    write_jsonl,
    read_jsonl,
)


def _example() -> CalibrationExample:
    return CalibrationExample(
        calib_example_id="ex1",
        dataset_id="ayers_askdocs",
        domain="medical",
        task_type="qa",
        score_dimension="quality",
        question_id="q1",
        question="question",
        context=None,
        answer="answer",
        answer_source="chatgpt",
        human_score=0.5,
        human_score_raw=3,
        human_score_scale={"min": 1, "max": 5},
        human_score_type="expert_scalar",
        human_rubric=None,
        source_file=None,
        split_group=make_split_group("ayers_askdocs", "q1", "quality"),
        metadata={"a": 1},
    )


def test_valid_calibration_example_creation() -> None:
    example = _example()
    assert example.human_score == 0.5
    assert example.split_group == "ayers_askdocs::q1::quality"


def test_invalid_score_rejection() -> None:
    with pytest.raises(ValueError):
        validate_score_01(1.2, "score")
    with pytest.raises(ValueError):
        CalibrationExample(**{**dataclass_to_json_dict(_example()), "human_score": -0.1})


def test_normalize_score_scales() -> None:
    assert normalize_score(3, 1, 5) == 0.5
    assert normalize_score(75, 0, 100) == 0.75


def test_deterministic_split_group_and_hash() -> None:
    assert make_split_group("d", "q", "s") == make_split_group("d", "q", "s")
    assert stable_text_hash("same text") == stable_text_hash("same text")


def test_jsonl_roundtrip_for_dataclasses(tmp_path) -> None:
    example = _example()
    score = JudgeScoreRecord(calib_example_id="ex1", judge_id="judge_a", score=0.4)
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [example, score])
    raw = read_jsonl(path)
    assert calibration_example_from_dict(raw[0]) == example
    assert judge_score_record_from_dict(raw[1]).score == 0.4
    assert json.loads(path.read_text().splitlines()[0])["calib_example_id"] == "ex1"
