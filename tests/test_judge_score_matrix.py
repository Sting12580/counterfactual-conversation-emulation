import pytest

from cce_data.judge_calibration.judge_clients import PHASE7_V2_DONOR_JUDGE_IDS
from cce_data.judge_calibration.schema import CalibrationExample, JudgeScoreRecord, make_split_group
from cce_data.judge_calibration.score_matrix import build_score_matrix, generate_mock_judge_scores


def _examples() -> list[CalibrationExample]:
    examples = []
    for idx in range(3):
        examples.append(
            CalibrationExample(
                calib_example_id=f"ex{idx}",
                dataset_id="d",
                domain="medical",
                task_type="qa",
                score_dimension="quality",
                question_id=f"q{idx}",
                question="q",
                context=None,
                answer="a",
                answer_source="chatgpt",
                human_score=idx / 4,
                human_score_raw=idx / 4,
                human_score_scale={"min": 0, "max": 1},
                human_score_type="expert_scalar",
                human_rubric=None,
                source_file=None,
                split_group=make_split_group("d", f"q{idx}", "quality"),
                metadata={},
            )
        )
    return examples


def test_matrix_has_one_row_per_example() -> None:
    examples = _examples()
    scores = generate_mock_judge_scores(examples, ["a", "b"], seed=1, noise_std=0.0)
    df, report = build_score_matrix(examples, scores)
    assert len(df) == len(examples)
    assert {"judge::a", "judge::b"}.issubset(df.columns)
    assert report["examples_with_complete_coverage"] == 3


def test_duplicate_judge_score_raises() -> None:
    examples = _examples()
    duplicate = [
        JudgeScoreRecord("ex0", "a", 0.1),
        JudgeScoreRecord("ex0", "a", 0.2),
    ]
    with pytest.raises(ValueError):
        build_score_matrix(examples, duplicate)


def test_required_judge_filtering_and_missingness_report() -> None:
    examples = _examples()
    scores = [JudgeScoreRecord("ex0", "a", 0.1), JudgeScoreRecord("ex1", "a", 0.2)]
    df, report = build_score_matrix(examples, scores, required_judges=["a"])
    assert len(df) == 2
    assert report["examples_dropped_by_coverage_filtering"] == 1


def test_mock_generation_is_deterministic() -> None:
    examples = _examples()
    left = generate_mock_judge_scores(examples, ["a"], seed=7, noise_std=0.1)
    right = generate_mock_judge_scores(examples, ["a"], seed=7, noise_std=0.1)
    assert [record.score for record in left] == [record.score for record in right]


def test_v2_required_judge_columns_use_flash_gemini_and_cohere() -> None:
    examples = _examples()
    scores = generate_mock_judge_scores(
        examples,
        PHASE7_V2_DONOR_JUDGE_IDS,
        seed=7,
        noise_std=0.0,
    )
    df, report = build_score_matrix(
        examples,
        scores,
        required_judges=PHASE7_V2_DONOR_JUDGE_IDS,
    )
    assert report["donor_judge_ids"] == PHASE7_V2_DONOR_JUDGE_IDS
    assert "judge::google_gemini25flash_think1024" in df.columns
    assert "judge::cohere_command_a_plus" in df.columns
    assert "judge::deepseek_v4flash" not in df.columns
    assert "judge::google_gemini25pro" not in df.columns
