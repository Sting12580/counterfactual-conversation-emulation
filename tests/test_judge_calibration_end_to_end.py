import json

import pandas as pd

from cce_data.judge_calibration.canonicalize import build_canonical_dataset, write_canonical_outputs
from cce_data.judge_calibration.judge_clients import PHASE7_V2_DONOR_JUDGE_IDS
from cce_data.judge_calibration.runner import run_weighted_judge_combination
from cce_data.judge_calibration.score_matrix import (
    build_score_matrix,
    generate_mock_judge_scores,
    load_calibration_examples,
    write_missingness_report,
    write_score_matrix_csv,
)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _paired_row(prefix: str, idx: int, responder: str) -> dict:
    return {
        "example_id": f"{prefix}_ex{idx}",
        "source_id": f"{prefix}_q{idx}",
        "x_patient_context": f"Patient question:\nquestion {idx}",
        "a_clinician": f"human answer {idx}",
        "a_agent": f"{responder} answer {idx}",
        "y_score": (idx % 5) / 5,
        "y_agent_score": min(1.0, (idx % 5) / 5 + 0.2),
        "target_responder": responder,
        "target_policy": f"policy:{responder}",
        "metadata": {},
    }


def test_end_to_end_smoke_pipeline(tmp_path) -> None:
    for idx in range(10):
        _write_jsonl(
            tmp_path / "data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl",
            [_paired_row("a", idx, "chatgpt") for idx in range(10)],
        )
        _write_jsonl(
            tmp_path / "data/counselbench/phase3_gpt4_expert_scored.jsonl",
            [_paired_row("c", idx, "gpt4") for idx in range(10)],
        )

    examples, summary = build_canonical_dataset(tmp_path, fail_on_missing=False)
    canonical_path = tmp_path / "data/judge_calibration/canonical_examples.jsonl"
    summary_path = tmp_path / "outputs/judge_calibration/data_summary.json"
    write_canonical_outputs(examples, canonical_path, summary_path, summary)
    loaded = load_calibration_examples(canonical_path)
    assert loaded
    assert len({example.calib_example_id for example in loaded}) == len(loaded)

    mock_scores = generate_mock_judge_scores(loaded, PHASE7_V2_DONOR_JUDGE_IDS, 7, 0.02)
    matrix_df, report = build_score_matrix(
        loaded,
        mock_scores,
        required_judges=PHASE7_V2_DONOR_JUDGE_IDS,
    )
    matrix_path = tmp_path / "data/judge_calibration/score_matrix.csv"
    write_score_matrix_csv(matrix_df, matrix_path)
    write_missingness_report(report, tmp_path / "outputs/judge_calibration/missingness.json")
    assert any(column.startswith("judge::") for column in pd.read_csv(matrix_path).columns)
    assert "judge::cohere_command_a_plus" in matrix_df.columns
    assert "judge::google_gemini25flash_think1024" in matrix_df.columns

    out_dir = tmp_path / "outputs/judge_calibration/run"
    run_weighted_judge_combination(matrix_path, out_dir, seed=7)
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "weights.json").exists()
    assert (out_dir / "metrics_overall.json").exists()
    assert (out_dir / "report.md").exists()

    predictions = pd.read_csv(out_dir / "predictions.csv")
    assert predictions["pred_synthetic_judge"].between(0, 1).all()
    weights = json.loads((out_dir / "weights.json").read_text())
    assert abs(sum(weights["weights"].values()) - 1.0) < 1e-8
