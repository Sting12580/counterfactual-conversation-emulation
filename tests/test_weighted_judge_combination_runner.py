import json

import pandas as pd

from cce_data.judge_calibration.runner import run_weighted_judge_combination


def _score_matrix(path) -> None:
    rows = []
    for idx in range(36):
        y = (idx % 10) / 10
        dataset = "d1" if idx < 18 else "d2"
        rows.append(
            {
                "calib_example_id": f"ex{idx}",
                "dataset_id": dataset,
                "domain": "medical",
                "task_type": "qa",
                "score_dimension": "quality",
                "question_id": f"q{idx}",
                "question": "q",
                "context": "",
                "answer": "a",
                "answer_source": "chatgpt" if idx % 2 else "human",
                "human_score": y,
                "split_group": f"{dataset}::q{idx}::quality",
                "judge::a": min(1, max(0, y + 0.01)),
                "judge::b": min(1, max(0, y * 0.8 + 0.05)),
                "judge::c": min(1, max(0, 1 - y)),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_runner_grouped_random_outputs_files(tmp_path) -> None:
    matrix = tmp_path / "score_matrix.csv"
    _score_matrix(matrix)
    out_dir = tmp_path / "run"
    run_weighted_judge_combination(matrix, out_dir, seed=3)
    assert (out_dir / "weights.json").exists()
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "metrics_overall.json").exists()
    assert (out_dir / "report.md").exists()
    assert (out_dir / "worst_groups.csv").exists()
    assert (out_dir / "leave_one_judge_sensitivity.csv").exists()
    assert (out_dir / "judge_correlation.csv").exists()
    weights = json.loads((out_dir / "weights.json").read_text())
    assert abs(sum(weights["weights"].values()) - 1.0) < 1e-8
    metrics = json.loads((out_dir / "metrics_overall.json").read_text())
    assert {row["model"] for row in metrics} >= {"pred_synthetic_judge", "pred_unweighted_mean"}


def test_runner_lodo_outputs_one_folder_per_dataset(tmp_path) -> None:
    matrix = tmp_path / "score_matrix.csv"
    _score_matrix(matrix)
    out_dir = tmp_path / "lodo"
    run_weighted_judge_combination(matrix, out_dir, split_protocol="leave_one_dataset_out", seed=4)
    assert (out_dir / "leave_one_dataset_id_d1" / "weights.json").exists()
    assert (out_dir / "leave_one_dataset_id_d2" / "weights.json").exists()
    assert (out_dir / "aggregate_summary.csv").exists()
