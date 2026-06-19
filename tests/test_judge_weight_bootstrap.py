import pandas as pd

from cce_data.judge_calibration.stability import write_bootstrap_weight_stability


def test_bootstrap_weight_stability_outputs_files(tmp_path) -> None:
    rows = []
    for idx in range(60):
        y = (idx % 10) / 10
        dataset = "d1" if idx < 30 else "d2"
        dimension = "quality" if idx % 2 else "empathy"
        rows.append(
            {
                "calib_example_id": f"ex{idx}",
                "dataset_id": dataset,
                "score_dimension": dimension,
                "question_id": f"q{idx}",
                "answer_source": "human" if idx % 2 else "agent",
                "human_score": y,
                "split_group": f"{dataset}::q{idx}::{dimension}",
                "judge::a": min(1, max(0, y + 0.02)),
                "judge::b": min(1, max(0, y * 0.7 + 0.1)),
                "judge::c": min(1, max(0, 1 - y)),
            }
        )
    matrix = tmp_path / "score_matrix.csv"
    pd.DataFrame(rows).to_csv(matrix, index=False)
    out_dir = tmp_path / "bootstrap"

    summary = write_bootstrap_weight_stability(
        score_matrix=matrix,
        out_dir=out_dir,
        n_bootstrap=5,
        seed=11,
        group_balance=True,
    )

    assert summary["n_bootstrap"] == 5
    assert (out_dir / "bootstrap_weight_samples.csv").exists()
    assert (out_dir / "bootstrap_weight_summary.csv").exists()
    assert (out_dir / "bootstrap_config.json").exists()
    weight_summary = pd.read_csv(out_dir / "bootstrap_weight_summary.csv")
    assert set(weight_summary["judge_id"]) == {"a", "b", "c"}
