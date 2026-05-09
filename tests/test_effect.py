import json

from cce_data.effect import compute_ground_truth_effect


def test_compute_ground_truth_effect(tmp_path):
    path = tmp_path / "scored.jsonl"
    rows = [
        {"example_id": "a", "y_score": 0.6, "y_agent_score": 0.8},
        {"example_id": "b", "y_score": 0.7, "y_agent_score": 0.5},
        {"example_id": "c", "y_score": 0.5, "y_agent_score": 0.6},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output = tmp_path / "effect.json"
    result = compute_ground_truth_effect(path, output, bootstrap=20, seed=7)

    assert result["n_paired"] == 3
    assert round(result["v_true_pi_b"], 4) == 0.6
    assert round(result["v_true_pi_agent"], 4) == 0.6333
    assert round(result["true_effect"], 4) == 0.0333
    assert output.exists()
