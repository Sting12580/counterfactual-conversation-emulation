import json

import pandas as pd

from cce_data.ayers_askdocs import aggregate_human_scores, build_ayers_askdocs_dataset


def _row(idx: int) -> dict:
    return {
        "postID": f"N{idx:03d}",
        "Question": f"patient question {idx}",
        "submissionID": f"submission_{idx}",
        "commentID": f"comment_{idx}",
        "Physician Response": f"physician response {idx}",
        "ChatGPT Response": f"chatgpt response {idx}",
        "Eval 1 Preference": "ChatGPT",
        "Eval 2 Preference": "ChatGPT",
        "Eval 3 Preference": "Physician",
        "Eval 1 Quality (Physician)": 3,
        "Eval 2 Quality (Physician)": 3,
        "Eval 3 Quality (Physician)": 3,
        "Eval 1 Quality (ChatGPT)": 5,
        "Eval 2 Quality (ChatGPT)": 5,
        "Eval 3 Quality (ChatGPT)": 5,
        "Eval 1 Empathy (Physician)": 3,
        "Eval 2 Empathy (Physician)": 3,
        "Eval 3 Empathy (Physician)": 3,
        "Eval 1 Empathy (ChatGPT)": 4,
        "Eval 2 Empathy (ChatGPT)": 4,
        "Eval 3 Empathy (ChatGPT)": 4,
        "Physician Length": 100,
        "ChatGPT Length": 200,
    }


def _frame(n_rows: int = 195) -> pd.DataFrame:
    return pd.DataFrame([_row(idx) for idx in range(n_rows)])


def test_aggregate_human_scores_composite_reward() -> None:
    agg = aggregate_human_scores(_frame(n_rows=1), reward_mode="composite")
    row = agg.iloc[0]

    assert row["n_evaluators"] == 3
    assert row["physician_quality_mean"] == 3
    assert row["physician_empathy_mean"] == 3
    assert row["physician_reward"] == 0.5
    assert row["chatgpt_quality_mean"] == 5
    assert row["chatgpt_empathy_mean"] == 4
    assert row["chatgpt_reward"] == 0.875
    assert row["preference"]["chatgpt_rate"] == 2 / 3


def test_build_ayers_askdocs_dataset_outputs_paired_file(tmp_path) -> None:
    manifest = build_ayers_askdocs_dataset(
        output_dir=tmp_path,
        reward_mode="composite",
        bootstrap=5,
        seed=1,
        frame=_frame(),
    )

    assert manifest["human_baseline_rows"] == 195
    assert manifest["target_counts"] == {"chatgpt": 195}

    target_path = tmp_path / "phase3_chatgpt_expert_scored.jsonl"
    first = json.loads(target_path.read_text().splitlines()[0])
    assert first["a_clinician"].startswith("physician response")
    assert first["a_agent"].startswith("chatgpt response")
    assert first["y_score"] == 0.5
    assert first["y_agent_score"] == 0.875
    assert first["target_responder"] == "chatgpt"

    effect = json.loads((tmp_path / "ground_truth_effect_chatgpt.json").read_text())
    assert effect["n_paired"] == 195
    assert effect["true_effect"] == 0.375
