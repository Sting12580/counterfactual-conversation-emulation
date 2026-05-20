import json

import pandas as pd

from cce_data.counselbench import aggregate_expert_scores, build_counselbench_dataset


def _row(question_id: str, responder: str, survey_id: int, overall: int, response: str) -> dict:
    return {
        "questionID": question_id,
        "questionTitle": f"title {question_id}",
        "questionText": f"text {question_id}",
        "response": response,
        "topic": "anxiety",
        "responder": responder,
        "survey_id": survey_id,
        "overall_score": overall,
        "overall_reason": f"reason {survey_id}",
        "empathy_score": overall,
        "specificity_score": overall,
        "medical_advice_score": "No",
        "medical_copy": None,
        "medical_reason": None,
        "factual_consistency_score": "4",
        "factual_copy": None,
        "factual_reason": None,
        "toxicity_score": 1,
        "toxicity_copy": None,
        "toxicity_reason": None,
    }


def _frame(n_questions: int = 100) -> pd.DataFrame:
    rows = []
    responders = {
        "human": 3,
        "gpt4": 4,
        "llama3": 5,
        "gemini": 2,
    }
    for q_idx in range(n_questions):
        question_id = f"questionID_{q_idx}"
        for responder, overall in responders.items():
            for survey_id in range(5):
                rows.append(
                    _row(
                        question_id=question_id,
                        responder=responder,
                        survey_id=survey_id,
                        overall=overall,
                        response=f"{responder} response {q_idx}",
                    )
                )
    return pd.DataFrame(rows)


def test_aggregate_expert_scores_composite_reward() -> None:
    agg = aggregate_expert_scores(_frame(n_questions=1), reward_mode="composite")

    human = agg[(agg["questionID"] == "questionID_0") & (agg["responder"] == "human")].iloc[0]
    assert human["n_annotations"] == 5
    assert human["overall_mean"] == 3
    assert 0.0 <= human["expert_reward"] <= 1.0
    assert human["medical_advice_yes_rate"] == 0.0


def test_build_counselbench_dataset_outputs_paired_files(tmp_path) -> None:
    manifest = build_counselbench_dataset(
        output_dir=tmp_path,
        reward_mode="overall",
        bootstrap=5,
        seed=1,
        frame=_frame(),
    )

    assert manifest["human_baseline_rows"] == 100
    assert manifest["target_counts"] == {"gpt4": 100, "llama3": 100, "gemini": 100}

    gpt4_path = tmp_path / "phase3_gpt4_expert_scored.jsonl"
    first = json.loads(gpt4_path.read_text().splitlines()[0])
    assert first["a_clinician"].startswith("human response")
    assert first["a_agent"].startswith("gpt4 response")
    assert first["y_score"] == 0.5
    assert first["y_agent_score"] == 0.75

    effect = json.loads((tmp_path / "ground_truth_effect_gpt4.json").read_text())
    assert effect["n_paired"] == 100
    assert effect["true_effect"] == 0.25
