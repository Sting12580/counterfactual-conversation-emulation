import json
from pathlib import Path

from cce_data.judge_calibration.canonicalize import (
    build_canonical_dataset,
    canonicalize_ayers_file,
    canonicalize_counselbench_files,
    canonicalize_liveqa_medical_2017_source,
    canonicalize_mediqa_qa_2019_source,
)


def _paired_row(example_id: str, source_id: str, target_responder: str = "gpt4") -> dict:
    return {
        "example_id": example_id,
        "source_id": source_id,
        "x_patient_context": f"Patient question:\nquestion {source_id}",
        "a_clinician": f"human answer {source_id}",
        "a_agent": f"{target_responder} answer {source_id}",
        "y_score": 0.25,
        "y_agent_score": 0.75,
        "y_rubric": "human rubric",
        "y_agent_rubric": "agent rubric",
        "target_responder": target_responder,
        "target_policy": f"policy:{target_responder}",
        "metadata": {"topic": "anxiety"},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_ayers_paired_row_becomes_two_records(tmp_path) -> None:
    path = tmp_path / "phase3_chatgpt_expert_scored.jsonl"
    _write_jsonl(path, [_paired_row("ex1", "q1", "chatgpt")])
    records = canonicalize_ayers_file(path, "quality")
    assert len(records) == 2
    assert {record.answer_source for record in records} == {"human_clinician", "chatgpt"}
    assert {record.human_score for record in records} == {0.25, 0.75}


def test_counselbench_repeated_human_baseline_is_deduplicated(tmp_path) -> None:
    path1 = tmp_path / "phase3_gpt4_expert_scored.jsonl"
    path2 = tmp_path / "phase3_llama3_expert_scored.jsonl"
    _write_jsonl(path1, [_paired_row("ex1", "q1", "gpt4")])
    _write_jsonl(path2, [_paired_row("ex1", "q1", "llama3")])
    records, removed = canonicalize_counselbench_files([path1, path2])
    assert removed == 1
    assert [record.answer_source for record in records].count("human_counselor") == 1
    assert {record.answer_source for record in records} == {"human_counselor", "gpt4", "llama3"}


def test_build_canonical_dataset_on_temp_repo(tmp_path) -> None:
    ayers_path = tmp_path / "data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl"
    counsel_path = tmp_path / "data/counselbench/phase3_gpt4_expert_scored.jsonl"
    _write_jsonl(ayers_path, [_paired_row("ayers_ex1", "aq1", "chatgpt")])
    _write_jsonl(counsel_path, [_paired_row("counsel_ex1", "cq1", "gpt4")])
    examples, summary = build_canonical_dataset(
        repo_root=tmp_path,
        include_ayers=True,
        include_counselbench=True,
        include_mediqa_qa_2019=False,
        include_liveqa_medical_2017=False,
        fail_on_missing=False,
    )
    assert len(examples) == 4
    assert summary["total_examples"] == 4
    assert summary["human_score_min"] == 0.25
    assert all(example.split_group for example in examples)


def test_ayers_default_uses_composite_only_even_when_all_dimensions_exist(tmp_path) -> None:
    for dirname in ["ayers_askdocs", "ayers_askdocs_quality", "ayers_askdocs_empathy"]:
        _write_jsonl(
            tmp_path / f"data/{dirname}/phase3_chatgpt_expert_scored.jsonl",
            [_paired_row(f"{dirname}_ex1", "aq1", "chatgpt")],
        )
    examples, summary = build_canonical_dataset(
        repo_root=tmp_path,
        include_ayers=True,
        include_counselbench=False,
        include_mediqa_qa_2019=False,
        include_liveqa_medical_2017=False,
        fail_on_missing=False,
    )
    assert len(examples) == 2
    assert {example.score_dimension for example in examples} == {"composite"}
    assert summary["by_score_dimension"] == {"composite": 2}


def test_ayers_all_dimensions_are_diagnostic_opt_in(tmp_path) -> None:
    for dirname in ["ayers_askdocs", "ayers_askdocs_quality", "ayers_askdocs_empathy"]:
        _write_jsonl(
            tmp_path / f"data/{dirname}/phase3_chatgpt_expert_scored.jsonl",
            [_paired_row(f"{dirname}_ex1", "aq1", "chatgpt")],
        )
    examples, summary = build_canonical_dataset(
        repo_root=tmp_path,
        include_ayers=True,
        ayers_dimensions="composite,quality,empathy",
        include_counselbench=False,
        include_mediqa_qa_2019=False,
        include_liveqa_medical_2017=False,
        fail_on_missing=False,
    )
    assert len(examples) == 6
    assert summary["by_score_dimension"] == {"composite": 2, "quality": 2, "empathy": 2}


def test_summary_reports_missing_and_skipped_counts(tmp_path) -> None:
    mediqa_path = tmp_path / "data/mediqa_qa_2019/answers.jsonl"
    _write_jsonl(
        mediqa_path,
        [
            {
                "id": "q1",
                "question": "What should I do for a sore throat?",
                "answer": "Drink fluids and rest.",
            }
        ],
    )
    examples, summary = build_canonical_dataset(
        repo_root=tmp_path,
        include_ayers=False,
        include_counselbench=False,
        include_mediqa_qa_2019=True,
        include_liveqa_medical_2017=True,
        fail_on_missing=False,
    )
    assert examples == []
    for key in [
        "by_dataset_id",
        "by_score_dimension",
        "by_answer_source",
        "missing_files",
        "skipped_examples",
        "skipped_by_dataset_id",
    ]:
        assert key in summary
    assert summary["skipped_by_dataset_id"]["mediqa_qa_2019"] == 1
    assert summary["dataset_summaries"]["mediqa_qa_2019"]["skipped_reasons"] == {
        "no_usable_score": 1
    }
    assert summary["missing_files"]


def test_mediqa_loader_normalizes_raw_1_to_4_scores_and_skips_unscored(tmp_path) -> None:
    path = tmp_path / "data/mediqa_qa_2019/answers.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "mediqa_q1",
                "question": "What can help mild dehydration?",
                "answer": "Oral fluids can help if symptoms are mild.",
                "raw_score": 4,
                "system": "system_a",
            },
            {
                "id": "mediqa_q2",
                "question": "What can help a cough?",
                "answer": "No score here.",
                "system": "system_b",
            }
        ],
    )
    examples, summary = canonicalize_mediqa_qa_2019_source(path.parent)
    assert len(examples) == 1
    assert summary["included_examples"] == 1
    assert summary["skipped_examples"] == 1
    assert summary["skipped_reasons"] == {"no_usable_score": 1}
    assert examples[0].dataset_id == "mediqa_qa_2019"
    assert examples[0].domain == "medical"
    assert examples[0].score_dimension == "medical_answer_quality"
    assert examples[0].human_score == 1.0
    assert examples[0].human_score_type == "medical_expert_rating_1_4"
    assert len({example.calib_example_id for example in examples}) == len(examples)


def test_liveqa_loader_accepts_official_0_to_3_and_raw_1_to_4_grade(tmp_path) -> None:
    path = tmp_path / "data/liveqa_medical_2017/answers.jsonl"
    _write_jsonl(
        path,
        [
            {
                "qid": "liveqa_q1",
                "question": "Can a fever be dangerous?",
                "answer": "High or persistent fever needs medical attention.",
                "official_score": 3,
                "answer_source": "team_a",
            },
            {
                "qid": "liveqa_q2",
                "question": "Can I take antibiotics for a cold?",
                "answer": "Antibiotics do not treat viral colds.",
                "grade": 1,
                "answer_source": "team_b",
            },
        ],
    )
    examples, summary = canonicalize_liveqa_medical_2017_source(path.parent)
    assert len(examples) == 2
    assert summary["included_examples"] == 2
    assert {example.dataset_id for example in examples} == {"liveqa_medical_2017"}
    assert {example.score_dimension for example in examples} == {"medical_answer_quality"}
    scores = {example.question_id: example.human_score for example in examples}
    assert scores == {"liveqa_q1": 1.0, "liveqa_q2": 0.0}
    types = {example.question_id: example.human_score_type for example in examples}
    assert types == {
        "liveqa_q1": "official_liveqa_score_0_3",
        "liveqa_q2": "raw_liveqa_grade_1_4",
    }
