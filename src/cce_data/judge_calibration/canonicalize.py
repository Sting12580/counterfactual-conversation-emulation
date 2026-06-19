from __future__ import annotations

import json
import math
import csv
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from cce_data.judge_calibration.schema import (
    CalibrationExample,
    make_split_group,
    read_jsonl,
    stable_text_hash,
    validate_score_01,
    write_jsonl,
)


AYERS_COMPOSITE_SOURCES = [
    (
        "ayers_askdocs",
        "composite",
        Path("data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl"),
    ),
]

AYERS_ALL_DIMENSION_SOURCES = [
    *AYERS_COMPOSITE_SOURCES,
    (
        "ayers_askdocs",
        "quality",
        Path("data/ayers_askdocs_quality/phase3_chatgpt_expert_scored.jsonl"),
    ),
    (
        "ayers_askdocs",
        "empathy",
        Path("data/ayers_askdocs_empathy/phase3_chatgpt_expert_scored.jsonl"),
    ),
]

# Backward-compatible public name. The Phase 7 v2 default is composite only.
AYERS_SOURCES = AYERS_COMPOSITE_SOURCES
AYERS_SOURCE_BY_DIMENSION = {
    score_dimension: (dataset_id, score_dimension, path)
    for dataset_id, score_dimension, path in AYERS_ALL_DIMENSION_SOURCES
}

COUNSELBENCH_SOURCES = [
    Path("data/counselbench/phase3_gpt4_expert_scored.jsonl"),
    Path("data/counselbench/phase3_llama3_expert_scored.jsonl"),
    Path("data/counselbench/phase3_gemini_expert_scored.jsonl"),
]

MEDIQA_QA_2019_SOURCES = [
    Path("data/mediqa_qa_2019"),
    Path("data/raw/mediqa_qa_2019"),
]

LIVEQA_MEDICAL_2017_SOURCES = [
    Path("data/liveqa_medical_2017"),
    Path("data/raw/liveqa_medical_2017"),
]


def _score(value: Any, field_name: str) -> float:
    return validate_score_01(value, field_name)


def _question_from_context(context: str) -> str:
    if not context:
        return ""
    marker = "Patient question:\n"
    if marker in context:
        return context.split(marker, 1)[1].strip()
    return context.strip()


def _example_id(
    dataset_id: str,
    question_id: str,
    answer_source: str,
    answer: str,
    score_dimension: str,
) -> str:
    return "::".join(
        [
            dataset_id,
            score_dimension,
            question_id,
            answer_source,
            stable_text_hash(answer),
        ]
    )


def _base_metadata(row: dict[str, Any], answer_field: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "source_example_id": row.get("example_id"),
            "target_policy": row.get("target_policy"),
            "target_responder": row.get("target_responder"),
            "answer_field": answer_field,
        }
    )
    return metadata


def canonicalize_ayers_file(
    path: str | Path,
    score_dimension: str,
    dataset_id: str = "ayers_askdocs",
) -> list[CalibrationExample]:
    source_path = Path(path)
    examples: list[CalibrationExample] = []
    for row in read_jsonl(source_path):
        question_id = str(row.get("source_id") or row.get("example_id"))
        context = row.get("x_patient_context") or ""
        question = _question_from_context(context)
        split_group = make_split_group(dataset_id, question_id, score_dimension)

        clinician_answer = row.get("a_clinician") or ""
        examples.append(
            CalibrationExample(
                calib_example_id=_example_id(
                    dataset_id,
                    question_id,
                    "human_clinician",
                    clinician_answer,
                    score_dimension,
                ),
                dataset_id=dataset_id,
                domain="medical",
                task_type="medical_response",
                score_dimension=score_dimension,
                question_id=question_id,
                question=question or context,
                context=context,
                answer=clinician_answer,
                answer_source="human_clinician",
                human_score=_score(row.get("y_score"), "y_score"),
                human_score_raw=row.get("y_score"),
                human_score_scale={"min": 0, "max": 1},
                human_score_type="expert_scalar",
                human_rubric=row.get("y_rubric"),
                source_file=str(source_path),
                split_group=split_group,
                metadata=_base_metadata(row, "a_clinician"),
            )
        )

        agent_answer = row.get("a_agent") or ""
        target_responder = str(row.get("target_responder") or "chatgpt")
        examples.append(
            CalibrationExample(
                calib_example_id=_example_id(
                    dataset_id,
                    question_id,
                    target_responder,
                    agent_answer,
                    score_dimension,
                ),
                dataset_id=dataset_id,
                domain="medical",
                task_type="medical_response",
                score_dimension=score_dimension,
                question_id=question_id,
                question=question or context,
                context=context,
                answer=agent_answer,
                answer_source=target_responder,
                human_score=_score(row.get("y_agent_score"), "y_agent_score"),
                human_score_raw=row.get("y_agent_score"),
                human_score_scale={"min": 0, "max": 1},
                human_score_type="expert_scalar",
                human_rubric=row.get("y_agent_rubric"),
                source_file=str(source_path),
                split_group=split_group,
                metadata=_base_metadata(row, "a_agent"),
            )
        )
    return examples


def canonicalize_counselbench_files(
    paths: Iterable[str | Path],
    score_dimension: str = "counseling_quality",
    dataset_id: str = "counselbench",
) -> tuple[list[CalibrationExample], int]:
    examples: list[CalibrationExample] = []
    seen_human: set[tuple[str, str, str, str, str]] = set()
    duplicates_removed = 0

    for path in paths:
        source_path = Path(path)
        for row in read_jsonl(source_path):
            question_id = str(row.get("source_id") or row.get("example_id"))
            context = row.get("x_patient_context") or ""
            question = _question_from_context(context)
            split_group = make_split_group(dataset_id, question_id, score_dimension)

            human_answer = row.get("a_clinician") or ""
            human_key = (
                dataset_id,
                question_id,
                "human_counselor",
                stable_text_hash(human_answer),
                score_dimension,
            )
            if human_key in seen_human:
                duplicates_removed += 1
            else:
                seen_human.add(human_key)
                examples.append(
                    CalibrationExample(
                        calib_example_id=_example_id(
                            dataset_id,
                            question_id,
                            "human_counselor",
                            human_answer,
                            score_dimension,
                        ),
                        dataset_id=dataset_id,
                        domain="counseling",
                        task_type="counseling_qa",
                        score_dimension=score_dimension,
                        question_id=question_id,
                        question=question or context,
                        context=context,
                        answer=human_answer,
                        answer_source="human_counselor",
                        human_score=_score(row.get("y_score"), "y_score"),
                        human_score_raw=row.get("y_score"),
                        human_score_scale={"min": 0, "max": 1},
                        human_score_type="expert_scalar",
                        human_rubric=row.get("y_rubric"),
                        source_file=str(source_path),
                        split_group=split_group,
                        metadata=_base_metadata(row, "a_clinician"),
                    )
                )

            target_answer = row.get("a_agent") or ""
            target_responder = str(row.get("target_responder") or "target_model")
            examples.append(
                CalibrationExample(
                    calib_example_id=_example_id(
                        dataset_id,
                        question_id,
                        target_responder,
                        target_answer,
                        score_dimension,
                    ),
                    dataset_id=dataset_id,
                    domain="counseling",
                    task_type="counseling_qa",
                    score_dimension=score_dimension,
                    question_id=question_id,
                    question=question or context,
                    context=context,
                    answer=target_answer,
                    answer_source=target_responder,
                    human_score=_score(row.get("y_agent_score"), "y_agent_score"),
                    human_score_raw=row.get("y_agent_score"),
                    human_score_scale={"min": 0, "max": 1},
                    human_score_type="expert_scalar",
                    human_rubric=row.get("y_agent_rubric"),
                    source_file=str(source_path),
                    split_group=split_group,
                    metadata=_base_metadata(row, "a_agent"),
                )
            )
    return examples, duplicates_removed


POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "correct",
    "accurate",
    "supported",
    "fully_supported",
    "useful",
    "helpful",
    "reliable",
    "worthy",
    "informative",
    "satisfied",
    "met",
    "pass",
    "passes",
}

NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "incorrect",
    "inaccurate",
    "unsupported",
    "not_supported",
    "not_useful",
    "unhelpful",
    "unreliable",
    "not_worthy",
    "uninformative",
    "not_satisfied",
    "unmet",
    "fail",
    "fails",
}

PARTIAL_LABELS = {
    "partial": 0.5,
    "partially_supported": 0.5,
    "partially_correct": 0.5,
    "mixed": 0.5,
    "somewhat": 0.5,
    "mostly": 0.75,
    "mostly_correct": 0.75,
    "mostly_supported": 0.75,
    "minor_issue": 0.75,
    "major_issue": 0.25,
}


def _label_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_component_value(value: Any, field_name: str) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, dict):
        for nested_key in ("value", "score", "label", "rating"):
            if nested_key in value:
                return _normalize_component_value(value[nested_key], field_name)
        return None
    if isinstance(value, str):
        key = _label_key(value)
        if key in POSITIVE_LABELS:
            return 1.0
        if key in NEGATIVE_LABELS:
            return 0.0
        if key in PARTIAL_LABELS:
            return PARTIAL_LABELS[key]
        try:
            value = float(value)
        except ValueError:
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric
    if 1.0 < numeric <= 5.0:
        return (numeric - 1.0) / 4.0
    if 0.0 <= numeric <= 100.0:
        return numeric / 100.0
    return None


def _text_from_obj(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _compact_context(parts: list[tuple[str, Any]], max_chars: int = 4000) -> str | None:
    lines: list[str] = []
    for label, value in parts:
        text = _text_from_obj(value).strip()
        if text:
            lines.append(f"{label}: {text}")
    if not lines:
        return None
    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[: max_chars - 20].rstrip() + "\n[truncated]"
    return context


MEDICAL_QA_QUESTION_FIELDS = (
    "question",
    "question_text",
    "question_body",
    "query",
    "qtext",
    "body",
    "problem",
    "title",
)

MEDICAL_QA_ANSWER_FIELDS = (
    "answer",
    "answer_text",
    "answer_body",
    "response",
    "candidate_answer",
    "system_answer",
    "summary",
    "text",
    "output",
)

MEDIQA_SCORE_FIELDS_1_TO_4 = (
    "raw_score",
    "raw_rating",
    "rating",
    "medical_expert_rating",
    "expert_rating",
    "expert_score",
    "score",
    "grade",
)

LIVEQA_OFFICIAL_SCORE_FIELDS_0_TO_3 = (
    "official_score",
    "official_0_3_score",
    "trec_score",
    "score_0_3",
    "score",
)

LIVEQA_GRADE_FIELDS_1_TO_4 = (
    "raw_grade",
    "grade",
    "rating",
    "raw_rating",
)


def _read_records_from_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    return list(value)
            return [payload]
        return []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    return []


def _source_record_files(path: str | Path) -> list[Path]:
    source_path = Path(path)
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        return sorted(
            file_path
            for file_path in source_path.rglob("*")
            if file_path.is_file() and file_path.suffix.lower() in {".jsonl", ".json", ".csv", ".tsv"}
        )
    return []


def _first_text_field(row: dict[str, Any], fields: Iterable[str]) -> tuple[str, str | None]:
    for field_name in fields:
        text = _text_from_obj(row.get(field_name)).strip()
        if text:
            return text, field_name
    return "", None


def _medical_qa_question(row: dict[str, Any]) -> tuple[str, str | None]:
    return _first_text_field(row, MEDICAL_QA_QUESTION_FIELDS)


def _answer_text(answer_obj: dict[str, Any]) -> tuple[str, str | None]:
    return _first_text_field(answer_obj, MEDICAL_QA_ANSWER_FIELDS)


def _medical_qa_answer_items(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    answers = row.get("answers") or row.get("model_answers") or row.get("responses")
    items: list[tuple[str, dict[str, Any]]] = []
    if isinstance(answers, dict):
        for source, value in answers.items():
            answer_obj = value if isinstance(value, dict) else {"answer": value}
            items.append((str(source), answer_obj))
    elif isinstance(answers, list):
        for idx, value in enumerate(answers):
            answer_obj = value if isinstance(value, dict) else {"answer": value}
            source = (
                answer_obj.get("model")
                or answer_obj.get("model_name")
                or answer_obj.get("source")
                or answer_obj.get("answer_source")
                or f"answer_{idx}"
            )
            items.append((str(source), answer_obj))
    elif any(key in row for key in MEDICAL_QA_ANSWER_FIELDS):
        source = (
            row.get("system")
            or row.get("model")
            or row.get("source")
            or row.get("answer_source")
            or row.get("team")
            or "submitted_answer"
        )
        items.append((str(source), row))
    return items


def _raw_numeric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _normalize_raw_1_to_4(value: Any) -> float | None:
    numeric = _raw_numeric(value)
    if numeric is None or numeric < 1.0 or numeric > 4.0:
        return None
    return validate_score_01((numeric - 1.0) / 3.0, "normalized_1_to_4_score")


def _normalize_official_0_to_3(value: Any) -> float | None:
    numeric = _raw_numeric(value)
    if numeric is None or numeric < 0.0 or numeric > 3.0:
        return None
    return validate_score_01(numeric / 3.0, "normalized_0_to_3_score")


def _score_from_fields(
    row: dict[str, Any],
    fields: Iterable[str],
    normalizer,
    score_type: str,
) -> tuple[float | None, dict[str, Any]]:
    for field_name in fields:
        if field_name not in row:
            continue
        score = normalizer(row[field_name])
        if score is not None:
            return score, {
                "score_source": field_name,
                "raw_score": row[field_name],
                "human_score_type": score_type,
            }
    return None, {"skip_reason": "no_usable_score"}


def normalize_mediqa_qa_2019_score(row: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    return _score_from_fields(
        row,
        MEDIQA_SCORE_FIELDS_1_TO_4,
        _normalize_raw_1_to_4,
        "medical_expert_rating_1_4",
    )


def normalize_liveqa_medical_2017_score(row: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    score, metadata = _score_from_fields(
        row,
        LIVEQA_OFFICIAL_SCORE_FIELDS_0_TO_3,
        _normalize_official_0_to_3,
        "official_liveqa_score_0_3",
    )
    if score is not None:
        return score, metadata
    return _score_from_fields(
        row,
        LIVEQA_GRADE_FIELDS_1_TO_4,
        _normalize_raw_1_to_4,
        "raw_liveqa_grade_1_4",
    )


def _medical_qa_question_id(row: dict[str, Any], question: str) -> str:
    return str(
        row.get("question_id")
        or row.get("qid")
        or row.get("question_number")
        or row.get("id")
        or row.get("example_id")
        or row.get("topic_id")
        or stable_text_hash(question)
    )


def _medical_qa_context(row: dict[str, Any]) -> str | None:
    return _compact_context(
        [
            ("category", row.get("category")),
            ("topic", row.get("topic")),
            ("description", row.get("description")),
            ("reference_answer", row.get("reference_answer")),
            ("metadata", row.get("metadata")),
        ]
    )


def canonicalize_medical_qa_sources(
    path: str | Path,
    *,
    dataset_id: str,
    task_type: str,
    score_normalizer,
) -> tuple[list[CalibrationExample], dict[str, Any]]:
    source_path = Path(path)
    source_files = _source_record_files(source_path)
    examples: list[CalibrationExample] = []
    skipped_reasons: Counter[str] = Counter()
    total_rows = 0
    total_answers_seen = 0

    for source_file in source_files:
        for row in _read_records_from_file(source_file):
            total_rows += 1
            question, question_field = _medical_qa_question(row)
            if not question:
                skipped_reasons["missing_question"] += 1
                continue
            question_id = _medical_qa_question_id(row, question)
            context = _medical_qa_context(row)
            split_group = make_split_group(dataset_id, question_id, "medical_answer_quality")

            for answer_index, (answer_source, answer_obj) in enumerate(_medical_qa_answer_items(row)):
                total_answers_seen += 1
                answer, answer_field = _answer_text(answer_obj)
                if not answer:
                    skipped_reasons["empty_answer"] += 1
                    continue
                score_row = {**row, **answer_obj}
                score, score_metadata = score_normalizer(score_row)
                if score is None:
                    skipped_reasons[score_metadata.get("skip_reason", "no_usable_score")] += 1
                    continue
                examples.append(
                    CalibrationExample(
                        calib_example_id=_example_id(
                            dataset_id,
                            question_id,
                            answer_source,
                            answer,
                            "medical_answer_quality",
                        ),
                        dataset_id=dataset_id,
                        domain="medical",
                        task_type=task_type,
                        score_dimension="medical_answer_quality",
                        question_id=question_id,
                        question=question,
                        context=context,
                        answer=answer,
                        answer_source=answer_source,
                        human_score=score,
                        human_score_raw=score_metadata["raw_score"],
                        human_score_scale=score_metadata,
                        human_score_type=score_metadata["human_score_type"],
                        human_rubric=None,
                        source_file=str(source_file),
                        split_group=split_group,
                        metadata={
                            "source_question_id": question_id,
                            "answer_index": answer_index,
                            "question_field": question_field,
                            "answer_field": answer_field,
                            "score_source": score_metadata["score_source"],
                        },
                    )
                )

    summary = {
        "dataset_id": dataset_id,
        "source_path": str(source_path),
        "source_files": [str(path) for path in source_files],
        "total_rows": total_rows,
        "total_answers_seen": total_answers_seen,
        "included_examples": len(examples),
        "skipped_examples": sum(skipped_reasons.values()),
        "skipped_reasons": dict(skipped_reasons),
    }
    return examples, summary


def canonicalize_mediqa_qa_2019_source(
    path: str | Path,
) -> tuple[list[CalibrationExample], dict[str, Any]]:
    return canonicalize_medical_qa_sources(
        path,
        dataset_id="mediqa_qa_2019",
        task_type="medical_qa",
        score_normalizer=normalize_mediqa_qa_2019_score,
    )


def canonicalize_liveqa_medical_2017_source(
    path: str | Path,
) -> tuple[list[CalibrationExample], dict[str, Any]]:
    return canonicalize_medical_qa_sources(
        path,
        dataset_id="liveqa_medical_2017",
        task_type="medical_qa",
        score_normalizer=normalize_liveqa_medical_2017_score,
    )


def _as_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _first_existing_path(
    root: Path,
    explicit_path: str | Path | None,
    candidate_paths: Iterable[Path],
) -> tuple[Path | None, list[str]]:
    if explicit_path:
        path = _as_path(root, explicit_path)
        return (path, []) if path.exists() else (None, [str(path)])
    missing: list[str] = []
    for relative_path in candidate_paths:
        path = root / relative_path
        if path.exists():
            return path, []
        missing.append(str(path))
    return None, missing


def _parse_ayers_dimensions(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        dimensions = [item.strip() for item in value.split(",") if item.strip()]
    else:
        dimensions = [str(item).strip() for item in value if str(item).strip()]
    if not dimensions:
        raise ValueError("At least one Ayers score dimension is required")
    unknown = [item for item in dimensions if item not in AYERS_SOURCE_BY_DIMENSION]
    if unknown:
        raise ValueError(f"Unsupported Ayers score dimensions: {unknown}")
    return list(dict.fromkeys(dimensions))


def build_canonical_dataset(
    repo_root: str | Path = ".",
    include_ayers: bool = True,
    include_counselbench: bool = True,
    include_mediqa_qa_2019: bool = True,
    include_liveqa_medical_2017: bool = True,
    fail_on_missing: bool = False,
    ayers_dimensions: str | Iterable[str] = ("composite",),
    ayers_sources: Iterable[tuple[str, str, Path]] | None = None,
    counselbench_sources: Iterable[Path] = COUNSELBENCH_SOURCES,
    mediqa_qa_2019_path: str | Path | None = None,
    liveqa_medical_2017_path: str | Path | None = None,
    mediqa_qa_2019_sources: Iterable[Path] = MEDIQA_QA_2019_SOURCES,
    liveqa_medical_2017_sources: Iterable[Path] = LIVEQA_MEDICAL_2017_SOURCES,
) -> tuple[list[CalibrationExample], dict[str, Any]]:
    root = Path(repo_root)
    examples: list[CalibrationExample] = []
    duplicates_removed = 0
    missing_files: list[str] = []
    dataset_summaries: dict[str, Any] = {}
    skipped_by_dataset_id: dict[str, int] = {}
    skipped_reasons_by_dataset_id: dict[str, dict[str, int]] = {}

    if include_ayers:
        selected_sources = (
            list(ayers_sources)
            if ayers_sources is not None
            else [
                AYERS_SOURCE_BY_DIMENSION[dimension]
                for dimension in _parse_ayers_dimensions(ayers_dimensions)
            ]
        )
        for dataset_id, score_dimension, relative_path in selected_sources:
            path = root / relative_path
            if not path.exists():
                missing_files.append(str(path))
                continue
            examples.extend(canonicalize_ayers_file(path, score_dimension, dataset_id=dataset_id))

    if include_counselbench:
        existing_paths: list[Path] = []
        for relative_path in counselbench_sources:
            path = root / relative_path
            if path.exists():
                existing_paths.append(path)
            else:
                missing_files.append(str(path))
        if existing_paths:
            counsel_examples, removed = canonicalize_counselbench_files(existing_paths)
            examples.extend(counsel_examples)
            duplicates_removed += removed

    if include_mediqa_qa_2019:
        mediqa_source, mediqa_missing = _first_existing_path(
            root,
            mediqa_qa_2019_path,
            mediqa_qa_2019_sources,
        )
        if mediqa_source is None:
            missing_files.extend(mediqa_missing)
        else:
            mediqa_examples, mediqa_summary = canonicalize_mediqa_qa_2019_source(mediqa_source)
            examples.extend(mediqa_examples)
            dataset_summaries["mediqa_qa_2019"] = mediqa_summary
            skipped_by_dataset_id["mediqa_qa_2019"] = mediqa_summary["skipped_examples"]
            skipped_reasons_by_dataset_id["mediqa_qa_2019"] = mediqa_summary["skipped_reasons"]

    if include_liveqa_medical_2017:
        liveqa_source, liveqa_missing = _first_existing_path(
            root,
            liveqa_medical_2017_path,
            liveqa_medical_2017_sources,
        )
        if liveqa_source is None:
            missing_files.extend(liveqa_missing)
        else:
            liveqa_examples, liveqa_summary = canonicalize_liveqa_medical_2017_source(
                liveqa_source
            )
            examples.extend(liveqa_examples)
            dataset_summaries["liveqa_medical_2017"] = liveqa_summary
            skipped_by_dataset_id["liveqa_medical_2017"] = liveqa_summary["skipped_examples"]
            skipped_reasons_by_dataset_id["liveqa_medical_2017"] = liveqa_summary[
                "skipped_reasons"
            ]

    missing_files = list(dict.fromkeys(missing_files))
    if fail_on_missing and missing_files:
        raise FileNotFoundError(f"Missing calibration source files: {missing_files}")

    ids = [example.calib_example_id for example in examples]
    duplicate_ids = sorted([item for item, count in Counter(ids).items() if count > 1])
    if duplicate_ids:
        raise ValueError(f"Duplicate calib_example_id values: {duplicate_ids[:10]}")

    summary = summarize_examples(examples)
    summary["duplicates_removed"] = duplicates_removed
    summary["missing_files"] = missing_files
    summary["skipped_examples"] = sum(skipped_by_dataset_id.values())
    summary["skipped_by_dataset_id"] = skipped_by_dataset_id
    summary["skipped_reasons_by_dataset_id"] = skipped_reasons_by_dataset_id
    summary["dataset_summaries"] = dataset_summaries
    return examples, summary


def summarize_examples(examples: list[CalibrationExample]) -> dict[str, Any]:
    scores = [example.human_score for example in examples]
    return {
        "total_examples": len(examples),
        "by_dataset_id": dict(Counter(example.dataset_id for example in examples)),
        "by_score_dimension": dict(Counter(example.score_dimension for example in examples)),
        "by_answer_source": dict(Counter(example.answer_source for example in examples)),
        "human_score_min": min(scores) if scores else None,
        "human_score_max": max(scores) if scores else None,
    }


def write_canonical_outputs(
    examples: list[CalibrationExample],
    out: str | Path,
    summary_out: str | Path,
    summary: dict[str, Any] | None = None,
) -> None:
    write_jsonl(out, examples)
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary if summary is not None else summarize_examples(examples)
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
