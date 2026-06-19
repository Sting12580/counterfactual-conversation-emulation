from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _require_nonempty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def validate_score_01(value: Any, field_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if math.isnan(score) or score < 0.0 or score > 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")
    return score


def normalize_score(value: Any, min_value: float, max_value: float) -> float:
    raw = float(value)
    low = float(min_value)
    high = float(max_value)
    if math.isnan(raw) or math.isnan(low) or math.isnan(high) or high <= low:
        raise ValueError("score scale must be finite with max_value > min_value")
    return validate_score_01((raw - low) / (high - low), "normalized_score")


def stable_text_hash(text: str) -> str:
    if text is None:
        text = ""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def make_split_group(dataset_id: str, question_id: str, score_dimension: str) -> str:
    return "::".join(
        [
            _require_nonempty_string(dataset_id, "dataset_id"),
            _require_nonempty_string(question_id, "question_id"),
            _require_nonempty_string(score_dimension, "score_dimension"),
        ]
    )


@dataclass
class CalibrationExample:
    calib_example_id: str
    dataset_id: str
    domain: str
    task_type: str
    score_dimension: str
    question_id: str
    question: str
    context: str | None
    answer: str
    answer_source: str
    human_score: float
    human_score_raw: object | None
    human_score_scale: dict | None
    human_score_type: str
    human_rubric: str | None
    source_file: str | None
    split_group: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in [
            "calib_example_id",
            "dataset_id",
            "domain",
            "task_type",
            "score_dimension",
            "question_id",
            "question",
            "answer",
            "answer_source",
            "human_score_type",
            "split_group",
        ]:
            _require_nonempty_string(str(getattr(self, field_name)), field_name)
        self.human_score = validate_score_01(self.human_score, "human_score")
        expected_split_group = make_split_group(
            self.dataset_id, self.question_id, self.score_dimension
        )
        if self.split_group != expected_split_group:
            raise ValueError(
                f"split_group must equal {expected_split_group!r}, got {self.split_group!r}"
            )
        if self.metadata is None:
            self.metadata = {}


@dataclass
class JudgeScoreRecord:
    calib_example_id: str
    judge_id: str
    score: float
    score_raw: object | None = None
    score_scale: dict | None = None
    rubric_id: str | None = None
    prompt_hash: str | None = None
    model_version: str | None = None
    created_at: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_string(self.calib_example_id, "calib_example_id")
        _require_nonempty_string(self.judge_id, "judge_id")
        self.score = validate_score_01(self.score, "score")
        if self.metadata is None:
            self.metadata = {}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dataclass_to_json_dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def calibration_example_from_dict(d: dict[str, Any]) -> CalibrationExample:
    data = dict(d)
    data.setdefault("context", None)
    data.setdefault("human_score_raw", data.get("human_score"))
    data.setdefault("human_score_scale", {"min": 0, "max": 1})
    data.setdefault("human_rubric", None)
    data.setdefault("source_file", None)
    data.setdefault("metadata", {})
    if not data.get("split_group"):
        data["split_group"] = make_split_group(
            data["dataset_id"], data["question_id"], data["score_dimension"]
        )
    return CalibrationExample(**data)


def judge_score_record_from_dict(d: dict[str, Any]) -> JudgeScoreRecord:
    data = dict(d)
    data.setdefault("score_raw", data.get("score"))
    data.setdefault("score_scale", {"min": 0, "max": 1})
    data.setdefault("rubric_id", None)
    data.setdefault("prompt_hash", None)
    data.setdefault("model_version", None)
    data.setdefault("created_at", None)
    data.setdefault("metadata", {})
    return JudgeScoreRecord(**data)


def dataclass_to_json_dict(obj: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Expected dataclass or dict, got {type(obj)!r}")
