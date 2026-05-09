from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_COLUMNS = [
    "example_id",
    "source",
    "split",
    "source_id",
    "x_patient_context",
    "a_clinician",
    "e_action_repr",
    "y_score",
    "y_rubric",
    "y_source",
    "dialogue",
    "note",
    "section_header",
    "section_text",
    "time_zero_policy",
    "extraction_method",
    "inclusion_status",
    "exclusion_reasons",
    "metadata",
]


@dataclass
class CanonicalExample:
    example_id: str
    source: str
    split: str
    source_id: str
    x_patient_context: str
    a_clinician: str
    e_action_repr: str | None = None
    y_score: float | None = None
    y_rubric: str | None = None
    y_source: str = "unscored"
    dialogue: str = ""
    note: str = ""
    section_header: str | None = None
    section_text: str | None = None
    time_zero_policy: str = (
        "X contains patient presentation/dialogue available before final clinician "
        "assessment and plan; extracted A_clinician must not be included in X."
    )
    extraction_method: str = ""
    inclusion_status: str = "included"
    exclusion_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        return {column: record.get(column) for column in CANONICAL_COLUMNS}


def stable_example_id(source: str, split: str, source_id: str, suffix: str | None = None) -> str:
    parts = [source, split, str(source_id)]
    if suffix:
        parts.append(str(suffix))
    return "::".join(part.replace(" ", "_") for part in parts if part is not None)
