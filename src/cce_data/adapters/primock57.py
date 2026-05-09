from __future__ import annotations

import json
import re
from pathlib import Path

from cce_data.schema import CanonicalExample, stable_example_id
from cce_data.text import extract_action_from_note, merge_speaker_intervals, normalize_space


def _source_root(raw_dir: Path) -> Path:
    direct = raw_dir / "primock57"
    if direct.exists():
        return direct
    candidates = sorted(raw_dir.glob("primock57-*"))
    if candidates:
        return candidates[0]
    return direct


def _consultation_key(path: Path) -> str:
    return path.stem


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_primock57(raw_dir: Path) -> list[CanonicalExample]:
    root = _source_root(raw_dir)
    notes_dir = root / "notes"
    transcripts_dir = root / "transcripts"
    examples: list[CanonicalExample] = []

    for note_path in sorted(notes_dir.glob("*.json")):
        if note_path.name == "README.md":
            continue
        note_json = json.loads(note_path.read_text(encoding="utf-8"))
        key = _consultation_key(note_path)
        doctor_grid = _read_text(transcripts_dir / f"{key}_doctor.TextGrid")
        patient_grid = _read_text(transcripts_dir / f"{key}_patient.TextGrid")
        dialogue = merge_speaker_intervals(doctor_grid, patient_grid) if doctor_grid and patient_grid else ""
        note = normalize_space(note_json.get("note", ""))
        action, method, reasons = extract_action_from_note(note)

        if not action and note:
            match = re.search(r"(?is)(imp\s*:.*)", note)
            if match:
                action = normalize_space(match.group(1))
                method = "primock_imp_plan_regex"

        if not dialogue:
            reasons.append("missing_transcript")
        if not action:
            reasons.append("missing_action")
        low_confidence = method == "full_note_fallback"
        if low_confidence:
            reasons.append("low_confidence_action_extraction")

        examples.append(
            CanonicalExample(
                example_id=stable_example_id("primock57", "all", key),
                source="primock57",
                split="all",
                source_id=key,
                x_patient_context=dialogue,
                a_clinician=action,
                dialogue=dialogue,
                note=note,
                extraction_method=f"primock57_{method}",
                inclusion_status="included" if dialogue and action and not low_confidence else "excluded",
                exclusion_reasons=reasons,
                metadata={
                    "day": note_json.get("day"),
                    "consultation": note_json.get("consultation"),
                    "presenting_complaint": note_json.get("presenting_complaint"),
                    "highlights": note_json.get("highlights", []),
                },
            )
        )
    return examples
