from __future__ import annotations

import html
import re


ACTION_SECTION_HEADERS = {
    "assessment",
    "diagnosis",
    "plan",
    "disposition",
    "edcourse",
    "emergency department course",
    "impression",
    "assessment and plan",
    "a/p",
}

CONTEXT_SECTION_HEADERS = {
    "cc",
    "chief complaint",
    "genhx",
    "history of present illness",
    "hpi",
    "pastmedicalhx",
    "past medical history",
    "pastsurgical",
    "past surgical history",
    "fam/sochx",
    "family history",
    "social history",
    "ros",
    "review of systems",
    "medications",
    "allergy",
    "allergies",
    "exam",
    "labs",
    "imaging",
}

SECTION_ALIASES = {
    "genhx": "history of present illness",
    "cc": "chief complaint",
    "physical examination": "exam",
    "vitals reviewed": "vitals",
    "fam/sochx": "family history/social history",
    "pastmedicalhx": "past medical history",
    "pastsurgical": "past surgical history",
    "ros": "review of systems",
    "edcourse": "emergency department course",
}

NOTE_HEADINGS = [
    "ASSESSMENT AND PLAN",
    "CHIEF COMPLAINT",
    "HISTORY OF PRESENT ILLNESS",
    "REVIEW OF SYSTEMS",
    "PHYSICAL EXAMINATION",
    "VITALS REVIEWED",
    "RESULTS",
    "ASSESSMENT",
    "IMPRESSION",
    "DIAGNOSIS",
    "PLAN",
    "DISPOSITION",
    "SUBJECTIVE",
    "OBJECTIVE",
    "HPI",
    "EXAM",
    "MEDICATIONS",
    "ALLERGIES",
]


def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    text = html.unescape(str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_header(header: str | None) -> str:
    value = normalize_space(header).lower()
    value = value.strip(":[]() ")
    return SECTION_ALIASES.get(value, value)


def strip_action_from_context(dialogue: str, action: str) -> str:
    """Best-effort leakage guard when a source repeats final action in the dialogue."""
    dialogue = normalize_space(dialogue)
    action = normalize_space(action)
    if not dialogue or not action:
        return dialogue
    if len(action) < 40:
        return dialogue
    return dialogue.replace(action, "[REMOVED_CLINICIAN_FINAL_ACTION]")


def parse_note_sections(note: str) -> dict[str, str]:
    """Extract common note sections from loosely formatted clinical notes.

    This is intentionally conservative. If no labeled section is found, callers should
    keep the note as raw provenance and mark action extraction as lower confidence.
    """
    note = normalize_space(note)
    if not note:
        return {}

    colon_pattern = (
        r"(?im)^\s*(assessment\s*(?:and\s*plan)?|a/p|impression|diagnosis|plan|"
        r"disposition|subjective|objective|history|hpi|exam|medications|allergies?)\s*:\s*"
    )
    heading_pattern = r"(?m)^\s*(" + "|".join(re.escape(h) for h in NOTE_HEADINGS) + r")\s*$"
    matches: list[re.Match[str]] = []
    matches.extend(re.finditer(colon_pattern, note))
    matches.extend(re.finditer(heading_pattern, note))
    if not matches:
        return {}
    matches = sorted(matches, key=lambda m: m.start())

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(note)
        header = normalize_header(match.group(1))
        body = normalize_space(note[start:end])
        if body:
            sections[header] = body
    return sections


def extract_action_from_note(note: str) -> tuple[str, str, list[str]]:
    sections = parse_note_sections(note)
    selected: list[str] = []
    for header in ("assessment and plan", "a/p", "impression", "assessment", "diagnosis", "plan"):
        if header in sections:
            selected.append(f"{header}: {sections[header]}")

    if selected:
        return normalize_space("\n\n".join(selected)), "section_labels", []

    note = normalize_space(note)
    imp_plan = re.search(r"(?is)\b(imp(?:ression)?\s*:.*)$", note)
    if imp_plan:
        return normalize_space(imp_plan.group(1)), "regex_imp_plan", []

    # Fallback: keep the whole note as an action candidate but flag it. This lets the
    # row remain auditable while downstream code can filter low-confidence examples.
    reasons = ["no_action_section_found"]
    return note, "full_note_fallback", reasons


def textgrid_intervals(textgrid: str) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    current: dict[str, str] = {}
    for line in textgrid.splitlines():
        line = line.strip()
        if line.startswith("xmin ="):
            current["xmin"] = line.split("=", 1)[1].strip()
        elif line.startswith("xmax ="):
            current["xmax"] = line.split("=", 1)[1].strip()
        elif line.startswith("text ="):
            raw = line.split("=", 1)[1].strip()
            if raw.startswith('"') and raw.endswith('"'):
                raw = raw[1:-1]
            text = normalize_space(raw)
            if text and "xmin" in current and "xmax" in current:
                intervals.append((float(current["xmin"]), float(current["xmax"]), text))
            current = {}
    return intervals


def merge_speaker_intervals(
    doctor_grid: str,
    patient_grid: str,
    doctor_label: str = "Doctor",
    patient_label: str = "Patient",
) -> str:
    intervals: list[tuple[float, str, str]] = []
    intervals.extend((start, doctor_label, text) for start, _, text in textgrid_intervals(doctor_grid))
    intervals.extend((start, patient_label, text) for start, _, text in textgrid_intervals(patient_grid))
    intervals.sort(key=lambda item: item[0])
    return "\n".join(f"{speaker}: {text}" for _, speaker, text in intervals)
