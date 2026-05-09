from __future__ import annotations

import json
from pathlib import Path

from cce_data.schema import CanonicalExample, stable_example_id
from cce_data.text import normalize_space


def load_meddialog(raw_dir: Path, include_dialog_only: bool = False) -> list[CanonicalExample]:
    if not include_dialog_only:
        return []

    root = raw_dir / "meddialog"
    examples: list[CanonicalExample] = []
    for path in sorted(root.rglob("*.jsonl")):
        split = path.parent.name
        with path.open("r", encoding="utf-8") as handle:
            for row_idx, line in enumerate(handle):
                row = json.loads(line)
                turns = row.get("dialogue_turns") or row.get("turns") or []
                rendered = []
                for turn in turns:
                    speaker = turn.get("speaker", "")
                    utterance = normalize_space(turn.get("utterance", ""))
                    if utterance:
                        rendered.append(f"{speaker}: {utterance}")
                if len(rendered) < 2:
                    continue
                context = "\n".join(rendered[:-1])
                final_action = rendered[-1]
                source_id = str(row.get("dialogue_id", row_idx))
                examples.append(
                    CanonicalExample(
                        example_id=stable_example_id("meddialog", split, source_id),
                        source="meddialog",
                        split=split,
                        source_id=source_id,
                        x_patient_context=context,
                        a_clinician=final_action,
                        dialogue="\n".join(rendered),
                        note="",
                        extraction_method="dialog_only_final_doctor_turn",
                        inclusion_status="auxiliary_only",
                        exclusion_reasons=["no_clinician_note_or_rubric_source"],
                        metadata={"raw_file": str(path.relative_to(root))},
                    )
                )
    return examples
