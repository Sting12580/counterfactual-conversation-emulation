from __future__ import annotations

from pathlib import Path

import pandas as pd

from cce_data.schema import CanonicalExample, stable_example_id
from cce_data.text import extract_action_from_note, normalize_space, strip_action_from_context


ACI_FILES = {
    "train": "data/challenge_data/train.csv",
    "valid": "data/challenge_data/valid.csv",
    "test1": "data/challenge_data/clinicalnlp_taskB_test1.csv",
    "test2": "data/challenge_data/clinicalnlp_taskC_test2.csv",
    "test3": "data/challenge_data/clef_taskC_test3.csv",
}


def _source_root(raw_dir: Path) -> Path:
    direct = raw_dir / "aci_bench"
    if direct.exists():
        return direct
    candidates = sorted(raw_dir.glob("aci-bench-*"))
    if candidates:
        return candidates[0]
    return direct


def load_aci_bench(raw_dir: Path) -> list[CanonicalExample]:
    root = _source_root(raw_dir)
    examples: list[CanonicalExample] = []

    for split, rel_path in ACI_FILES.items():
        path = root / rel_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row_idx, row in frame.iterrows():
            encounter_id = str(row.get("encounter_id", row_idx))
            dialogue = normalize_space(row.get("dialogue", ""))
            note = normalize_space(row.get("note", ""))
            action, method, reasons = extract_action_from_note(note)
            context = strip_action_from_context(dialogue, action)
            low_confidence = method == "full_note_fallback"
            if low_confidence:
                reasons.append("low_confidence_action_extraction")
            inclusion_status = "included" if action and context and not low_confidence else "excluded"
            if not context:
                reasons.append("missing_context")
            if not action:
                reasons.append("missing_action")
            examples.append(
                CanonicalExample(
                    example_id=stable_example_id("aci_bench", split, encounter_id),
                    source="aci_bench",
                    split=split,
                    source_id=encounter_id,
                    x_patient_context=context,
                    a_clinician=action,
                    dialogue=dialogue,
                    note=note,
                    extraction_method=f"aci_note_{method}",
                    inclusion_status=inclusion_status,
                    exclusion_reasons=reasons,
                    metadata={
                        "dataset": row.get("dataset"),
                        "row_index": int(row_idx),
                        "raw_file": rel_path,
                    },
                )
            )
    return examples
