from __future__ import annotations

from pathlib import Path

import pandas as pd

from cce_data.schema import CanonicalExample, stable_example_id
from cce_data.text import ACTION_SECTION_HEADERS, normalize_header, normalize_space


MTS_FILES = {
    "train": "Main-Dataset/MTS-Dialog-TrainingSet.csv",
    "valid": "Main-Dataset/MTS-Dialog-ValidationSet.csv",
    "test1": "Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
    "test2": "Main-Dataset/MTS-Dialog-TestSet-2-MEDIQA-Sum-2023.csv",
}


def _source_root(raw_dir: Path) -> Path:
    direct = raw_dir / "mts_dialog"
    if direct.exists():
        return direct
    candidates = sorted(raw_dir.glob("MTS-Dialog-*"))
    if candidates:
        return candidates[0]
    return direct


def load_mts_dialog(raw_dir: Path) -> list[CanonicalExample]:
    root = _source_root(raw_dir)
    examples: list[CanonicalExample] = []

    for split, rel_path in MTS_FILES.items():
        path = root / rel_path
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for row_idx, row in frame.iterrows():
            source_id = str(row.get("ID", row_idx))
            header_raw = str(row.get("section_header", ""))
            header = normalize_header(header_raw)
            section_text = normalize_space(row.get("section_text", ""))
            dialogue = normalize_space(row.get("dialogue", ""))
            reasons: list[str] = []
            is_action_section = header in ACTION_SECTION_HEADERS
            if not is_action_section:
                reasons.append(f"non_action_section:{header or 'missing'}")
            if not dialogue:
                reasons.append("missing_context")
            if not section_text:
                reasons.append("missing_action")

            examples.append(
                CanonicalExample(
                    example_id=stable_example_id("mts_dialog", split, source_id, header or row_idx),
                    source="mts_dialog",
                    split=split,
                    source_id=source_id,
                    x_patient_context=dialogue,
                    a_clinician=section_text if is_action_section else "",
                    dialogue=dialogue,
                    note=section_text,
                    section_header=header,
                    section_text=section_text,
                    extraction_method="mts_action_section_filter",
                    inclusion_status="included" if is_action_section and dialogue and section_text else "excluded",
                    exclusion_reasons=reasons,
                    metadata={"row_index": int(row_idx), "raw_file": rel_path},
                )
            )
    return examples
