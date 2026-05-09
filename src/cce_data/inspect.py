from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def inspect_dataset(path: Path = Path("data/processed/phase2_dataset.jsonl"), n: int = 3) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= n:
                break
            row = json.loads(line)
            rows.append(
                {
                    "example_id": row["example_id"],
                    "source": row["source"],
                    "split": row["split"],
                    "status": row["inclusion_status"],
                    "x_chars": len(row.get("x_patient_context") or ""),
                    "a_chars": len(row.get("a_clinician") or ""),
                    "reasons": row.get("exclusion_reasons") or [],
                }
            )
    frame = pd.DataFrame(rows)
    return frame.to_string(index=False)
