from __future__ import annotations

from pathlib import Path

from cce_data.schema import CanonicalExample


def load_examples(source: str, raw_dir: Path, include_dialog_only: bool = False) -> list[CanonicalExample]:
    if source == "aci_bench":
        from cce_data.adapters.aci_bench import load_aci_bench

        return load_aci_bench(raw_dir)
    if source == "mts_dialog":
        from cce_data.adapters.mts_dialog import load_mts_dialog

        return load_mts_dialog(raw_dir)
    if source == "primock57":
        from cce_data.adapters.primock57 import load_primock57

        return load_primock57(raw_dir)
    if source == "meddialog":
        from cce_data.adapters.meddialog import load_meddialog

        return load_meddialog(raw_dir, include_dialog_only=include_dialog_only)
    raise ValueError(f"Unsupported source: {source}")
