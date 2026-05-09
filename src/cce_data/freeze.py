from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _read_jsonl_summary(path: Path) -> dict[str, Any]:
    total = 0
    sources: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    first_example_id = None
    last_example_id = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total += 1
            sources[str(row.get("source", ""))] += 1
            splits[str(row.get("split", ""))] += 1
            statuses[str(row.get("inclusion_status", ""))] += 1
            example_id = row.get("example_id")
            if first_example_id is None:
                first_example_id = example_id
            last_example_id = example_id
    return {
        "total": total,
        "by_source": dict(sources),
        "by_split": dict(splits),
        "by_status": dict(statuses),
        "first_example_id": first_example_id,
        "last_example_id": last_example_id,
    }


def freeze_dataset(
    input_path: Path = Path("data/processed_included/phase2_dataset.jsonl"),
    output_path: Path = Path("dataset_freezes/phase2_v1.json"),
    label: str = "phase2_v1",
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {input_path}")

    manifest: dict[str, Any] | None = None
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    git_status = _git_output(["status", "--short"]) or ""
    freeze = {
        "label": label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(input_path),
        "dataset_sha256": sha256_file(input_path),
        "dataset_summary": _read_jsonl_summary(input_path),
        "source_manifest": manifest,
        "git": {
            "commit": _git_output(["rev-parse", "HEAD"]),
            "branch": _git_output(["branch", "--show-current"]),
            "remote": _git_output(["remote", "get-url", "origin"]),
            "dirty": bool(git_status),
            "status_short": git_status.splitlines(),
        },
        "repro_command": (
            "cce-download && "
            "cce-build --included-only --output-dir data/processed_included && "
            f"cce-freeze --input {input_path} --output {output_path}"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False), encoding="utf-8")
    return freeze
