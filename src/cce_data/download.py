from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests
import yaml
from tqdm import tqdm


DEFAULT_CONFIG = Path("configs/sources.yaml")


def load_source_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["sources"]


def default_sources(config: dict[str, Any]) -> list[str]:
    return [name for name, meta in config.items() if meta.get("enabled_by_default")]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with dest.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))


def download_github_archive(source: str, meta: dict[str, Any], raw_dir: Path, force: bool) -> Path:
    target = raw_dir / source
    if target.exists() and not force:
        return target
    if target.exists():
        shutil.rmtree(target)

    branch = meta.get("branch", "main")
    repo = meta["repo"]
    url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    tmp_dir = raw_dir / "_downloads"
    archive_path = tmp_dir / f"{source}.zip"
    tmp_extract = raw_dir / f"_{source}_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    download_file(url, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(tmp_extract)

    children = [path for path in tmp_extract.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"Expected one top-level directory in {archive_path}, got {children}")
    shutil.move(str(children[0]), str(target))
    shutil.rmtree(tmp_extract)
    return target


def download_huggingface_dataset(source: str, meta: dict[str, Any], raw_dir: Path, force: bool) -> Path:
    target = raw_dir / source
    if target.exists() and not force:
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    repo = meta["repo"]
    if source == "meddialog":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Downloading meddialog requires optional dependency: "
                "pip install -e '.[hf]'"
            ) from exc
        dataset = load_dataset(repo, "en")
        for split, rows in dataset.items():
            split_dir = target / split
            split_dir.mkdir(parents=True, exist_ok=True)
            output = split_dir / "data.jsonl"
            with output.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"Downloading {source} requires optional dependency: pip install -e '.[hf]'"
        ) from exc
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(target))
    return target


def download_sources(
    sources: list[str] | None = None,
    raw_dir: Path = Path("data/raw"),
    config_path: Path = DEFAULT_CONFIG,
    force: bool = False,
) -> dict[str, str]:
    config = load_source_config(config_path)
    selected = sources or default_sources(config)
    raw_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, str] = {}
    for source in selected:
        if source not in config:
            raise ValueError(f"Unknown source {source}. Known sources: {sorted(config)}")
        meta = config[source]
        kind = meta["kind"]
        if kind == "github_archive":
            path = download_github_archive(source, meta, raw_dir, force)
        elif kind == "huggingface_dataset":
            path = download_huggingface_dataset(source, meta, raw_dir, force)
        else:
            raise ValueError(f"Unsupported source kind {kind!r} for {source}")
        results[source] = str(path)
    return results
