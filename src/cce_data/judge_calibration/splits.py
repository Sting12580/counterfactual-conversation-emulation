from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass
class SplitAssignment:
    name: str
    assignments: pd.Series
    held_out_column: str | None = None
    held_out_value: str | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"split": self.assignments})


def _validate_fracs(train_frac: float, val_frac: float, test_frac: float) -> None:
    if min(train_frac, val_frac, test_frac) < 0:
        raise ValueError("split fractions must be non-negative")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to 1")


def _split_groups(groups: list[str], train_frac: float, val_frac: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        raise ValueError("Cannot split an empty set of groups")
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    if train_frac > 0 and n_train == 0 and n >= 1:
        n_train = 1
    if val_frac > 0 and n_val == 0 and n - n_train >= 1:
        n_val = 1
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    labels: dict[str, str] = {}
    for idx, group in enumerate(shuffled):
        if idx < n_train:
            labels[group] = "train"
        elif idx < n_train + n_val:
            labels[group] = "val"
        else:
            labels[group] = "test"
    return labels


def grouped_random_split(
    df: pd.DataFrame,
    group_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> SplitAssignment:
    _validate_fracs(train_frac, val_frac, test_frac)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}")
    groups = sorted(df[group_col].astype(str).unique().tolist())
    group_labels = _split_groups(groups, train_frac, val_frac, seed)
    assignments = df[group_col].astype(str).map(group_labels)
    return SplitAssignment(name="grouped_random", assignments=assignments)


def leave_one_dataset_out_splits(
    df: pd.DataFrame,
    dataset_col: str = "dataset_id",
    group_col: str = "split_group",
    val_frac: float = 0.2,
    seed: int = 0,
) -> Iterator[SplitAssignment]:
    yield from leave_one_column_value_out_splits(
        df, column=dataset_col, group_col=group_col, val_frac=val_frac, seed=seed
    )


def leave_one_column_value_out_splits(
    df: pd.DataFrame,
    column: str,
    group_col: str,
    val_frac: float,
    seed: int,
) -> Iterator[SplitAssignment]:
    if column not in df.columns:
        raise ValueError(f"Missing column {column!r}")
    if group_col not in df.columns:
        raise ValueError(f"Missing group column {group_col!r}")
    for held_out in sorted(df[column].astype(str).unique().tolist()):
        assignments = pd.Series(index=df.index, dtype=object)
        test_mask = df[column].astype(str) == held_out
        assignments.loc[test_mask] = "test"
        remaining = df.loc[~test_mask]
        if remaining.empty:
            raise ValueError(f"No training rows remain after holding out {held_out!r}")
        groups = sorted(remaining[group_col].astype(str).unique().tolist())
        group_labels = _split_groups(groups, 1.0 - val_frac, val_frac, seed)
        assignments.loc[remaining.index] = remaining[group_col].astype(str).map(group_labels)
        yield SplitAssignment(
            name=f"leave_one_{column}_{held_out}",
            assignments=assignments,
            held_out_column=column,
            held_out_value=held_out,
        )


def write_split_manifest(split_assignments: SplitAssignment, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = split_assignments.assignments.value_counts(dropna=False).to_dict()
    payload = {
        "name": split_assignments.name,
        "held_out_column": split_assignments.held_out_column,
        "held_out_value": split_assignments.held_out_value,
        "counts": {str(key): int(value) for key, value in counts.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
