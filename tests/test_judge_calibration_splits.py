import pandas as pd

from cce_data.judge_calibration.splits import grouped_random_split, leave_one_dataset_out_splits


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset_id": ["a"] * 6 + ["b"] * 6,
            "split_group": [f"a::q{i}::s" for i in range(6)] + [f"b::q{i}::s" for i in range(6)],
        }
    )


def test_grouped_split_has_no_group_leakage() -> None:
    df = _df()
    split = grouped_random_split(df, "split_group", 0.5, 0.25, 0.25, seed=1)
    for group, rows in df.assign(split=split.assignments).groupby("split_group"):
        del group
        assert rows["split"].nunique() == 1


def test_lodo_holds_out_one_dataset_at_a_time() -> None:
    df = _df()
    splits = list(leave_one_dataset_out_splits(df, val_frac=0.2, seed=2))
    assert len(splits) == 2
    for split in splits:
        assigned = df.assign(split=split.assignments)
        test_datasets = set(assigned[assigned["split"] == "test"]["dataset_id"])
        assert test_datasets == {split.held_out_value}
