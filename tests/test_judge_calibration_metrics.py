import math

import pandas as pd

from cce_data.judge_calibration.metrics import group_metrics, regression_metrics


def test_metrics_match_simple_case() -> None:
    metrics = regression_metrics([0.0, 1.0], [0.0, 0.5])
    assert metrics["n"] == 2
    assert abs(metrics["mae"] - 0.25) < 1e-12
    assert abs(metrics["mean_bias"] + 0.25) < 1e-12


def test_constant_predictions_do_not_crash() -> None:
    metrics = regression_metrics([0.0, 1.0], [0.5, 0.5])
    assert math.isnan(metrics["pearson"])
    assert math.isnan(metrics["spearman"])


def test_group_metrics_expected_columns() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "y": [0.0, 1.0, 0.5], "pred": [0.1, 0.9, 0.6]})
    out = group_metrics(df, "y", "pred", ["g"])
    assert {"g", "n", "y_mean", "pred_mean", "bias", "rmse", "mae"}.issubset(out.columns)
    assert set(out["g"]) == {"a", "b"}
