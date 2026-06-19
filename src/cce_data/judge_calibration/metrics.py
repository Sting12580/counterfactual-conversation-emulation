from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    mask = np.isfinite(y) & np.isfinite(pred)
    return y[mask], pred[mask]


def _safe_corr(y: np.ndarray, pred: np.ndarray, method: str) -> float:
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return math.nan
    if method == "pearson":
        return float(np.corrcoef(y, pred)[0, 1])
    if method == "spearman":
        y_rank = pd.Series(y).rank(method="average").to_numpy()
        pred_rank = pd.Series(pred).rank(method="average").to_numpy()
        if np.std(y_rank) == 0 or np.std(pred_rank) == 0:
            return math.nan
        return float(np.corrcoef(y_rank, pred_rank)[0, 1])
    raise ValueError(f"Unsupported correlation method: {method}")


def _calibration_line(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    if len(y) < 2 or np.std(pred) == 0:
        return math.nan, math.nan
    design = np.c_[np.ones(len(pred)), pred]
    intercept, slope = np.linalg.pinv(design) @ y
    return float(slope), float(intercept)


def _ece_binned(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> float:
    if len(y) == 0:
        return math.nan
    clipped = np.clip(pred, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (clipped >= bins[idx]) & (clipped <= bins[idx + 1])
        else:
            mask = (clipped >= bins[idx]) & (clipped < bins[idx + 1])
        if np.any(mask):
            total += (np.sum(mask) / len(y)) * abs(float(np.mean(y[mask]) - np.mean(pred[mask])))
    return float(total)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    y, pred = _arrays(y_true, y_pred)
    if len(y) == 0:
        return {
            "n": 0,
            "rmse": math.nan,
            "mae": math.nan,
            "mean_bias": math.nan,
            "pearson": math.nan,
            "spearman": math.nan,
            "calibration_slope": math.nan,
            "calibration_intercept": math.nan,
            "ece_binned": math.nan,
        }
    errors = pred - y
    slope, intercept = _calibration_line(y, pred)
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "mean_bias": float(np.mean(errors)),
        "pearson": _safe_corr(y, pred, "pearson"),
        "spearman": _safe_corr(y, pred, "spearman"),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece_binned": _ece_binned(y, pred),
    }


def group_metrics(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = regression_metrics(group[y_col], group[pred_col])
        row = {column: value for column, value in zip(group_cols, keys)}
        row.update(
            {
                "n": metrics["n"],
                "y_mean": float(pd.to_numeric(group[y_col], errors="coerce").mean()),
                "pred_mean": float(pd.to_numeric(group[pred_col], errors="coerce").mean()),
                "bias": metrics["mean_bias"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_models_metrics(
    predictions_df: pd.DataFrame,
    y_col: str,
    model_pred_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for pred_col in model_pred_cols:
        metrics = regression_metrics(predictions_df[y_col], predictions_df[pred_col])
        rows.append({"model": pred_col, **metrics})
    return pd.DataFrame(rows)
