"""Support and positivity diagnostics for real-data OPE runs.

These helpers are deliberately estimator-neutral. They inspect the fitted
feature representation and the held-out diagnostic target rewards, but do not
train, tune, or select production hyperparameters from target rewards.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from cce_data.estimators.real_runner import RealData


REWARD_QUANTILES: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


def swap_behavior_target(data: "RealData") -> "RealData":
    """Return a diagnostic copy with behavior and target sides swapped.

    In ``RealData`` naming, ``clinician`` is the estimator behavior side and
    ``agent`` is the estimator target side. The policy-swap diagnostic treats
    the old agent policy as behavior and the old clinician policy as target.
    """
    from cce_data.estimators.real_runner import RealData

    return RealData(
        phi_x=data.phi_x,
        phi_a_clinician=data.phi_a_agent,
        phi_a_agent=data.phi_a_clinician,
        y_clinician=data.y_agent,
        y_agent=data.y_clinician,
        extra_clinician=data.extra_agent,
        extra_agent=data.extra_clinician,
    )


def target_progress(v_hat: float, v_behavior: float, v_target: float) -> float:
    """Fraction of the behavior-to-target gap reached by an estimate."""
    denominator = float(v_target) - float(v_behavior)
    if denominator == 0.0:
        return float("nan")
    return float((float(v_hat) - float(v_behavior)) / denominator)


def ci_miss_side(ci_low, ci_high, v_target) -> str:
    """Classify whether a confidence interval covers, undershoots, or overshoots."""
    try:
        low = float(ci_low)
        high = float(ci_high)
        target = float(v_target)
    except (TypeError, ValueError):
        return "no_ci"

    if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(target):
        return "no_ci"
    if low <= target <= high:
        return "covers"
    if high < target:
        return "below"
    if low > target:
        return "above"
    return "no_ci"


def reward_distribution_summary(y_behavior, y_target) -> dict:
    """Summarize behavior and target reward distributions.

    ``y_target`` is for diagnostic evaluation only. This function performs no
    fitting or tuning; it reports moments and quantiles needed to interpret
    support failures.
    """
    behavior = _as_finite_1d("y_behavior", y_behavior)
    target = _as_finite_1d("y_target", y_target)

    summary = {
        "n_behavior": int(behavior.size),
        "n_target": int(target.size),
    }
    summary.update(_reward_summary_for_prefix("behavior", behavior))
    summary.update(_reward_summary_for_prefix("target", target))
    return summary


def required_topk_fraction(y_behavior, target_mean) -> dict:
    """Compute how much top-tail behavior reward support can match target mean.

    Rewards are sorted from high to low. The returned ``required_topk_k`` is
    the largest top-k prefix whose average remains at least ``target_mean``.
    Smaller fractions indicate that the target mean lives in a thinner upper
    tail of the logged behavior reward distribution.
    """
    behavior = _as_finite_1d("y_behavior", y_behavior)
    target = float(target_mean)
    if not np.isfinite(target):
        raise ValueError("target_mean must be finite.")

    sorted_rewards = np.sort(behavior)[::-1]
    cumulative_mean = np.cumsum(sorted_rewards) / np.arange(1, sorted_rewards.size + 1)
    achievable = bool(target <= sorted_rewards[0])

    if target <= float(behavior.mean()):
        k = int(behavior.size)
    elif achievable:
        qualifying = np.flatnonzero(cumulative_mean >= target)
        k = int(qualifying[-1] + 1) if qualifying.size else 0
    else:
        k = 0

    topk_mean = float(cumulative_mean[k - 1]) if k > 0 else float("nan")
    return {
        "target_mean": target,
        "behavior_mean": float(behavior.mean()),
        "behavior_max": float(sorted_rewards[0]),
        "achievable_by_topk": achievable,
        "required_topk_k": k,
        "required_topk_fraction": float(k / behavior.size),
        "required_topk_mean": topk_mean,
    }


def nearest_neighbor_support_diagnostics(
    data: "RealData",
    k_values=(1, 3, 5, 10, 20),
    metric: str = "euclidean",
) -> dict:
    """Summarize target-action support from behavior nearest neighbors.

    The diagnostic compares each target-policy feature row against logged
    behavior feature rows, then reports local behavior reward levels and
    target-vs-local gaps for each requested k. Target rewards are used only in
    the post-hoc diagnostic fractions.
    """
    ks = _validated_k_values(k_values)
    distances, indices = _kneighbors(data, max(ks), metric=metric)
    y_behavior = _as_finite_1d("data.y_clinician", data.y_clinician)
    y_target = _as_finite_1d("data.y_agent", data.y_agent)
    target_mean = float(y_target.mean())

    min_distances = distances[:, 0]
    summary = {
        "metric": metric,
        "n_behavior": int(data.n),
        "n_target": int(data.n),
        "max_k_requested": int(max(ks)),
        "max_k_effective": int(indices.shape[1]),
        "nn_distance_min_mean": float(min_distances.mean()),
        "nn_distance_min_median": float(np.median(min_distances)),
        "nn_distance_min_p90": float(np.quantile(min_distances, 0.90)),
        "nn_distance_min_max": float(min_distances.max()),
    }

    for k in ks:
        effective_k = min(k, indices.shape[1])
        neighbor_rewards = y_behavior[indices[:, :effective_k]]
        neighbor_distances = distances[:, :effective_k]
        local_mean = neighbor_rewards.mean(axis=1)
        local_max = neighbor_rewards.max(axis=1)
        prefix = f"knn{k}"
        summary.update(
            {
                f"{prefix}_effective_k": int(effective_k),
                f"{prefix}_mean_behavior_reward": float(local_mean.mean()),
                f"{prefix}_median_behavior_reward": float(np.median(local_mean)),
                f"{prefix}_p10_behavior_reward": float(np.quantile(local_mean, 0.10)),
                f"{prefix}_p90_behavior_reward": float(np.quantile(local_mean, 0.90)),
                f"{prefix}_mean_max_behavior_reward": float(local_max.mean()),
                f"{prefix}_gap_to_agent_mean": float(target_mean - local_mean.mean()),
                f"frac_agent_reward_above_{prefix}_max": float(np.mean(y_target > local_max)),
                f"{prefix}_distance_mean": float(neighbor_distances.mean()),
                f"{prefix}_distance_p90": float(np.quantile(neighbor_distances, 0.90)),
            }
        )

    return summary


def make_knn_rows(data: "RealData", records: list[dict], k: int = 10):
    """Return one row per target example and nearest behavior neighbor."""
    import pandas as pd

    if k <= 0:
        raise ValueError("k must be positive.")

    records_for_data = _records_aligned_to_data(records, data.n)
    distances, indices = _kneighbors(data, k, metric="euclidean")
    y_behavior = _as_finite_1d("data.y_clinician", data.y_clinician)
    y_target = _as_finite_1d("data.y_agent", data.y_agent)
    effective_k = indices.shape[1]

    rows: list[dict] = []
    for target_idx in range(data.n):
        local_behavior_rewards = y_behavior[indices[target_idx]]
        local_mean = float(local_behavior_rewards.mean())
        local_max = float(local_behavior_rewards.max())
        target_record = records_for_data[target_idx]
        for rank in range(effective_k):
            behavior_idx = int(indices[target_idx, rank])
            behavior_record = records_for_data[behavior_idx]
            rows.append(
                {
                    "target_index": int(target_idx),
                    "neighbor_rank": int(rank + 1),
                    "neighbor_index": behavior_idx,
                    "distance": float(distances[target_idx, rank]),
                    "target_example_id": target_record.get("example_id", ""),
                    "neighbor_example_id": behavior_record.get("example_id", ""),
                    "target_source": target_record.get("source", ""),
                    "neighbor_source": behavior_record.get("source", ""),
                    "target_source_id": target_record.get("source_id", ""),
                    "neighbor_source_id": behavior_record.get("source_id", ""),
                    "target_agent_reward": float(y_target[target_idx]),
                    "target_logged_behavior_reward": float(y_behavior[target_idx]),
                    "neighbor_behavior_reward": float(y_behavior[behavior_idx]),
                    "knn_k_effective": int(effective_k),
                    "knn_mean_behavior_reward": local_mean,
                    "knn_max_behavior_reward": local_max,
                    "target_agent_minus_knn_mean": float(y_target[target_idx] - local_mean),
                    "target_agent_above_knn_max": bool(y_target[target_idx] > local_max),
                }
            )
    return pd.DataFrame(rows)


def _reward_summary_for_prefix(prefix: str, values: np.ndarray) -> dict:
    out = {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }
    for q in REWARD_QUANTILES:
        out[f"{prefix}_q{int(round(q * 100)):02d}"] = float(np.quantile(values, q))
    return out


def _as_finite_1d(name: str, values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validated_k_values(k_values) -> tuple[int, ...]:
    ks = tuple(sorted({int(k) for k in k_values}))
    if not ks:
        raise ValueError("k_values must be non-empty.")
    if ks[0] <= 0:
        raise ValueError("All k_values must be positive.")
    return ks


def _kneighbors(data: "RealData", k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if data.n <= 0:
        raise ValueError("data must contain at least one row.")
    effective_k = min(int(k), data.n)
    X_behavior = np.asarray(data.features_at("clinician"), dtype=float)
    X_target = np.asarray(data.features_at("agent"), dtype=float)
    if X_behavior.ndim != 2 or X_target.ndim != 2:
        raise ValueError("RealData features must be 2D arrays.")
    if not np.all(np.isfinite(X_behavior)) or not np.all(np.isfinite(X_target)):
        raise ValueError("RealData features must contain only finite values.")

    nn = NearestNeighbors(n_neighbors=effective_k, metric=metric)
    nn.fit(X_behavior)
    distances, indices = nn.kneighbors(X_target)
    return distances, indices


def _records_aligned_to_data(records: list[dict], n: int) -> list[dict]:
    included = [
        r
        for r in records
        if r.get("inclusion_status") == "included"
        and r.get("y_score") is not None
        and r.get("y_agent_score") is not None
    ]
    if len(included) == n:
        return included
    if len(records) == n:
        return records
    raise ValueError(
        "records must either match data.n directly or include the same filtered "
        "included/y_score/y_agent_score rows used to build RealData."
    )
