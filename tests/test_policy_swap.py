from __future__ import annotations

import numpy as np

from cce_data.estimators.diagnostics import (
    ci_miss_side,
    swap_behavior_target,
    target_progress,
)
from cce_data.estimators.real_runner import RealData


def test_swap_behavior_target_swaps_rewards() -> None:
    data = _toy_data()

    swapped = swap_behavior_target(data)

    assert np.allclose(swapped.y_clinician, data.y_agent)
    assert np.allclose(swapped.y_agent, data.y_clinician)


def test_swap_behavior_target_swaps_action_embeddings() -> None:
    data = _toy_data()

    swapped = swap_behavior_target(data)

    assert np.allclose(swapped.phi_a_clinician, data.phi_a_agent)
    assert np.allclose(swapped.phi_a_agent, data.phi_a_clinician)


def test_swap_behavior_target_preserves_context_embeddings() -> None:
    data = _toy_data()

    swapped = swap_behavior_target(data)

    assert np.allclose(swapped.phi_x, data.phi_x)


def test_swap_behavior_target_swaps_extra_features() -> None:
    data = _toy_data()

    swapped = swap_behavior_target(data)

    assert swapped.extra_clinician is not None
    assert swapped.extra_agent is not None
    assert np.allclose(swapped.extra_clinician, data.extra_agent)
    assert np.allclose(swapped.extra_agent, data.extra_clinician)


def test_target_progress_positive_treatment_effect() -> None:
    assert np.isclose(target_progress(v_hat=0.7, v_behavior=0.4, v_target=0.8), 0.75)


def test_target_progress_negative_treatment_effect() -> None:
    assert np.isclose(target_progress(v_hat=0.7, v_behavior=0.8, v_target=0.6), 0.5)


def test_target_progress_returns_nan_for_zero_treatment_effect() -> None:
    assert np.isnan(target_progress(v_hat=0.5, v_behavior=0.4, v_target=0.4))


def test_ci_miss_side_covers() -> None:
    assert ci_miss_side(0.2, 0.8, 0.5) == "covers"


def test_ci_miss_side_below() -> None:
    assert ci_miss_side(0.2, 0.4, 0.5) == "below"


def test_ci_miss_side_above() -> None:
    assert ci_miss_side(0.6, 0.8, 0.5) == "above"


def test_ci_miss_side_no_ci() -> None:
    assert ci_miss_side(None, 0.8, 0.5) == "no_ci"
    assert ci_miss_side(0.2, np.nan, 0.5) == "no_ci"


def _toy_data() -> RealData:
    return RealData(
        phi_x=np.array([[1.0, 0.0], [0.0, 1.0]]),
        phi_a_clinician=np.array([[0.1, 0.2], [0.3, 0.4]]),
        phi_a_agent=np.array([[0.5, 0.6], [0.7, 0.8]]),
        y_clinician=np.array([0.2, 0.4]),
        y_agent=np.array([0.6, 0.8]),
        extra_clinician=np.array([[1.0, 2.0], [3.0, 4.0]]),
        extra_agent=np.array([[5.0, 6.0], [7.0, 8.0]]),
    )
