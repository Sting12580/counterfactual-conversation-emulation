"""Unit tests for Phase 4 W6: DM estimator + KL-control.

Validates that DM recovers V_TRUE_TARGET on the synthetic toy within
the plan's pass criterion of |bias| < 0.05 at n >= 2000.
"""
from __future__ import annotations

import numpy as np

from cce_data.estimators.dm import (
    dm_estimate,
    dm_kl_estimate,
    featurize_batch,
    sample_agent_actions,
    train_outcome_model,
)
from cce_data.estimators.synthetic_toy import (
    V_TRUE_TARGET,
    sample_logged_data,
)


def test_dm_recovers_v_true_on_toy() -> None:
    """Plan W6 pass criterion: |V_DM - V_TRUE_TARGET| < 0.05 at n=2000."""
    data = sample_logged_data(2000)
    xs = [d[0] for d in data]
    f_hat = train_outcome_model(data)
    a_agents = sample_agent_actions(xs)
    v_dm, _ = dm_estimate(f_hat, xs, a_agents)
    bias = v_dm - V_TRUE_TARGET
    assert abs(bias) < 0.05, f"|bias|={abs(bias):.4f} exceeds 0.05; V_DM={v_dm:.4f} vs V*={V_TRUE_TARGET:.4f}"


def test_dm_kl_returns_finite_value() -> None:
    """KL-control variant is finite and in [0, 1] for valid inputs."""
    data = sample_logged_data(500)
    xs = [d[0] for d in data]
    a_b = [d[1] for d in data]
    ys = np.array([d[2] for d in data])
    f_hat = train_outcome_model(data)
    X_train = featurize_batch(xs, a_b)
    a_agents = sample_agent_actions(xs)
    v_kl = dm_kl_estimate(f_hat, X_train, xs, a_agents, ys.mean())
    assert np.isfinite(v_kl)
    assert 0.0 <= v_kl <= 1.0


def test_dm_bias_shrinks_with_sample_size() -> None:
    """Sanity: larger n -> smaller (or comparable) bias."""
    biases = []
    for n in [500, 2000]:
        data = sample_logged_data(n)
        xs = [d[0] for d in data]
        f_hat = train_outcome_model(data)
        a_agents = sample_agent_actions(xs)
        v_dm, _ = dm_estimate(f_hat, xs, a_agents)
        biases.append(abs(v_dm - V_TRUE_TARGET))
    # n=2000 bias should not be much larger than n=500 (allow some slack
    # since RNG state drifts between calls).
    assert biases[1] <= biases[0] + 0.03
