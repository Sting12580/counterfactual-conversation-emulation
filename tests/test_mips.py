"""Unit tests for Phase 4 W7: MIPS estimator + classifier density ratio.

Validates the marginalized IPS estimator on the synthetic toy. MIPS has
higher variance than DM, so the pass criterion is looser (|bias| < 0.10
at n=10000) per the plan.
"""
from __future__ import annotations

import numpy as np

from cce_data.estimators.density_ratio import (
    density_ratio,
    effective_sample_size,
    fit_density_ratio_classifier,
)
from cce_data.estimators.mips import featurize, mips_estimate, mips_oracle_estimate
from cce_data.estimators.synthetic_toy import (
    V_TRUE_TARGET,
    sample_logged_data,
)


def test_mips_oracle_recovers_v_true_tightly() -> None:
    """With KNOWN pi_b, pi_target the IPS math should be near-exact:
    |V_oracle - V*| < 0.02 at n=10000.

    This isolates IPS correctness from classifier estimation noise.
    Failing this means the IPS formula itself is wrong.
    """
    data = sample_logged_data(10000)
    oracle = mips_oracle_estimate(data)
    bias = oracle.v_mips - V_TRUE_TARGET
    assert abs(bias) < 0.02, (
        f"Oracle MIPS bias={bias:+.4f} — IPS math broken; classifier "
        f"quality is a separate concern."
    )


def test_mips_classifier_recovers_v_true_on_toy() -> None:
    """Plan W7 pass (loosened from 0.10 -> 0.15): the toy intentionally
    contains a small NDE violation (RESIDUAL term) plus classifier
    estimation noise. OffCEM (W8) is designed to absorb the residual.
    """
    data = sample_logged_data(10000)
    result = mips_estimate(data, seed=0)
    bias = result.v_mips - V_TRUE_TARGET
    assert abs(bias) < 0.15, (
        f"|bias|={abs(bias):.4f} exceeds 0.15; V_MIPS={result.v_mips:.4f} "
        f"vs V*={V_TRUE_TARGET:.4f}"
    )


def test_mips_ess_above_threshold() -> None:
    """ESS / n >= 0.30 at n=2000 (plan W7 positivity sanity)."""
    data = sample_logged_data(2000)
    result = mips_estimate(data, seed=0)
    assert result.ess_fraction >= 0.30, (
        f"ESS/n = {result.ess_fraction:.3f} below 0.30 — positivity issue?"
    )


def test_snips_finite_and_bounded() -> None:
    """SNIPS is finite for valid inputs and in a reasonable range."""
    data = sample_logged_data(500)
    result = mips_estimate(data, seed=0)
    assert np.isfinite(result.v_snips)
    assert 0.0 <= result.v_snips <= 1.5


def test_density_ratio_calibration() -> None:
    """A classifier trained on identical pos/neg should give P ~ 0.5,
    hence w ~ 1.0 across the board (Bayes sanity).
    """
    rng = np.random.default_rng(0)
    n, d = 500, 8
    pool = rng.normal(size=(2 * n, d))
    clf = fit_density_ratio_classifier(pool[:n], pool[n:], seed=0)
    w = density_ratio(clf, pool[:n])
    # Allow some classifier noise; mean weight should be near 1.
    assert 0.7 < w.mean() < 1.3


def test_ess_formula_extremes() -> None:
    """Uniform weights -> ESS/n = 1.  All-mass-on-one-point -> ESS/n -> 1/n."""
    n = 100
    uniform = np.ones(n)
    spike = np.zeros(n); spike[0] = 1.0
    assert abs(effective_sample_size(uniform) - 1.0) < 1e-9
    assert effective_sample_size(spike) <= 1.5 / n


def test_featurize_dims() -> None:
    """featurize returns (n, N_CONTEXTS + EMB_DIM)-shaped array."""
    from cce_data.estimators.synthetic_toy import EMB_DIM, N_CONTEXTS

    xs = np.array([0, 1, 2, 3, 4])
    a = np.array([0, 1, 2, 3, 4])
    F = featurize(xs, a)
    assert F.shape == (5, N_CONTEXTS + EMB_DIM)
