"""Unit tests for Phase 4 W8: OffCEM estimator (Saito 2023).

Validates the headline claim from the paper plan:
    |V_OffCEM bias| < |V_MIPS bias|   (OffCEM absorbs NDE violation)

and that OffCEM with KNOWN pi_b, pi_target is near-exact (isolates DR
math from classifier-estimation quality).
"""
from __future__ import annotations

import numpy as np

from cce_data.estimators.mips import mips_estimate
from cce_data.estimators.offcem import (
    offcem_estimate,
    offcem_oracle_estimate,
)
from cce_data.estimators.synthetic_toy import (
    V_TRUE_TARGET,
    sample_logged_data,
)


def test_offcem_oracle_recovers_v_true_tightly() -> None:
    """With known pi_b, pi_target, OffCEM is exact up to MC noise:
    |bias| < 0.03 at n=10000."""
    data = sample_logged_data(10000)
    result = offcem_oracle_estimate(data, seed=0)
    bias = result.v_offcem - V_TRUE_TARGET
    assert abs(bias) < 0.03, (
        f"Oracle OffCEM bias={bias:+.4f} — DR math is broken; classifier "
        f"quality is a separate concern."
    )


def test_offcem_recovers_v_true_on_toy() -> None:
    """Plan W8 pass: |V_OffCEM - V_TRUE_TARGET| < 0.10 at n=10000.

    OffCEM should beat MIPS on this toy because the toy has a small NDE
    violation (RESIDUAL term) that DM's outcome model absorbs.
    """
    data = sample_logged_data(10000)
    result = offcem_estimate(data, seed=0)
    bias = result.v_offcem - V_TRUE_TARGET
    assert abs(bias) < 0.10, (
        f"|bias|={abs(bias):.4f} exceeds 0.10; V_OffCEM={result.v_offcem:.4f} "
        f"vs V*={V_TRUE_TARGET:.4f}"
    )


def test_offcem_beats_mips_on_toy() -> None:
    """Headline plan claim: |bias(OffCEM)| < |bias(MIPS)| at n=10000.

    OffCEM's DM term gives it a much better cluster-effect estimate than
    MIPS's classifier-based ratio alone can achieve.
    """
    data = sample_logged_data(10000)
    v_mips = mips_estimate(data, seed=0).v_mips
    v_off = offcem_estimate(data, seed=0).v_offcem

    bias_mips = abs(v_mips - V_TRUE_TARGET)
    bias_off = abs(v_off - V_TRUE_TARGET)
    assert bias_off < bias_mips, (
        f"OffCEM should beat MIPS: |OffCEM bias|={bias_off:.4f} "
        f"vs |MIPS bias|={bias_mips:.4f}"
    )


def test_offcem_returns_finite() -> None:
    data = sample_logged_data(500)
    result = offcem_estimate(data, seed=0)
    assert np.isfinite(result.v_offcem)
    assert np.isfinite(result.dm_term)
    assert np.isfinite(result.correction_term)


def test_offcem_decomposition_sums() -> None:
    """Sanity: v_offcem == dm_term + correction_term."""
    data = sample_logged_data(500)
    result = offcem_estimate(data, seed=0)
    assert abs(result.v_offcem - (result.dm_term + result.correction_term)) < 1e-9
