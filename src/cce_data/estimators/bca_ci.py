"""
Bias-corrected and accelerated (BCa) bootstrap CI (Efron 1987).

Plain quantile bootstrap CI assumes the bootstrap distribution of V_hat
is approximately unbiased and symmetric around the true parameter. When
the underlying estimator has a known bias (e.g., MIPS extrapolation in
positivity-violated regions), the plain CI is centered on a biased
distribution and systematically misses truth.

BCa fixes this with two small adjustments:
  - bias-correction term z0  = Phi^{-1}( P(V_hat* < V_hat_point) )
                                = standard-normal Z that explains how
                                  often bootstrap values fall below the
                                  point estimate. z0 = 0 iff unbiased.
  - acceleration term     a   = Σ(V_jack_i - V_jack_mean)^3
                                / [6 (Σ(V_jack_i - V_jack_mean)^2)^{1.5}]
                                = scale-invariant skewness measure from
                                  jackknife resamples. a = 0 iff symmetric.

The CI percentiles get shifted from [alpha/2, 1-alpha/2] to:
  p_lo = Phi( z0 + (z0 + z_{alpha/2}) / (1 - a (z0 + z_{alpha/2})) )
  p_hi = Phi( z0 + (z0 + z_{1-alpha/2}) / (1 - a (z0 + z_{1-alpha/2})) )

Then read the empirical p_lo and p_hi quantiles of the bootstrap sample.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bca_ci(
    bootstrap_values: np.ndarray,
    point_estimate: float,
    jackknife_values: np.ndarray | None = None,
    alpha: float = 0.05,
) -> tuple[float, float, dict]:
    """Compute BCa CI from bootstrap and optional jackknife resamples.

    Parameters
    ----------
    bootstrap_values
        Array of B bootstrap replicates of V_hat.
    point_estimate
        Original V_hat from the full sample (anchor for bias correction).
    jackknife_values
        Optional array of n leave-one-out V_hat values; needed for the
        acceleration term `a`. If None, falls back to BC (a=0), which is
        the bias-corrected-only flavor — still better than plain quantile.
    alpha
        Significance level. Default 0.05 -> 95% CI.

    Returns
    -------
    (ci_low, ci_high, diagnostics) where diagnostics is {'z0', 'a', 'p_lo', 'p_hi'}.
    """
    bv = np.asarray(bootstrap_values, dtype=float)
    n_boot = len(bv)

    # --- bias-correction z0 ---
    # P(V_hat* < V_hat). Edge guard: avoid 0 and 1 which blow up Phi^-1.
    frac_below = (bv < point_estimate).sum() / n_boot
    frac_below = float(np.clip(frac_below, 1.0 / (2 * n_boot), 1 - 1.0 / (2 * n_boot)))
    z0 = norm.ppf(frac_below)

    # --- acceleration a ---
    if jackknife_values is not None:
        jv = np.asarray(jackknife_values, dtype=float)
        mean_j = jv.mean()
        diffs = mean_j - jv  # note: jackknife sign convention (Efron)
        num = (diffs ** 3).sum()
        den = 6.0 * ((diffs ** 2).sum() ** 1.5)
        a = float(num / den) if den > 0 else 0.0
    else:
        a = 0.0  # falls back to BC (bias-corrected only)

    # --- shifted percentiles ---
    z_a_lo = norm.ppf(alpha / 2)
    z_a_hi = norm.ppf(1 - alpha / 2)
    p_lo = norm.cdf(z0 + (z0 + z_a_lo) / (1 - a * (z0 + z_a_lo)))
    p_hi = norm.cdf(z0 + (z0 + z_a_hi) / (1 - a * (z0 + z_a_hi)))

    # --- read empirical quantiles ---
    ci_low = float(np.quantile(bv, p_lo))
    ci_high = float(np.quantile(bv, p_hi))

    return ci_low, ci_high, {
        "z0": float(z0),
        "a": a,
        "p_lo": float(p_lo),
        "p_hi": float(p_hi),
        "n_boot": n_boot,
    }
