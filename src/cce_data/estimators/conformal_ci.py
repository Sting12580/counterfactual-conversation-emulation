"""Split-conformal and jackknife+ CIs for OPE point estimates.

Given a point estimator V_hat that produces per-sample influence scores
psi_i, the split-conformal procedure guarantees finite-sample valid
intervals under iid:

    1. Split data into train (1 - frac_calib) and calibration (frac_calib).
    2. Fit nuisance + estimator on train; point := mean(psi_train).
    3. Compute residual scores R_i = |psi_calib_i - point| on calibration.
    4. (1-alpha)(n+1)/n empirical quantile of R is the CI half-width.

For OPE, the natural psi_i is the per-sample doubly-robust score:
    psi_i = q_hat(x_i, a^agent) + w_i (y_i - q_hat(x_i, a_clinician))

For MIPS-style estimators without an outcome model, use
    psi_i = w_i * y_i.

Mean(psi_i) is the corresponding V_hat. Bootstrap CI on psi gives an
asymptotic CI; conformal gives a finite-sample CI without normality or
unbiasedness assumptions — width auto-widens when the estimator is
biased, so coverage is preserved at the cost of interval width.

Adapted from /Users/niuniu/Counterfactual-Conversation-Emulation-for-Medical-AI-Agents
(Taufiq et al. 2022 split-conformal OPE).
"""
from __future__ import annotations

import numpy as np


def split_conformal_ci(
    psi_train: np.ndarray,
    psi_calib: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Returns (point, lower, upper) for split-conformal interval.

    Both psi_train and psi_calib are arrays of per-sample influence scores
    (same estimator, just different sample halves). The point estimate is
    mean(psi_train); the half-width is the (1-alpha)(n+1)/n quantile of
    |psi_calib - mean(psi_train)|.

    Default alpha=0.05 -> 95% CI to match the bootstrap convention used
    elsewhere in this package.
    """
    if len(psi_train) == 0 or len(psi_calib) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(psi_train.mean())
    residuals = np.abs(psi_calib - point)
    n = len(residuals)
    q_level = min(1.0, (1 - alpha) * (n + 1) / n)
    half_width = float(np.quantile(residuals, q_level))
    return point, point - half_width, point + half_width


def jackknife_plus_ci(
    psi: np.ndarray, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Jackknife+ CI on a 1d influence array (no train/calib split).

    Useful for small n where split-conformal wastes data. Each i gets a
    leave-one-out 'predicted' mean; CI half-width is the (1-alpha)(n+1)/n
    quantile of |psi_i - lo_mean_i|.
    """
    n = len(psi)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    point = float(psi.mean())
    sums = psi.sum()
    lo_means = (sums - psi) / (n - 1)
    residuals = np.abs(psi - lo_means)
    q_level = min(1.0, (1 - alpha) * (n + 1) / n)
    half_width = float(np.quantile(residuals, q_level))
    return point, point - half_width, point + half_width


def coverage(
    values_true: np.ndarray, lowers: np.ndarray, uppers: np.ndarray,
) -> float:
    """Empirical coverage of CIs across a simulation."""
    return float(((lowers <= values_true) & (values_true <= uppers)).mean())
