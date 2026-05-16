"""
Classifier-based density ratio estimation for MIPS (Saito & Joachims 2022).

Estimates w(x, phi) = p(phi | x, pi_target) / p(phi | x, pi_b)
without ever computing pi_b or pi_target in action space.

Training:
  - Positive samples: (x_i, phi_target_i) from target policy
  - Negative samples: (x_i, phi_b_i)     from behavior policy
  - Train binary classifier P(class=target | x, phi)
  - With balanced classes, w = P / (1 - P) by Bayes' rule.

Real-data note:
  pi_b is unknown (human clinician), but logged (x, phi(a_clinician)) IS a
  sample from p(phi | x, pi_b) by definition. So the "negative" pool comes
  for free from logged data. For "positive" we generate a_agent ~ pi_target
  for each logged x and embed.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_density_ratio_classifier(
    features_target: np.ndarray,
    features_b: np.ndarray,
    seed: int = 0,
    C: float = 100.0,
    calibrate: bool = False,
):
    """Train logistic P(class=target | x, phi).

    Inputs are already-featurized concatenations of (x_features, phi_features).
    C defaults to 100 (weak regularization) — appropriate for low-dim toy
    settings. For high-dim real text embeddings (n << p), use C <= 1.0 to
    avoid perfect-separation overfit and set calibrate=True for Platt scaling.

    Returns a fitted sklearn classifier exposing .predict_proba(...).
    """
    X = np.vstack([features_target, features_b])
    y = np.concatenate([
        np.ones(len(features_target), dtype=int),
        np.zeros(len(features_b), dtype=int),
    ])
    base = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    if calibrate:
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    else:
        clf = base
    clf.fit(X, y)
    return clf


def density_ratio(
    clf: LogisticRegression,
    features: np.ndarray,
    clip: float = 20.0,
) -> np.ndarray:
    """w(x, phi) = P / (1 - P), clipped to [0, clip] for stability."""
    p = clf.predict_proba(features)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    w = p / (1 - p)
    return np.clip(w, 0.0, clip)


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (sum w)^2 / sum w^2. High ESS means stable IS estimator.

    Returned as fraction of n in [0, 1]; ESS / n >= 0.30 is a common
    plan W7 sanity threshold.
    """
    w = np.asarray(weights, dtype=float)
    n = len(w)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    if s2 <= 0:
        return 0.0
    return float((s1 ** 2) / s2 / n)
