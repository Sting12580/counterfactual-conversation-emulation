"""
Direct Method (DM) + KL-control estimator for the CCE project.

Phase 4 W6 deliverable. Implements:
  1. Featurize (x, a) -> vector
  2. Train f_hat(x, a) ~ E[Y|x,a] on logged data
  3. Sample a_agent ~ pi_target(.|x_i)
  4. Naive DM:    V_DM    = mean_i f_hat(x_i, a_agent_i)
  5. KL-control:  k-NN OOD shrinkage toward base rate (Jaques 2019 style)

Validated on the synthetic toy in cce_data.estimators.synthetic_toy:
    PASS criterion: |V_DM - V_TRUE_TARGET| < 0.05 at n >= 2000.

Real-data adaptation:
  - Replace `featurize(x, a)` with text embedding:
        np.concatenate([emb(x_text), emb(a_text), emb(x_text)*emb(a_text)])
  - Everything else stays the same.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from cce_data.estimators.synthetic_toy import (
    ACTION_EMB,
    N_ACTIONS,
    N_CONTEXTS,
    PI_TARGET,
    RNG,
    V_TRUE_TARGET,
    sample_logged_data,
)


# ----------------------------- Step 1: featurize -----------------------------
def featurize(x: int, a: int) -> np.ndarray:
    """One-hot(x) || ACTION_EMB[a] -> (N_CONTEXTS + EMB_DIM)-dim vector.

    Real-data version will swap to:
        [emb(x_text), emb(a_text), emb(x_text) * emb(a_text)]
    """
    x_onehot = np.eye(N_CONTEXTS)[x]
    return np.concatenate([x_onehot, ACTION_EMB[a]])


def featurize_batch(xs, as_) -> np.ndarray:
    return np.stack([featurize(x, a) for x, a in zip(xs, as_)])


# ----------------------------- Step 2: train f_hat ---------------------------
def train_outcome_model(data) -> GradientBoostingRegressor:
    xs = [d[0] for d in data]
    as_ = [d[1] for d in data]
    ys = np.array([d[2] for d in data])
    X = featurize_batch(xs, as_)
    f_hat = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0
    )
    f_hat.fit(X, ys)
    return f_hat


# ----------------------------- Step 3: sample agent actions ------------------
def sample_agent_actions(xs):
    """For each x_i in logged data, sample a_agent ~ pi_target(.|x_i)."""
    return [int(RNG.choice(N_ACTIONS, p=PI_TARGET[x])) for x in xs]


# ----------------------------- Step 4: naive DM ------------------------------
def dm_estimate(f_hat, xs, a_agents) -> tuple[float, np.ndarray]:
    X_eval = featurize_batch(xs, a_agents)
    y_pred = f_hat.predict(X_eval)
    return float(y_pred.mean()), y_pred


# ----------------------------- Step 5: KL-control (k-NN shrinkage) ----------
def dm_kl_estimate(
    f_hat,
    X_train: np.ndarray,
    xs,
    a_agents,
    y_mean: float,
    k: int = 5,
    temp: float = 0.3,
) -> float:
    """OOD-aware DM: shrink predictions toward base rate when agent's (x,a)
    is far from training distribution.

        alpha_i = exp(-normalized_knn_dist_i / temp)
        v_i    = alpha_i * f_hat(x_i, a_agent_i) + (1 - alpha_i) * y_mean
        V_DM_KL = mean(v_i)
    """
    nn = NearestNeighbors(n_neighbors=k).fit(X_train)
    X_eval = featurize_batch(xs, a_agents)
    dists, _ = nn.kneighbors(X_eval)
    ood = dists.mean(axis=1)
    rng_d = ood.max() - ood.min()
    ood_norm = (ood - ood.min()) / (rng_d + 1e-9)
    alpha = np.exp(-ood_norm / temp)
    y_pred = f_hat.predict(X_eval)
    y_shrunk = alpha * y_pred + (1 - alpha) * y_mean
    return float(y_shrunk.mean())


# ----------------------------- Bootstrap CI ---------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ----------------------------- Run on toy -----------------------------------
def main() -> None:
    print("=" * 64)
    print("CCE DM Estimator -- synthetic toy validation")
    print("=" * 64)
    print(f"  V_true(pi_target) = {V_TRUE_TARGET:.4f}")
    print()

    for n in [500, 2000, 10000]:
        data = sample_logged_data(n)
        xs = [d[0] for d in data]
        a_b = [d[1] for d in data]
        ys = np.array([d[2] for d in data])

        f_hat = train_outcome_model(data)
        a_agents = sample_agent_actions(xs)

        v_dm, y_eval = dm_estimate(f_hat, xs, a_agents)

        X_train = featurize_batch(xs, a_b)
        v_dm_kl = dm_kl_estimate(f_hat, X_train, xs, a_agents, ys.mean())

        ci_low, ci_high = bootstrap_ci(y_eval)
        bias_dm = v_dm - V_TRUE_TARGET
        bias_klc = v_dm_kl - V_TRUE_TARGET
        verdict = "PASS" if abs(bias_dm) < 0.05 else "FAIL"

        print(
            f"  n={n:>5}  "
            f"V_DM    = {v_dm:.4f}  bias = {bias_dm:+.4f}  "
            f"95% CI=[{ci_low:.3f},{ci_high:.3f}]  {verdict}"
        )
        print(
            f"           "
            f"V_DM+KL = {v_dm_kl:.4f}  bias = {bias_klc:+.4f}"
        )

    print()
    print("Pass criterion (plan W6): |bias| < 0.05 at n >= 2000.")


if __name__ == "__main__":
    main()
