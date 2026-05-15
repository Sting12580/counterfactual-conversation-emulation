"""
OffCEM estimator (Saito 2023, ICML).

Phase 4 W8 deliverable. Doubly robust combination of DM and MIPS:

    V_hat_OffCEM = E_i [
        f_hat(x_i, phi(a_agent_i))
        + w(x_i, phi(a_clinician_i)) * ( y_i - f_hat(x_i, phi(a_clinician_i)) )
    ]

Intuition (Saito 2023):
  - The first term is a DM-style cluster-effect prediction at the agent's
    embedding. Reliable because phi is well-supported under both pi_b
    and pi_target (cluster-level positivity).
  - The second term is a MIPS-style residual correction. f_hat captures
    the part of reward mediated by phi; (y - f_hat) is the residual that
    embeddings did NOT capture. The classifier-estimated w corrects this
    residual for distribution shift.

Double robustness: if EITHER f_hat or w is correct, V_hat is unbiased.

Expected ordering on the toy (plan W8):
  V_OffCEM   bias smaller than MIPS, comparable to or better than DM,
             because OffCEM combines DM's bias-control with MIPS's
             NDE-violation correction.

Validated on the synthetic toy:
  PASS criterion: |V_OffCEM - V_TRUE_TARGET| < 0.10 at n >= 2000, AND
                  |V_OffCEM bias| < |V_MIPS bias|  (the headline claim).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from cce_data.estimators.density_ratio import (
    density_ratio,
    effective_sample_size,
    fit_density_ratio_classifier,
)
from cce_data.estimators.mips import featurize
from cce_data.estimators.synthetic_toy import (
    N_ACTIONS,
    PI_B,
    PI_TARGET,
    V_TRUE_TARGET,
    sample_logged_data,
)


@dataclass
class OffCEMResult:
    v_offcem: float
    dm_term: float        # mean f_hat(x, phi(a_agent))
    correction_term: float  # mean w * (y - f_hat at logged)
    ess_fraction: float
    n: int


def offcem_estimate(
    data: list[tuple[int, int, float]],
    seed: int = 0,
    clip: float = 20.0,
) -> OffCEMResult:
    """Run OffCEM on logged (x, a^b, y).

    Pipeline:
      1. Sample a_agent ~ pi_target(.|x_i) for each logged x_i.
      2. Featurize logged (behavior) and counterfactual (agent) tuples.
      3. Fit f_hat(x, phi) on (x_i, phi(a_b_i)) -> y_i.
      4. DM term:        mean_i f_hat(x_i, phi(a_agent_i))
      5. Density ratio:  classifier P(target | x, phi) -> w at logged.
      6. Correction:     mean_i w_i * (y_i - f_hat(x_i, phi(a_b_i)))
      7. V_OffCEM = DM_term + correction.
    """
    rng = np.random.default_rng(seed)
    xs = np.array([d[0] for d in data])
    a_b = np.array([d[1] for d in data])
    y = np.array([d[2] for d in data], dtype=float)

    a_agent = np.array([
        rng.choice(N_ACTIONS, p=PI_TARGET[x]) for x in xs
    ])

    feat_b = featurize(xs, a_b)
    feat_agent = featurize(xs, a_agent)

    # Outcome model: f_hat(x, phi(a))
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )
    model.fit(feat_b, y)

    # DM-style term at agent embeddings
    f_agent = model.predict(feat_agent)
    dm_term = float(f_agent.mean())

    # Density ratio (MIPS-style) at logged behavior samples
    clf = fit_density_ratio_classifier(feat_agent, feat_b, seed=seed)
    w = density_ratio(clf, feat_b, clip=clip)

    # MIPS-style residual correction
    f_at_logged = model.predict(feat_b)
    residual = y - f_at_logged
    correction = float((w * residual).mean())

    v_offcem = dm_term + correction
    ess = effective_sample_size(w)

    return OffCEMResult(
        v_offcem=v_offcem,
        dm_term=dm_term,
        correction_term=correction,
        ess_fraction=ess,
        n=len(data),
    )


def offcem_oracle_estimate(
    data: list[tuple[int, int, float]],
    seed: int = 0,
) -> OffCEMResult:
    """OffCEM with KNOWN pi_b, pi_target. Isolates DR math from
    classifier-estimation quality. Should recover V_TRUE_TARGET tightly."""
    rng = np.random.default_rng(seed)
    xs = np.array([d[0] for d in data])
    a_b = np.array([d[1] for d in data])
    y = np.array([d[2] for d in data], dtype=float)

    a_agent = np.array([
        rng.choice(N_ACTIONS, p=PI_TARGET[x]) for x in xs
    ])

    feat_b = featurize(xs, a_b)
    feat_agent = featurize(xs, a_agent)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )
    model.fit(feat_b, y)

    f_agent = model.predict(feat_agent)
    dm_term = float(f_agent.mean())

    w = np.array([
        PI_TARGET[x, a] / PI_B[x, a] for x, a in zip(xs, a_b)
    ])
    f_at_logged = model.predict(feat_b)
    correction = float((w * (y - f_at_logged)).mean())

    v_offcem = dm_term + correction
    ess = effective_sample_size(w)
    return OffCEMResult(
        v_offcem=v_offcem,
        dm_term=dm_term,
        correction_term=correction,
        ess_fraction=ess,
        n=len(data),
    )


def main() -> None:
    # Avoid circular dep at module import — local imports for the demo.
    from cce_data.estimators.dm import (
        dm_estimate,
        featurize_batch as dm_featurize_batch,
        sample_agent_actions,
        train_outcome_model,
    )
    from cce_data.estimators.mips import mips_estimate

    print("=" * 64)
    print("CCE OffCEM Estimator -- synthetic toy validation")
    print("=" * 64)
    print(f"  V_true(pi_target) = {V_TRUE_TARGET:.4f}")
    print()
    print("  Headline comparison (lower |bias| is better):")
    print("  DM   < OffCEM < MIPS  is the plan W8 prediction.")
    print()

    for n in [500, 2000, 10000]:
        data = sample_logged_data(n)

        f_hat = train_outcome_model(data)
        xs = [d[0] for d in data]
        a_agents = sample_agent_actions(xs)
        v_dm, _ = dm_estimate(f_hat, xs, a_agents)

        v_mips = mips_estimate(data, seed=0).v_mips
        offcem = offcem_estimate(data, seed=0)
        oracle = offcem_oracle_estimate(data, seed=0)

        b_dm = v_dm - V_TRUE_TARGET
        b_mips = v_mips - V_TRUE_TARGET
        b_off = offcem.v_offcem - V_TRUE_TARGET
        b_orc = oracle.v_offcem - V_TRUE_TARGET

        print(f"  n={n:>5}")
        print(f"    DM         : {v_dm:.4f}  bias = {b_dm:+.4f}")
        print(f"    MIPS       : {v_mips:.4f}  bias = {b_mips:+.4f}")
        print(f"    OffCEM     : {offcem.v_offcem:.4f}  bias = {b_off:+.4f}  "
              f"(DM={offcem.dm_term:.3f} + corr={offcem.correction_term:+.4f})")
        print(f"    OffCEM-ORC : {oracle.v_offcem:.4f}  bias = {b_orc:+.4f}  "
              f"(known pi_b, pi_target)")
        print()


if __name__ == "__main__":
    main()
