"""
Real-data runner for Phase 5 main experiment.

Wraps DM / MIPS / OffCEM estimators to take real (x_text, a_text, y) tuples
instead of toy integer indices. Plug-and-play with any text embedder:
sentence-BERT (local, default) or OpenAI text-embedding-3-small.

Pipeline:
    1. Embed x and a separately -> phi_x, phi_a
    2. Build features [phi_x ; phi_a ; phi_x * phi_a]  (interaction term)
    3. Hand features to the same estimator math used on the toy

Usage:
    from cce_data.estimators.real_runner import run_phase5_headline
    table = run_phase5_headline(
        records,         # list[dict] from JSONL
        embed_fn,        # callable: list[str] -> (n, d) ndarray
    )
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from cce_data.estimators.density_ratio import (
    density_ratio,
    effective_sample_size,
    fit_density_ratio_classifier,
)


@dataclass
class RealData:
    """Featurized real-data tuples ready for estimators."""
    phi_x: np.ndarray            # (n, d_x)
    phi_a_clinician: np.ndarray  # (n, d_a)
    phi_a_agent: np.ndarray      # (n, d_a)
    y_clinician: np.ndarray      # (n,)
    y_agent: np.ndarray          # (n,)
    n: int = field(init=False)

    def __post_init__(self) -> None:
        self.n = len(self.y_clinician)

    def features_at(self, which: str) -> np.ndarray:
        """which in {'clinician', 'agent'}: returns [phi_x ; phi_a ; phi_x*phi_a]."""
        if which == "clinician":
            phi_a = self.phi_a_clinician
        elif which == "agent":
            phi_a = self.phi_a_agent
        else:
            raise ValueError(which)
        return np.concatenate([self.phi_x, phi_a, self.phi_x * phi_a], axis=1)


def featurize_records(
    records: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> RealData:
    """Embed (x, a_clinician, a_agent) text fields once, build RealData."""
    included = [
        r for r in records
        if r.get("inclusion_status") == "included"
        and r.get("y_score") is not None
        and r.get("y_agent_score") is not None
    ]
    xs = [r["x_patient_context"] for r in included]
    a_cl = [r["a_clinician"] for r in included]
    a_ag = [r["a_agent"] for r in included]
    y_cl = np.array([r["y_score"] for r in included], dtype=float)
    y_ag = np.array([r["y_agent_score"] for r in included], dtype=float)

    print(f"  Embedding {len(xs)} contexts + {len(a_cl)} clinician + {len(a_ag)} agent actions ...")
    phi_x = embed_fn(xs)
    phi_a_cl = embed_fn(a_cl)
    phi_a_ag = embed_fn(a_ag)
    return RealData(phi_x, phi_a_cl, phi_a_ag, y_cl, y_ag)


def dm_real(data: RealData, seed: int = 0) -> dict:
    """V_DM = mean f_hat(x, phi(a_agent)), f_hat fit on logged clinician tuples."""
    X_train = data.features_at("clinician")
    X_eval = data.features_at("agent")
    f_hat = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )
    f_hat.fit(X_train, data.y_clinician)
    y_pred = f_hat.predict(X_eval)
    return {"v_hat": float(y_pred.mean()), "y_pred": y_pred}


def mips_real(
    data: RealData, seed: int = 0, clip: float = 20.0,
    C: float = 0.1, calibrate: bool = True,
) -> dict:
    """V_MIPS = mean w(phi) * y, w from classifier P(target | x, phi).

    High-dim text features (n << p) need strong regularization (small C)
    and Platt-scaled probabilities to avoid perfect-separation collapse
    of the density ratio. SNIPS (self-normalized) is robust to weight
    scale shifts and is preferred for the headline.
    """
    X_b = data.features_at("clinician")
    X_t = data.features_at("agent")
    clf = fit_density_ratio_classifier(X_t, X_b, seed=seed, C=C, calibrate=calibrate)
    w = density_ratio(clf, X_b, clip=clip)
    v_mips = float((w * data.y_clinician).mean())
    w_sum = w.sum()
    v_snips = float((w * data.y_clinician).sum() / w_sum) if w_sum > 0 else float("nan")
    ess = effective_sample_size(w)
    return {"v_hat": v_snips, "v_mips_unnorm": v_mips, "ess": ess, "weights": w}


def offcem_real(
    data: RealData, seed: int = 0, clip: float = 20.0,
    C: float = 0.1, calibrate: bool = True,
) -> dict:
    """V_OffCEM = mean f_hat(x, phi(a_agent)) + w * (y - f_hat(x, phi(a_b))).

    Uses the same regularization/calibration choices as mips_real to keep
    the density-ratio behavior consistent across estimators.
    """
    X_b = data.features_at("clinician")
    X_t = data.features_at("agent")

    f_hat = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )
    f_hat.fit(X_b, data.y_clinician)
    dm_term = float(f_hat.predict(X_t).mean())

    clf = fit_density_ratio_classifier(X_t, X_b, seed=seed, C=C, calibrate=calibrate)
    w = density_ratio(clf, X_b, clip=clip)
    residual = data.y_clinician - f_hat.predict(X_b)
    # Self-normalize the correction so it is in [-1, 1] scale, not weight scale.
    w_sum = w.sum()
    correction = float((w * residual).sum() / w_sum) if w_sum > 0 else 0.0

    return {
        "v_hat": dm_term + correction,
        "dm_term": dm_term,
        "correction": correction,
    }


def bootstrap_run(
    estimator: Callable[[RealData, int], dict],
    data: RealData,
    n_boot: int = 100,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Bootstrap on V_hat by resampling rows, refitting each time.

    Returns list of (V_hat_b, V_b_b) per bootstrap iteration where V_b_b
    is the bootstrap-resampled clinician mean. Keeping both lets us
    compute RMSE, direction agreement, and quantile-CI downstream.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for b in range(n_boot):
        idx = rng.integers(0, data.n, size=data.n)
        sub = RealData(
            phi_x=data.phi_x[idx],
            phi_a_clinician=data.phi_a_clinician[idx],
            phi_a_agent=data.phi_a_agent[idx],
            y_clinician=data.y_clinician[idx],
            y_agent=data.y_agent[idx],
        )
        v_hat = float(estimator(sub, seed=seed + b)["v_hat"])
        v_b = float(sub.y_clinician.mean())
        samples.append((v_hat, v_b))
    return samples


def metrics_from_bootstrap(
    samples: list[tuple[float, float]],
    v_true_agent: float,
    v_true_b: float,
) -> dict:
    """Compute RMSE, direction-agreement rate, and quantile 95% CI from
    bootstrap samples.

    - RMSE       = sqrt( mean (V_hat_b - V_true_agent)^2 )  (exact)
    - direction  = mean[ sign(V_hat_b - V_b_b) == sign(V_true_agent - V_true_b) ]
    - CI         = empirical [2.5%, 97.5%] quantiles of V_hat_b
    """
    v_hats = np.array([s[0] for s in samples])
    v_bs = np.array([s[1] for s in samples])
    true_sign = 1.0 if v_true_agent > v_true_b else -1.0
    return {
        "rmse": float(np.sqrt(np.mean((v_hats - v_true_agent) ** 2))),
        "direction_rate": float(np.mean(np.sign(v_hats - v_bs) == true_sign)),
        "ci_low": float(np.quantile(v_hats, 0.025)),
        "ci_high": float(np.quantile(v_hats, 0.975)),
        "v_hat_b": v_hats.tolist(),
        "v_b_b": v_bs.tolist(),
    }


def run_phase5_headline(
    records: list[dict],
    embed_fn: Callable[[list[str]], np.ndarray],
    n_boot: int = 100,
    seed: int = 0,
) -> dict:
    """Top-level Phase 5 main-experiment runner.

    Returns dict with V_true_b, V_true_agent, true_effect, and per-estimator
    {v_hat, bias, rmse, ci_low, ci_high, direction_correct}.
    """
    data = featurize_records(records, embed_fn)
    v_true_b = float(data.y_clinician.mean())
    v_true_agent = float(data.y_agent.mean())
    true_effect = v_true_agent - v_true_b

    print()
    print(f"Ground truth: V_b={v_true_b:.4f}  V_agent={v_true_agent:.4f}  "
          f"effect={true_effect:+.4f}")
    print(f"Running {n_boot} bootstrap iterations per estimator (may take a few min)...")
    print()

    results = {}
    for name, est in [("DM", dm_real), ("MIPS", mips_real), ("OffCEM", offcem_real)]:
        point = est(data, seed=seed)["v_hat"]
        bias = point - v_true_agent
        direction = (point - v_true_b > 0) == (true_effect > 0)
        entry = {
            "v_hat": point,
            "bias": bias,
            "rel_bias": bias / v_true_agent,
            "direction_correct": direction,
        }
        if n_boot > 0:
            samples = bootstrap_run(est, data, n_boot=n_boot, seed=seed)
            metrics = metrics_from_bootstrap(samples, v_true_agent, v_true_b)
            entry["rmse"] = metrics["rmse"]
            entry["direction_rate"] = metrics["direction_rate"]
            entry["ci_low"] = metrics["ci_low"]
            entry["ci_high"] = metrics["ci_high"]
            entry["ci_covers_truth"] = metrics["ci_low"] <= v_true_agent <= metrics["ci_high"]
            entry["bootstrap_v_hat"] = metrics["v_hat_b"]
            entry["bootstrap_v_b"] = metrics["v_b_b"]
        results[name] = entry

    return {
        "n": data.n,
        "v_true_b": v_true_b,
        "v_true_agent": v_true_agent,
        "true_effect": true_effect,
        "results": results,
    }


def format_headline_table(report: dict) -> str:
    """Pretty-print the Phase 5 headline table."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Phase 5 Headline Table (n={report['n']}, n_boot=100)")
    lines.append("=" * 80)
    lines.append(f"V_true(pi_b)     = {report['v_true_b']:.4f}")
    lines.append(f"V_true(pi_agent) = {report['v_true_agent']:.4f}")
    lines.append(f"True effect      = {report['true_effect']:+.4f}")
    lines.append("")
    has_ci = "ci_low" in next(iter(report["results"].values()))
    if has_ci:
        lines.append(
            f"{'Estimator':<10} {'V_hat':>8} {'Bias':>9} {'Rel Bias':>9} "
            f"{'RMSE':>7} {'95% CI':>21} {'Cov':>4} {'Dir%':>6}"
        )
    else:
        lines.append(f"{'Estimator':<10} {'V_hat':>8} {'Bias':>9} {'Rel Bias':>9} {'Dir':>5}")
    lines.append("-" * 88)
    for name, r in report["results"].items():
        dirn_point = "yes" if r["direction_correct"] else "no"
        if has_ci:
            ci = f"[{r['ci_low']:.3f},{r['ci_high']:.3f}]"
            cov = "yes" if r["ci_covers_truth"] else "no"
            lines.append(
                f"{name:<10} {r['v_hat']:>8.4f} {r['bias']:>+9.4f} {r['rel_bias']:>+9.2%} "
                f"{r['rmse']:>7.4f} {ci:>21} {cov:>4} {r['direction_rate']:>6.1%}"
            )
        else:
            lines.append(
                f"{name:<10} {r['v_hat']:>8.4f} {r['bias']:>+9.4f} {r['rel_bias']:>+9.2%} {dirn_point:>5}"
            )
    return "\n".join(lines)
