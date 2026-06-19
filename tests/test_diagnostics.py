from __future__ import annotations

import numpy as np

from cce_data.estimators.diagnostics import (
    make_knn_rows,
    nearest_neighbor_support_diagnostics,
    required_topk_fraction,
    reward_distribution_summary,
)
from cce_data.estimators.real_runner import RealData, bootstrap_run
from cce_data.estimators.surface_features import (
    SURFACE_FEATURE_NAMES,
    extract_action_surface_features,
)


def test_reward_distribution_summary_reports_quantiles() -> None:
    summary = reward_distribution_summary(
        y_behavior=np.array([0.0, 0.5, 1.0]),
        y_target=np.array([0.25, 0.75]),
    )

    assert summary["n_behavior"] == 3
    assert summary["n_target"] == 2
    assert summary["behavior_mean"] == 0.5
    assert summary["target_mean"] == 0.5
    assert summary["behavior_q00"] == 0.0
    assert summary["behavior_q50"] == 0.5
    assert summary["behavior_q100"] == 1.0
    assert summary["target_q25"] == 0.375


def test_required_topk_fraction_uses_largest_matching_top_tail() -> None:
    result = required_topk_fraction(
        y_behavior=np.array([0.0, 0.5, 1.0]),
        target_mean=0.75,
    )

    assert result["achievable_by_topk"] is True
    assert result["required_topk_k"] == 2
    assert result["required_topk_fraction"] == 2 / 3
    assert result["required_topk_mean"] == 0.75


def test_required_topk_fraction_handles_full_distribution_and_unachievable_target() -> None:
    full = required_topk_fraction(np.array([0.0, 0.5, 1.0]), target_mean=0.4)
    impossible = required_topk_fraction(np.array([0.0, 0.5, 1.0]), target_mean=1.2)

    assert full["required_topk_k"] == 3
    assert full["required_topk_fraction"] == 1.0
    assert impossible["achievable_by_topk"] is False
    assert impossible["required_topk_k"] == 0
    assert impossible["required_topk_fraction"] == 0.0
    assert np.isnan(impossible["required_topk_mean"])


def test_nearest_neighbor_support_diagnostics_summarizes_local_rewards() -> None:
    data = _toy_real_data()

    summary = nearest_neighbor_support_diagnostics(data, k_values=(1, 2))

    assert summary["n_behavior"] == 3
    assert summary["max_k_effective"] == 2
    assert np.isclose(summary["knn1_mean_behavior_reward"], (0.2 + 0.9 + 0.5) / 3)
    assert summary["knn2_effective_k"] == 2
    assert "knn2_gap_to_agent_mean" in summary
    assert 0.0 <= summary["frac_agent_reward_above_knn2_max"] <= 1.0
    assert summary["nn_distance_min_mean"] >= 0.0
    assert summary["nn_distance_min_p90"] >= summary["nn_distance_min_mean"]


def test_make_knn_rows_returns_one_row_per_target_neighbor() -> None:
    data = _toy_real_data()
    records = [
        {
            "example_id": f"ex{i}",
            "source": "toy",
            "source_id": str(i),
            "inclusion_status": "included",
            "y_score": float(data.y_clinician[i]),
            "y_agent_score": float(data.y_agent[i]),
        }
        for i in range(data.n)
    ]

    rows = make_knn_rows(data, records, k=2)

    assert len(rows) == 6
    assert set(rows["neighbor_rank"]) == {1, 2}
    assert rows.iloc[0]["target_example_id"] == "ex0"
    assert rows.iloc[0]["neighbor_example_id"] == "ex0"
    assert "target_agent_minus_knn_mean" in rows.columns


def test_realdata_without_extra_features_matches_original_feature_layout() -> None:
    phi_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    phi_a_cl = np.array([[5.0, 6.0], [7.0, 8.0]])
    phi_a_ag = np.array([[2.0, 1.0], [1.0, 2.0]])
    data = RealData(
        phi_x=phi_x,
        phi_a_clinician=phi_a_cl,
        phi_a_agent=phi_a_ag,
        y_clinician=np.array([0.1, 0.2]),
        y_agent=np.array([0.3, 0.4]),
    )

    expected = np.concatenate([phi_x, phi_a_cl, phi_x * phi_a_cl], axis=1)
    assert np.allclose(data.features_at("clinician"), expected)


def test_realdata_with_extra_features_appends_after_interaction() -> None:
    phi_x = np.ones((3, 2))
    phi_a = np.full((3, 2), 2.0)
    extra_cl = np.arange(9, dtype=float).reshape(3, 3)
    extra_ag = extra_cl + 100
    data = RealData(
        phi_x=phi_x,
        phi_a_clinician=phi_a,
        phi_a_agent=phi_a,
        y_clinician=np.array([0.1, 0.2, 0.3]),
        y_agent=np.array([0.4, 0.5, 0.6]),
        extra_clinician=extra_cl,
        extra_agent=extra_ag,
    )

    features = data.features_at("clinician")
    assert features.shape == (3, 9)
    assert np.allclose(features[:, :6], np.concatenate([phi_x, phi_a, phi_x * phi_a], axis=1))
    assert np.allclose(features[:, 6:], extra_cl)


def test_surface_only_realdata_with_zero_dim_embeddings_uses_extra_features() -> None:
    extra_cl = np.array([[1.0, 2.0], [3.0, 4.0]])
    extra_ag = np.array([[5.0, 6.0], [7.0, 8.0]])
    zeros = np.zeros((2, 0))
    data = RealData(
        phi_x=zeros,
        phi_a_clinician=zeros,
        phi_a_agent=zeros,
        y_clinician=np.array([0.1, 0.2]),
        y_agent=np.array([0.3, 0.4]),
        extra_clinician=extra_cl,
        extra_agent=extra_ag,
    )

    assert data.features_at("clinician").shape == (2, 2)
    assert np.allclose(data.features_at("clinician"), extra_cl)
    assert np.allclose(data.features_at("agent"), extra_ag)


def test_extract_action_surface_features_is_finite_with_stable_names() -> None:
    texts = [
        "I'm sorry this is scary. Please see your doctor in 2 weeks.",
        "1. Check symptoms\n2. Go to urgent care if chest pain occurs!",
    ]

    features, names = extract_action_surface_features(texts)

    assert names == SURFACE_FEATURE_NAMES
    assert features.shape == (2, len(SURFACE_FEATURE_NAMES))
    assert np.all(np.isfinite(features))
    assert features[0, names.index("empathy_phrase_count")] >= 1
    assert features[1, names.index("numbered_list_count")] == 2


def test_bootstrap_run_resamples_extra_features() -> None:
    phi = np.ones((5, 2))
    extra = np.arange(10, dtype=float).reshape(5, 2)
    data = RealData(
        phi_x=phi,
        phi_a_clinician=phi,
        phi_a_agent=phi,
        y_clinician=np.linspace(0.1, 0.5, 5),
        y_agent=np.linspace(0.2, 0.6, 5),
        extra_clinician=extra,
        extra_agent=extra + 10,
    )
    seen_feature_dims = []

    def estimator(sub: RealData, seed: int = 0) -> dict:
        assert sub.extra_clinician is not None
        assert sub.extra_agent is not None
        assert len(sub.extra_clinician) == sub.n
        seen_feature_dims.append(sub.features_at("clinician").shape[1])
        return {"v_hat": float(sub.extra_clinician[:, 0].mean())}

    samples = bootstrap_run(estimator, data, n_boot=4, seed=0)

    assert len(samples) == 4
    assert seen_feature_dims == [8, 8, 8, 8]


def _toy_real_data() -> RealData:
    phi_x = np.zeros((3, 2), dtype=np.float32)
    phi_a_clinician = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=np.float32,
    )
    phi_a_agent = np.array(
        [
            [0.1, 0.0],
            [19.0, 0.0],
            [11.0, 0.0],
        ],
        dtype=np.float32,
    )
    return RealData(
        phi_x=phi_x,
        phi_a_clinician=phi_a_clinician,
        phi_a_agent=phi_a_agent,
        y_clinician=np.array([0.2, 0.5, 0.9]),
        y_agent=np.array([0.3, 0.8, 0.6]),
    )
