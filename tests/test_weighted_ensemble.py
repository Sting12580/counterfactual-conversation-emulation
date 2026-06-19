import numpy as np

from cce_data.judge_calibration.weighted_ensemble import (
    SimplexWeightedJudge,
    fit_best_single_judge,
    fit_ridge_stacking,
    select_simplex_model,
)


def test_simplex_model_recovers_convex_weights() -> None:
    rng = np.random.default_rng(0)
    S = rng.uniform(0, 1, size=(120, 3))
    true_weights = np.array([0.2, 0.5, 0.3])
    y = S @ true_weights
    model = SimplexWeightedJudge(fit_intercept=False, lambda_reg=0.0, max_iter=8000).fit(S, y)
    assert np.all(model.weights_ >= -1e-8)
    assert abs(float(model.weights_.sum()) - 1.0) < 1e-8
    assert np.max(np.abs(model.weights_ - true_weights)) < 0.08


def test_intercept_and_clipping_work() -> None:
    S = np.array([[1.0, 1.0], [0.9, 1.0], [1.0, 0.8]])
    y = np.array([1.0, 0.95, 0.9])
    model = SimplexWeightedJudge(fit_intercept=True, clip_predictions=True).fit(S, y)
    pred = model.predict(np.array([[2.0, 2.0]]))
    assert pred[0] <= 1.0


def test_best_single_judge_selects_correct_column() -> None:
    S_val = np.array([[0.0, 0.1], [1.0, 0.9]])
    y_val = np.array([0.1, 0.9])
    model = fit_best_single_judge(S_val, y_val, S_val, y_val, ["bad", "good"])
    assert model.judge_name == "good"


def test_ridge_baseline_returns_finite_predictions() -> None:
    S = np.array([[0.0, 0.2], [0.5, 0.6], [1.0, 0.9]])
    y = np.array([0.1, 0.55, 0.95])
    model = fit_ridge_stacking(S, y, lambda_reg=0.1)
    assert np.all(np.isfinite(model.predict(S)))


def test_hyperparameter_selector_returns_model_and_diagnostics() -> None:
    S = np.linspace(0, 1, 40).reshape(20, 2)
    y = S[:, 0] * 0.7 + S[:, 1] * 0.3
    model, diag = select_simplex_model(
        S[:12], y[:12], S[12:16], y[12:16], [0.0, 0.1], [True, False]
    )
    assert model.weights_ is not None
    assert "effective_num_judges" in diag
