from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_2d_array(S: Any) -> np.ndarray:
    array = np.asarray(S, dtype=float)
    if array.ndim != 2:
        raise ValueError("S must be a 2D array")
    if not np.all(np.isfinite(array)):
        raise ValueError("S must contain only finite values")
    return array


def _as_1d_array(y: Any, expected_n: int | None = None) -> np.ndarray:
    array = np.asarray(y, dtype=float).reshape(-1)
    if expected_n is not None and len(array) != expected_n:
        raise ValueError(f"Expected {expected_n} labels, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError("y must contain only finite values")
    return array


def _sample_weight_array(sample_weight: Any, n: int) -> np.ndarray:
    if sample_weight is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(weights) != n:
            raise ValueError(f"Expected {n} sample weights, got {len(weights)}")
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample weights must be finite and non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("sample weights must sum to a positive value")
    return weights / float(weights.sum())


def _project_simplex(values: np.ndarray) -> np.ndarray:
    """Project values onto the probability simplex."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError("simplex projection expects a 1D vector")
    if len(vector) == 0:
        raise ValueError("simplex projection needs at least one value")
    sorted_values = np.sort(vector)[::-1]
    cssv = np.cumsum(sorted_values)
    rho = np.nonzero(sorted_values * np.arange(1, len(vector) + 1) > (cssv - 1.0))[0]
    if len(rho) == 0:
        return np.ones_like(vector) / len(vector)
    rho_idx = rho[-1]
    theta = (cssv[rho_idx] - 1.0) / (rho_idx + 1)
    return np.maximum(vector - theta, 0.0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


class SimplexWeightedJudge:
    def __init__(
        self,
        fit_intercept: bool = True,
        lambda_reg: float = 0.0,
        clip_predictions: bool = True,
        max_iter: int = 2000,
        tol: float = 1e-10,
        random_state: int | None = None,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.lambda_reg = float(lambda_reg)
        self.clip_predictions = clip_predictions
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.judge_names_: list[str] | None = None
        self.objective_: float | None = None

    def fit(
        self,
        S: Any,
        y: Any,
        sample_weight: Any = None,
        judge_names: list[str] | None = None,
    ) -> "SimplexWeightedJudge":
        scores = _as_2d_array(S)
        labels = _as_1d_array(y, expected_n=scores.shape[0])
        weights = _sample_weight_array(sample_weight, scores.shape[0])
        n_judges = scores.shape[1]
        if n_judges == 0:
            raise ValueError("At least one judge column is required")
        if judge_names is not None and len(judge_names) != n_judges:
            raise ValueError("judge_names length must match S columns")
        uniform = np.ones(n_judges, dtype=float) / n_judges

        scipy_result = self._fit_slsqp(scores, labels, weights, uniform)
        if scipy_result is None:
            fitted_weights, intercept, objective = self._fit_projected_gradient(
                scores, labels, weights, uniform
            )
        else:
            fitted_weights, intercept, objective = scipy_result

        self.weights_ = _project_simplex(fitted_weights)
        self.intercept_ = float(intercept if self.fit_intercept else 0.0)
        self.judge_names_ = judge_names or [f"judge_{idx}" for idx in range(n_judges)]
        self.objective_ = float(objective)
        return self

    def _objective(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
        uniform: np.ndarray,
        weights: np.ndarray,
        intercept: float,
    ) -> float:
        residual = labels - intercept - scores @ weights
        return float(np.sum(sample_weight * residual**2) + self.lambda_reg * np.sum((weights - uniform) ** 2))

    def _fit_slsqp(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
        uniform: np.ndarray,
    ) -> tuple[np.ndarray, float, float] | None:
        try:
            from scipy.optimize import minimize
        except Exception:
            return None

        n_judges = scores.shape[1]

        def unpack(params: np.ndarray) -> tuple[np.ndarray, float]:
            weights = params[:n_judges]
            intercept = float(params[-1]) if self.fit_intercept else 0.0
            return weights, intercept

        def objective(params: np.ndarray) -> float:
            weights, intercept = unpack(params)
            return self._objective(scores, labels, sample_weight, uniform, weights, intercept)

        x0 = np.r_[uniform, np.average(labels - scores @ uniform, weights=sample_weight)]
        if not self.fit_intercept:
            x0 = uniform.copy()

        constraints = [{"type": "eq", "fun": lambda params: float(np.sum(params[:n_judges]) - 1.0)}]
        bounds = [(0.0, 1.0)] * n_judges
        if self.fit_intercept:
            bounds.append((None, None))

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return None
        fitted_weights, intercept = unpack(np.asarray(result.x, dtype=float))
        return fitted_weights, intercept, float(result.fun)

    def _fit_projected_gradient(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        sample_weight: np.ndarray,
        uniform: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        n_judges = scores.shape[1]
        weights = uniform.copy()
        intercept = float(np.average(labels - scores @ weights, weights=sample_weight)) if self.fit_intercept else 0.0
        weighted_scores = scores * sample_weight[:, None]
        lipschitz = 2.0 * float(np.linalg.norm(scores.T @ weighted_scores, ord=2)) + 2.0 * self.lambda_reg
        step_size = 1.0 / max(lipschitz, 1e-8)
        previous_objective = math.inf

        for _ in range(self.max_iter):
            residual = labels - intercept - scores @ weights
            grad = -2.0 * (scores.T @ (sample_weight * residual)) + 2.0 * self.lambda_reg * (
                weights - uniform
            )
            weights = _project_simplex(weights - step_size * grad)
            if self.fit_intercept:
                intercept = float(np.average(labels - scores @ weights, weights=sample_weight))
            objective = self._objective(scores, labels, sample_weight, uniform, weights, intercept)
            if abs(previous_objective - objective) < self.tol:
                break
            previous_objective = objective
        return weights, intercept, objective

    def predict(self, S: Any) -> np.ndarray:
        if self.weights_ is None:
            raise ValueError("Model is not fitted")
        scores = _as_2d_array(S)
        if scores.shape[1] != len(self.weights_):
            raise ValueError("S column count does not match fitted weights")
        predictions = scores @ self.weights_ + self.intercept_
        if self.clip_predictions:
            predictions = np.clip(predictions, 0.0, 1.0)
        return predictions

    def to_dict(self) -> dict[str, Any]:
        if self.weights_ is None:
            raise ValueError("Model is not fitted")
        names = self.judge_names_ or [f"judge_{idx}" for idx in range(len(self.weights_))]
        weight_map = {name: float(weight) for name, weight in zip(names, self.weights_)}
        hhi = float(np.sum(self.weights_**2))
        return {
            "model_type": "simplex_weighted_judge",
            "weights": weight_map,
            "intercept": float(self.intercept_),
            "lambda_reg": float(self.lambda_reg),
            "fit_intercept": bool(self.fit_intercept),
            "clip_predictions": bool(self.clip_predictions),
            "max_weight": float(np.max(self.weights_)),
            "herfindahl_index": hhi,
            "effective_num_judges": float(1.0 / hhi) if hhi > 0 else math.nan,
            "objective": self.objective_,
        }


def predict_unweighted_mean(S: Any) -> np.ndarray:
    return np.mean(_as_2d_array(S), axis=1)


def predict_median(S: Any) -> np.ndarray:
    return np.median(_as_2d_array(S), axis=1)


@dataclass
class SingleJudgeModel:
    judge_index: int
    judge_name: str

    def predict(self, S: Any) -> np.ndarray:
        scores = _as_2d_array(S)
        return scores[:, self.judge_index]

    def to_dict(self) -> dict[str, Any]:
        return {"model_type": "best_single_judge", "judge_index": self.judge_index, "judge_name": self.judge_name}


def fit_best_single_judge(
    S_train: Any,
    y_train: Any,
    S_val: Any,
    y_val: Any,
    judge_names: list[str] | None = None,
) -> SingleJudgeModel:
    del S_train, y_train
    val_scores = _as_2d_array(S_val)
    labels = _as_1d_array(y_val, expected_n=val_scores.shape[0])
    names = judge_names or [f"judge_{idx}" for idx in range(val_scores.shape[1])]
    rmses = [_rmse(labels, val_scores[:, idx]) for idx in range(val_scores.shape[1])]
    best_idx = int(np.argmin(rmses))
    return SingleJudgeModel(judge_index=best_idx, judge_name=names[best_idx])


@dataclass
class RidgeStackingModel:
    coef: np.ndarray
    intercept: float
    fit_intercept: bool

    def predict(self, S: Any) -> np.ndarray:
        scores = _as_2d_array(S)
        return scores @ self.coef + self.intercept

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "ridge_stacking",
            "coef": [float(value) for value in self.coef],
            "intercept": float(self.intercept),
            "fit_intercept": self.fit_intercept,
        }


def fit_ridge_stacking(
    S_train: Any,
    y_train: Any,
    lambda_reg: float,
    fit_intercept: bool = True,
) -> RidgeStackingModel:
    scores = _as_2d_array(S_train)
    labels = _as_1d_array(y_train, expected_n=scores.shape[0])
    if fit_intercept:
        design = np.c_[np.ones(scores.shape[0]), scores]
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        solution = np.linalg.pinv(design.T @ design + float(lambda_reg) * penalty) @ design.T @ labels
        intercept = float(solution[0])
        coef = np.asarray(solution[1:], dtype=float)
    else:
        penalty = np.eye(scores.shape[1])
        coef = np.linalg.pinv(scores.T @ scores + float(lambda_reg) * penalty) @ scores.T @ labels
        intercept = 0.0
    return RidgeStackingModel(coef=np.asarray(coef, dtype=float), intercept=intercept, fit_intercept=fit_intercept)


def select_simplex_model(
    S_train: Any,
    y_train: Any,
    S_val: Any,
    y_val: Any,
    lambda_grid: list[float],
    fit_intercept_options: list[bool],
    sample_weight_train: Any = None,
    judge_names: list[str] | None = None,
    metric: str = "rmse",
    clip_predictions: bool = True,
) -> tuple[SimplexWeightedJudge, dict[str, Any]]:
    if metric != "rmse":
        raise ValueError("Only rmse selection is currently supported")
    train_scores = _as_2d_array(S_train)
    train_labels = _as_1d_array(y_train, expected_n=train_scores.shape[0])
    val_scores = _as_2d_array(S_val)
    val_labels = _as_1d_array(y_val, expected_n=val_scores.shape[0])
    best_model: SimplexWeightedJudge | None = None
    best_score = math.inf
    candidates: list[dict[str, Any]] = []
    for lambda_reg in lambda_grid:
        for fit_intercept in fit_intercept_options:
            model = SimplexWeightedJudge(
                fit_intercept=fit_intercept,
                lambda_reg=float(lambda_reg),
                clip_predictions=clip_predictions,
            ).fit(train_scores, train_labels, sample_weight=sample_weight_train, judge_names=judge_names)
            val_pred = model.predict(val_scores)
            score = _rmse(val_labels, val_pred)
            candidates.append(
                {
                    "lambda_reg": float(lambda_reg),
                    "fit_intercept": bool(fit_intercept),
                    "validation_rmse": score,
                }
            )
            if score < best_score:
                best_score = score
                best_model = model
    if best_model is None:
        raise ValueError("No simplex model candidates were fitted")
    diagnostics = best_model.to_dict()
    diagnostics.update(
        {
            "selected_lambda": float(best_model.lambda_reg),
            "selected_fit_intercept": bool(best_model.fit_intercept),
            "validation_rmse": float(best_score),
            "candidates": candidates,
        }
    )
    return best_model, diagnostics


def select_ridge_stacking_model(
    S_train: Any,
    y_train: Any,
    S_val: Any,
    y_val: Any,
    lambda_grid: list[float],
    fit_intercept: bool = True,
) -> tuple[RidgeStackingModel, dict[str, Any]]:
    val_scores = _as_2d_array(S_val)
    val_labels = _as_1d_array(y_val, expected_n=val_scores.shape[0])
    best_model: RidgeStackingModel | None = None
    best_score = math.inf
    candidates: list[dict[str, Any]] = []
    for lambda_reg in lambda_grid:
        model = fit_ridge_stacking(S_train, y_train, lambda_reg=lambda_reg, fit_intercept=fit_intercept)
        score = _rmse(val_labels, model.predict(val_scores))
        candidates.append({"lambda_reg": float(lambda_reg), "validation_rmse": score})
        if score < best_score:
            best_score = score
            best_model = model
            selected_lambda = float(lambda_reg)
    if best_model is None:
        raise ValueError("No ridge model candidates were fitted")
    diagnostics = best_model.to_dict()
    diagnostics.update(
        {
            "selected_lambda": selected_lambda,
            "validation_rmse": float(best_score),
            "candidates": candidates,
        }
    )
    return best_model, diagnostics
