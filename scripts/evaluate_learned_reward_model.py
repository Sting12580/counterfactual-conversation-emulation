"""Evaluate the learned-embedding two-tower reward model on a held-out test set.

This is an intrinsic reward-prediction diagnostic for the learned action
embedding layer. It trains the same two-tower model used by
``learned_embedding.py`` on logged clinician tuples:

    (phi_x, phi_a_clinician) -> y_score

The validation split is used only for early stopping. The test split is held
out until the final evaluation, so the reported test metrics are a cleaner
measure of reward-model generalization than the Phase 5 JSON
``validation_mse`` field.

Usage:
    PYTHONPATH=src python scripts/evaluate_learned_reward_model.py \
      --input data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl \
      --score-field y_score_claude_sonnet46 \
      --embedder bge \
      --latent-dim 8 \
      --hidden-dim 32 \
      --epochs 250 \
      --weight-decay 0.3 \
      --dropout 0.1 \
      --patience 25 \
      --output data/phase5/learned_reward_model_holdout_sonnet46_bge_tuned_regularized.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase5 import (  # noqa: E402
    load_jsonl,
    make_bge_m3_embedder,
    make_medcpt_bge_embedder,
    make_medcpt_embedder,
    make_openai_embedder,
    make_sbert_embedder,
    remap_score_fields,
)

from cce_data.estimators.learned_embedding import (  # noqa: E402
    LearnedEmbeddingConfig,
    _build_two_tower_reward_projector,
)


def _included_reward_rows(records: list[dict]) -> list[dict]:
    return [
        row
        for row in records
        if row.get("inclusion_status") == "included"
        and row.get("y_score") is not None
        and row.get("x_patient_context")
        and row.get("a_clinician")
    ]


def _split_indices(
    n: int,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if n < 5:
        raise ValueError("Need at least 5 rows for train/validation/test evaluation.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1).")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1).")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1.")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = min(max(int(round(n * test_fraction)), 1), n - 2)
    n_val = min(max(int(round(n * validation_fraction)), 1), n - n_test - 1)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split leaves no training rows.")

    return {
        "test": idx[:n_test],
        "validation": idx[n_test:n_test + n_val],
        "train": idx[n_test + n_val:],
    }


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks using only NumPy."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.isclose(a.std(), 0.0) or np.isclose(b.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline_value: float,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    baseline_mse = float(np.mean((baseline_value - y_true) ** 2))
    return {
        "n": int(len(y_true)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "baseline_mse_train_mean": baseline_mse,
        "mse_reduction_vs_train_mean": (
            float(1.0 - mse / baseline_mse) if baseline_mse > 0 else float("nan")
        ),
        "pearson": _correlation(y_true, y_pred),
        "spearman": _correlation(_rankdata_average(y_true), _rankdata_average(y_pred)),
        "target_mean": float(y_true.mean()),
        "prediction_mean": float(y_pred.mean()),
        "target_std": float(y_true.std(ddof=0)),
        "prediction_std": float(y_pred.std(ddof=0)),
    }


def evaluate_two_tower_holdout(
    phi_x: np.ndarray,
    phi_a_clinician: np.ndarray,
    y_clinician: np.ndarray,
    *,
    config: LearnedEmbeddingConfig,
    test_fraction: float,
) -> dict:
    """Train on train split, early-stop on validation, evaluate final test."""
    import torch
    from torch import nn

    x_np = np.asarray(phi_x, dtype=np.float32)
    a_np = np.asarray(phi_a_clinician, dtype=np.float32)
    y_np = np.asarray(y_clinician, dtype=np.float32).reshape(-1, 1)
    splits = _split_indices(
        len(y_np),
        validation_fraction=config.validation_fraction,
        test_fraction=test_fraction,
        seed=config.seed,
    )

    torch.manual_seed(config.seed)
    device = "cpu"
    x = torch.from_numpy(x_np).to(device)
    a = torch.from_numpy(a_np).to(device)

    train_idx = splits["train"]
    validation_idx = splits["validation"]
    test_idx = splits["test"]

    y_mean = float(y_np[train_idx].mean())
    y_std = float(y_np[train_idx].std() + 1e-6)
    y_standardized = torch.from_numpy((y_np - y_mean) / y_std).to(device)

    train_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    validation_t = torch.as_tensor(validation_idx, dtype=torch.long, device=device)
    model = _build_two_tower_reward_projector(
        context_dim=x_np.shape[1],
        action_dim=a_np.shape[1],
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_trained = 0

    for epoch in range(config.max_epochs):
        epochs_trained = epoch + 1
        model.train()
        optimizer.zero_grad()
        train_pred = model(x[train_t], a[train_t])
        train_loss = loss_fn(train_pred, y_standardized[train_t])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_fn(model(x[validation_t], a[validation_t]), y_standardized[validation_t]).item()
            )
        if validation_loss + 1e-7 < best_validation:
            best_validation = validation_loss
            best_epoch = epoch + 1
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if config.patience > 0 and stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(x, a).cpu().numpy().reshape(-1) * y_std + y_mean

    y_raw = y_np.reshape(-1).astype(float)
    metrics = {
        name: _regression_metrics(y_raw[idx], pred[idx], baseline_value=y_mean)
        for name, idx in splits.items()
    }
    return {
        "split_sizes": {name: int(len(idx)) for name, idx in splits.items()},
        "split_indices": {name: idx.astype(int).tolist() for name, idx in splits.items()},
        "target_standardization": {"train_mean": y_mean, "train_std": y_std},
        "epochs_trained": int(epochs_trained),
        "best_epoch": int(best_epoch),
        "best_validation_mse_standardized": float(best_validation),
        "metrics": metrics,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/phase3/agent_scored_all_judge_claude_sonnet46.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path(""))
    parser.add_argument(
        "--embedder",
        choices=["sbert", "openai", "medcpt", "bge", "medcpt-bge"],
        default="bge",
    )
    parser.add_argument(
        "--score-field",
        default="",
        help="Optional source field to expose as y_score before evaluation.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input: {args.input}")

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    records = remap_score_fields(records, score_field=args.score_field or None)
    included = _included_reward_rows(records)
    print(f"  {len(included)} included clinician reward rows")

    xs = [row["x_patient_context"] for row in included]
    actions = [row["a_clinician"] for row in included]
    y = np.array([row["y_score"] for row in included], dtype=float)

    embedder_factories = {
        "sbert": make_sbert_embedder,
        "openai": make_openai_embedder,
        "medcpt": make_medcpt_embedder,
        "bge": make_bge_m3_embedder,
        "medcpt-bge": make_medcpt_bge_embedder,
    }
    embed_fn = embedder_factories[args.embedder]()
    print(f"  Embedding {len(xs)} contexts + {len(actions)} clinician actions ...")
    t0 = time.time()
    phi_x = embed_fn(xs)
    phi_a = embed_fn(actions)
    print(f"  Embedding done in {time.time() - t0:.1f}s")

    config = LearnedEmbeddingConfig(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        seed=args.seed,
    )
    print(
        "Training two-tower reward model "
        f"(latent_dim={config.latent_dim}, hidden_dim={config.hidden_dim}) ..."
    )
    t1 = time.time()
    evaluation = evaluate_two_tower_holdout(
        phi_x,
        phi_a,
        y,
        config=config,
        test_fraction=args.test_fraction,
    )
    print(f"  Training/evaluation done in {time.time() - t1:.1f}s")

    report = {
        "input": str(args.input),
        "score_field": args.score_field or "y_score",
        "embedder": args.embedder,
        "n": int(len(included)),
        "context_dim": int(phi_x.shape[1]),
        "action_dim": int(phi_a.shape[1]),
        "config": asdict(config),
        "test_fraction": float(args.test_fraction),
        **evaluation,
    }

    print()
    print("Two-tower reward-model intrinsic performance")
    print("=" * 70)
    print(f"n={report['n']}  embedder={args.embedder}  score={report['score_field']}")
    print(
        f"best_epoch={report['best_epoch']} / epochs_trained={report['epochs_trained']}  "
        f"best_val_mse_std={report['best_validation_mse_standardized']:.4f}"
    )
    print("-" * 70)
    print(f"{'Split':<12} {'n':>5} {'RMSE':>9} {'MAE':>9} {'R2/base':>9} {'Pearson':>9} {'Spearman':>9}")
    for split in ["train", "validation", "test"]:
        m = report["metrics"][split]
        print(
            f"{split:<12} {m['n']:>5} {m['rmse']:>9.4f} {m['mae']:>9.4f} "
            f"{m['mse_reduction_vs_train_mean']:>9.2%} {m['pearson']:>9.3f} {m['spearman']:>9.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(report, f, indent=2, default=float)
        print()
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
