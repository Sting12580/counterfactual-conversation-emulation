"""Run support diagnostics for expert-scored datasets with the best embedding.

Outputs are intentionally confined to ``outputs/expert_best_embedding_diagnostics``.
The learned embedding is trained only from logged behavior rewards; target
``y_agent_score`` is used after fitting for diagnostic evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase5 import load_jsonl, make_bge_m3_embedder  # noqa: E402

from cce_data.estimators.density_ratio import (  # noqa: E402
    density_ratio_with_diagnostics,
    fit_density_ratio_classifier,
    summarize_density_ratio_weights,
)
from cce_data.estimators.diagnostics import (  # noqa: E402
    REWARD_QUANTILES,
    make_knn_rows,
    nearest_neighbor_support_diagnostics,
    required_topk_fraction,
    reward_distribution_summary,
)
from cce_data.estimators.learned_embedding import LearnedEmbeddingConfig  # noqa: E402
from cce_data.estimators.real_runner import (  # noqa: E402
    RealData,
    apply_learned_action_embedding,
    run_estimators_on_data,
)
from cce_data.estimators.surface_features import (  # noqa: E402
    extract_action_surface_features,
    standardize_behavior_target_features,
)


OUTPUT_ROOT = Path("outputs/expert_best_embedding_diagnostics")
BASE_EMBEDDING_NAME = "BAAI/bge-m3"
BASE_EMBEDDING_DIM = 1024
DENSITY_C_VALUES: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
DENSITY_CLIP_VALUES: tuple[float | None, ...] = (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, None)
DENSITY_CALIBRATE_VALUES: tuple[bool, ...] = (True, False)
FEATURE_MODES: tuple[str, ...] = ("embedding_only", "surface_only", "embedding_plus_surface")
FEATURE_ESTIMATORS: tuple[str, ...] = ("DM", "MIPS", "OffCEM")


@dataclass(frozen=True)
class ExpertTask:
    name: str
    input_path: Path
    display_name: str


EXPERT_TASKS: tuple[ExpertTask, ...] = (
    ExpertTask(
        name="ayers_composite",
        input_path=Path("data/ayers_askdocs/phase3_chatgpt_expert_scored.jsonl"),
        display_name="Ayers AskDocs composite",
    ),
    ExpertTask(
        name="ayers_quality",
        input_path=Path("data/ayers_askdocs_quality/phase3_chatgpt_expert_scored.jsonl"),
        display_name="Ayers AskDocs quality",
    ),
    ExpertTask(
        name="ayers_empathy",
        input_path=Path("data/ayers_askdocs_empathy/phase3_chatgpt_expert_scored.jsonl"),
        display_name="Ayers AskDocs empathy",
    ),
    ExpertTask(
        name="counselbench_gpt4",
        input_path=Path("data/counselbench/phase3_gpt4_expert_scored.jsonl"),
        display_name="CounselBench GPT-4 responder",
    ),
    ExpertTask(
        name="counselbench_llama3",
        input_path=Path("data/counselbench/phase3_llama3_expert_scored.jsonl"),
        display_name="CounselBench LLaMA-3 responder",
    ),
    ExpertTask(
        name="counselbench_gemini",
        input_path=Path("data/counselbench/phase3_gemini_expert_scored.jsonl"),
        display_name="CounselBench Gemini responder",
    ),
)


def learned_embedding_config(seed: int = 0) -> LearnedEmbeddingConfig:
    return LearnedEmbeddingConfig(
        latent_dim=64,
        hidden_dim=64,
        merge_strategy="concat-pca",
        pca_dim=128,
        max_epochs=100,
        learning_rate=1e-3,
        weight_decay=0.01,
        validation_fraction=0.2,
        patience=50,
        seed=seed,
    )


def included_records(records: list[dict]) -> list[dict]:
    return [
        r
        for r in records
        if r.get("inclusion_status") == "included"
        and r.get("y_score") is not None
        and r.get("y_agent_score") is not None
    ]


def featurize_records_with_cache(
    task: ExpertTask,
    records: list[dict],
    embed_fn,
    cache_dir: Path,
    use_cache: bool = True,
) -> tuple[RealData, list[dict]]:
    included = included_records(records)
    xs = [r["x_patient_context"] for r in included]
    a_clinician = [r["a_clinician"] for r in included]
    a_agent = [r["a_agent"] for r in included]
    y_behavior = np.array([r["y_score"] for r in included], dtype=float)
    y_agent = np.array([r["y_agent_score"] for r in included], dtype=float)

    print(
        f"  Embedding/cache: {len(xs)} contexts + {len(a_clinician)} clinician "
        f"+ {len(a_agent)} agent actions"
    )
    phi_x = embed_role_with_cache(task.name, "x_patient_context", xs, embed_fn, cache_dir, use_cache)
    phi_a_cl = embed_role_with_cache(task.name, "a_clinician", a_clinician, embed_fn, cache_dir, use_cache)
    phi_a_ag = embed_role_with_cache(task.name, "a_agent", a_agent, embed_fn, cache_dir, use_cache)
    data = RealData(
        phi_x=phi_x,
        phi_a_clinician=phi_a_cl,
        phi_a_agent=phi_a_ag,
        y_clinician=y_behavior,
        y_agent=y_agent,
    )
    return data, included


def embed_role_with_cache(
    dataset_name: str,
    role: str,
    texts: list[str],
    embed_fn,
    cache_dir: Path,
    use_cache: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset_name}__{role}__bge_m3.npz"
    expected_hash = text_hash(texts)

    if use_cache:
        cached = load_embedding_cache(path, len(texts), expected_hash)
        if cached is not None:
            print(f"    loaded cache {path}")
            return cached

    embeddings = np.asarray(embed_fn(texts), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Embedder returned shape {embeddings.shape} for {len(texts)} {dataset_name}/{role} texts."
        )
    if embeddings.shape[1] != BASE_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {BASE_EMBEDDING_DIM}-d {BASE_EMBEDDING_NAME} embeddings, "
            f"got {embeddings.shape[1]}."
        )
    if use_cache:
        np.savez_compressed(
            path,
            embeddings=embeddings,
            text_hash=np.array(expected_hash),
            n_texts=np.array(len(texts), dtype=np.int64),
            model=np.array(BASE_EMBEDDING_NAME),
        )
        print(f"    saved cache {path}")
    return embeddings


def load_embedding_cache(path: Path, expected_n: int, expected_hash: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
            cached_hash = str(cache["text_hash"].item()) if "text_hash" in cache else ""
            cached_model = str(cache["model"].item()) if "model" in cache else ""
    except Exception as exc:
        print(f"    ignoring unreadable cache {path}: {exc}")
        return None

    expected_shape = (expected_n, BASE_EMBEDDING_DIM)
    if embeddings.shape != expected_shape:
        print(f"    ignoring cache {path}: shape {embeddings.shape} != {expected_shape}")
        return None
    if cached_hash != expected_hash:
        print(f"    ignoring cache {path}: text hash mismatch")
        return None
    if cached_model and cached_model != BASE_EMBEDDING_NAME:
        print(f"    ignoring cache {path}: model {cached_model} != {BASE_EMBEDDING_NAME}")
        return None
    return embeddings


def text_hash(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def support_summary_row(
    task: ExpertTask,
    data: RealData,
    support_diag: dict,
    learned_diag: dict,
) -> dict:
    reward_diag = reward_distribution_summary(data.y_clinician, data.y_agent)
    topk_diag = required_topk_fraction(data.y_clinician, reward_diag["target_mean"])
    row = {
        "dataset": task.name,
        "display_name": task.display_name,
        "input_path": str(task.input_path),
        "n": int(data.n),
        "v_behavior": reward_diag["behavior_mean"],
        "v_agent": reward_diag["target_mean"],
        "true_effect": float(reward_diag["target_mean"] - reward_diag["behavior_mean"]),
        "required_topk_k": topk_diag["required_topk_k"],
        "required_topk_fraction": topk_diag["required_topk_fraction"],
        "required_topk_mean": topk_diag["required_topk_mean"],
        "required_topk_achievable": topk_diag["achievable_by_topk"],
        "learned_output_embedding_dim": learned_diag["output_embedding_dim"],
        "learned_feature_dim_after_concat": learned_diag["feature_dim_after_concat"],
        "learned_uses_agent_rewards": learned_diag["uses_agent_rewards"],
        "support_severity": support_severity(reward_diag, topk_diag, support_diag),
    }

    for q in REWARD_QUANTILES:
        suffix = f"q{int(round(q * 100)):02d}"
        row[f"behavior_reward_{suffix}"] = reward_diag[f"behavior_{suffix}"]
        row[f"agent_reward_{suffix}"] = reward_diag[f"target_{suffix}"]

    for key, value in support_diag.items():
        row[key] = value
    return row


def support_severity(reward_diag: dict, topk_diag: dict, support_diag: dict) -> str:
    topk_fraction = float(topk_diag["required_topk_fraction"])
    target_mean = float(reward_diag["target_mean"])
    behavior_q90 = float(reward_diag["behavior_q90"])
    behavior_q95 = float(reward_diag["behavior_q95"])
    knn10_gap = float(support_diag.get("knn10_gap_to_agent_mean", 0.0))
    frac_above_knn10 = float(support_diag.get("frac_agent_reward_above_knn10_max", 0.0))

    if (
        not topk_diag["achievable_by_topk"]
        or topk_fraction < 0.10
        or target_mean > behavior_q95
        or knn10_gap > 0.15
        or frac_above_knn10 > 0.40
    ):
        return "high"
    if (
        topk_fraction < 0.25
        or target_mean > behavior_q90
        or knn10_gap > 0.05
        or frac_above_knn10 > 0.10
    ):
        return "medium"
    return "low"


def run_support_section(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    cache_dir = ensure_output_dir(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    embed_fn = make_bge_m3_embedder()
    config = learned_embedding_config(seed=args.seed)
    rows: list[dict] = []
    started = time.time()

    for task in EXPERT_TASKS:
        print()
        print("=" * 80)
        print(f"Dataset: {task.name} ({task.display_name})")
        print("=" * 80)
        if not task.input_path.exists():
            raise FileNotFoundError(f"Missing input for {task.name}: {task.input_path}")

        records = load_jsonl(task.input_path)
        base_data, aligned_records = featurize_records_with_cache(
            task=task,
            records=records,
            embed_fn=embed_fn,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
        )

        print(
            "  Learning reward-informed embedding "
            f"(latent_dim={config.latent_dim}, pca_dim={config.pca_dim})"
        )
        learned_data, learned_diag = apply_learned_action_embedding(base_data, config)
        support_diag = nearest_neighbor_support_diagnostics(learned_data)
        row = support_summary_row(task, learned_data, support_diag, learned_diag)
        rows.append(row)

        knn_rows = make_knn_rows(learned_data, aligned_records, k=args.knn_rows_k)
        knn_rows.insert(0, "dataset", task.name)
        knn_path = output_dir / f"{task.name}_knn_rows.csv"
        knn_rows.to_csv(knn_path, index=False)
        print(f"  Saved {knn_path}")

    summary = pd.DataFrame(rows)
    summary = summary.reindex(columns=summary_columns(summary.columns))
    csv_path = output_dir / "support_summary.csv"
    json_path = output_dir / "support_summary.json"
    summary.to_csv(csv_path, index=False)
    json_payload = {
        "metadata": {
            "sections": ["support"],
            "embedding": BASE_EMBEDDING_NAME,
            "learned_embedding_config": config.__dict__,
            "uses_agent_rewards_for_training_or_tuning": False,
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "wall_time_seconds": time.time() - started,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=safe_json), encoding="utf-8")
    print()
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


def run_density_section(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    cache_dir = ensure_output_dir(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    embed_fn = make_bge_m3_embedder()
    config = learned_embedding_config(seed=args.seed)
    rows: list[dict] = []
    started = time.time()

    for task in EXPERT_TASKS:
        print()
        print("=" * 80)
        print(f"Density sweep: {task.name} ({task.display_name})")
        print("=" * 80)
        if not task.input_path.exists():
            raise FileNotFoundError(f"Missing input for {task.name}: {task.input_path}")

        records = load_jsonl(task.input_path)
        base_data, _ = featurize_records_with_cache(
            task=task,
            records=records,
            embed_fn=embed_fn,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
        )
        print(
            "  Learning reward-informed embedding "
            f"(latent_dim={config.latent_dim}, pca_dim={config.pca_dim})"
        )
        data, learned_diag = apply_learned_action_embedding(base_data, config)
        x_target = data.features_at("agent")
        x_behavior = data.features_at("clinician")
        v_behavior = float(data.y_clinician.mean())
        v_agent = float(data.y_agent.mean())

        for c_value in DENSITY_C_VALUES:
            for clip in DENSITY_CLIP_VALUES:
                for calibrate in DENSITY_CALIBRATE_VALUES:
                    clf = fit_density_ratio_classifier(
                        x_target,
                        x_behavior,
                        seed=args.seed,
                        C=c_value,
                        calibrate=calibrate,
                    )
                    weights, weight_diag = density_ratio_with_diagnostics(
                        clf,
                        x_behavior,
                        clip=clip,
                    )
                    summary = summarize_density_ratio_weights(
                        weights,
                        data.y_clinician,
                        raw_weights=weight_diag["raw_weights"],
                        clip=clip,
                    )
                    bias_snips = float(summary["v_snips"] - v_agent)
                    bias_unnormalized = float(summary["v_unnormalized_mips"] - v_agent)
                    row = {
                        "dataset": task.name,
                        "display_name": task.display_name,
                        "input_path": str(task.input_path),
                        "n": int(data.n),
                        "embedding": "BGE-M3 + learned concat-PCA",
                        "learned_output_embedding_dim": learned_diag["output_embedding_dim"],
                        "learned_feature_dim_after_concat": learned_diag[
                            "feature_dim_after_concat"
                        ],
                        "learned_uses_agent_rewards": learned_diag["uses_agent_rewards"],
                        "C": c_value,
                        "clip": clip,
                        "clip_label": "none" if clip is None else str(int(clip)),
                        "calibrate": calibrate,
                        "v_behavior": v_behavior,
                        "v_agent": v_agent,
                        "true_effect": float(v_agent - v_behavior),
                        "bias_snips": bias_snips,
                        "bias_unnormalized": bias_unnormalized,
                        "abs_bias_snips": abs(bias_snips),
                        "abs_bias_unnormalized": abs(bias_unnormalized),
                    }
                    row.update(summary)
                    rows.append(row)
            print(f"  Completed C={c_value}")

    sweep = pd.DataFrame(rows)
    sweep = sweep.reindex(columns=density_columns(sweep.columns))
    sweep_path = output_dir / "density_sweep.csv"
    json_path = output_dir / "density_sweep.json"
    baseline_path = output_dir / "density_baseline_summary.csv"

    sweep.to_csv(sweep_path, index=False)
    baseline = sweep[
        (sweep["C"] == 0.1)
        & (sweep["clip_label"] == "20")
        & (sweep["calibrate"] == True)  # noqa: E712
    ].copy()
    baseline.to_csv(baseline_path, index=False)
    json_payload = {
        "metadata": {
            "sections": ["density"],
            "embedding": BASE_EMBEDDING_NAME,
            "learned_embedding_config": config.__dict__,
            "C_values": list(DENSITY_C_VALUES),
            "clip_values": [None if c is None else float(c) for c in DENSITY_CLIP_VALUES],
            "calibrate_values": list(DENSITY_CALIBRATE_VALUES),
            "uses_agent_rewards_for_training_or_tuning": False,
            "agent_rewards_used_for": "diagnostic bias columns only",
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "wall_time_seconds": time.time() - started,
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=safe_json), encoding="utf-8")
    print()
    print(f"Saved {sweep_path}")
    print(f"Saved {json_path}")
    print(f"Saved {baseline_path}")


def run_features_section(args: argparse.Namespace) -> None:
    output_dir = ensure_output_dir(args.output_dir)
    cache_dir = ensure_output_dir(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    embed_fn = make_bge_m3_embedder()
    config = learned_embedding_config(seed=args.seed)
    ablation_rows: list[dict] = []
    shift_rows: list[dict] = []
    started = time.time()

    for task in EXPERT_TASKS:
        print()
        print("=" * 80)
        print(f"Feature ablation: {task.name} ({task.display_name})")
        print("=" * 80)
        if not task.input_path.exists():
            raise FileNotFoundError(f"Missing input for {task.name}: {task.input_path}")

        records = load_jsonl(task.input_path)
        base_data, aligned_records = featurize_records_with_cache(
            task=task,
            records=records,
            embed_fn=embed_fn,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
        )
        print(
            "  Learning reward-informed embedding "
            f"(latent_dim={config.latent_dim}, pca_dim={config.pca_dim})"
        )
        learned_data, learned_diag = apply_learned_action_embedding(base_data, config)
        surface = build_surface_feature_bundle(task, aligned_records)
        shift_rows.extend(surface["shift_rows"])

        mode_data = {
            "embedding_only": learned_data,
            "surface_only": make_surface_only_data(base_data, surface),
            "embedding_plus_surface": RealData(
                phi_x=learned_data.phi_x,
                phi_a_clinician=learned_data.phi_a_clinician,
                phi_a_agent=learned_data.phi_a_agent,
                y_clinician=learned_data.y_clinician,
                y_agent=learned_data.y_agent,
                extra_clinician=surface["behavior_standardized"],
                extra_agent=surface["agent_standardized"],
            ),
        }

        for mode in FEATURE_MODES:
            data = mode_data[mode]
            report = run_estimators_on_data(
                data,
                n_boot=0,
                seed=args.seed,
                estimator_names=list(FEATURE_ESTIMATORS),
            )
            for estimator, result in report["results"].items():
                row = {
                    "dataset": task.name,
                    "display_name": task.display_name,
                    "input_path": str(task.input_path),
                    "mode": mode,
                    "estimator": estimator,
                    "primary_for_interpretation": estimator == "MIPS",
                    "n": report["n"],
                    "v_behavior": report["v_true_b"],
                    "v_agent": report["v_true_agent"],
                    "true_effect": report["true_effect"],
                    "v_hat": result["v_hat"],
                    "bias": result["bias"],
                    "abs_bias": abs(result["bias"]),
                    "rel_bias": result["rel_bias"],
                    "direction_correct": result["direction_correct"],
                    "feature_dim_behavior": int(data.features_at("clinician").shape[1]),
                    "feature_dim_agent": int(data.features_at("agent").shape[1]),
                    "extra_dim": int(
                        0 if data.extra_clinician is None else data.extra_clinician.shape[1]
                    ),
                    "surface_feature_count": len(surface["feature_names"]),
                    "learned_output_embedding_dim": learned_diag["output_embedding_dim"],
                    "learned_feature_dim_after_concat": learned_diag[
                        "feature_dim_after_concat"
                    ],
                    "learned_uses_agent_rewards": learned_diag["uses_agent_rewards"],
                    "surface_standardized_on": surface["standardization"][
                        "standardized_on"
                    ],
                }
                ablation_rows.append(row)
            print(f"  Completed mode={mode}")

    ablation = pd.DataFrame(ablation_rows)
    ablation = ablation.reindex(columns=feature_ablation_columns(ablation.columns))
    shift = pd.DataFrame(shift_rows)
    shift = shift.reindex(columns=surface_shift_columns(shift.columns))

    ablation_path = output_dir / "feature_ablation.csv"
    json_path = output_dir / "feature_ablation.json"
    shift_path = output_dir / "surface_feature_shift.csv"
    ablation.to_csv(ablation_path, index=False)
    shift.to_csv(shift_path, index=False)
    json_payload = {
        "metadata": {
            "sections": ["features"],
            "embedding": BASE_EMBEDDING_NAME,
            "learned_embedding_config": config.__dict__,
            "feature_modes": list(FEATURE_MODES),
            "estimators": list(FEATURE_ESTIMATORS),
            "uses_agent_rewards_for_training_or_tuning": False,
            "agent_rewards_used_for": "diagnostic bias columns only",
            "surface_features_standardized_on": "behavior",
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "wall_time_seconds": time.time() - started,
        },
        "ablation_rows": ablation_rows,
        "surface_feature_shift_rows": shift_rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=safe_json), encoding="utf-8")
    print()
    print(f"Saved {ablation_path}")
    print(f"Saved {json_path}")
    print(f"Saved {shift_path}")


def build_surface_feature_bundle(task: ExpertTask, records: list[dict]) -> dict:
    behavior_texts = [r["a_clinician"] for r in records]
    agent_texts = [r["a_agent"] for r in records]
    behavior_raw, names = extract_action_surface_features(behavior_texts)
    agent_raw, agent_names = extract_action_surface_features(agent_texts)
    if names != agent_names:
        raise ValueError("Behavior and agent surface feature names differ.")
    behavior_std, agent_std, standardization = standardize_behavior_target_features(
        behavior_raw,
        agent_raw,
    )
    shift_rows = []
    for i, name in enumerate(names):
        behavior_mean = float(behavior_raw[:, i].mean())
        agent_mean = float(agent_raw[:, i].mean())
        behavior_std_raw = float(behavior_raw[:, i].std(ddof=0))
        agent_std_raw = float(agent_raw[:, i].std(ddof=0))
        standardized_behavior_mean = float(behavior_std[:, i].mean())
        standardized_agent_mean = float(agent_std[:, i].mean())
        shift_rows.append(
            {
                "dataset": task.name,
                "display_name": task.display_name,
                "feature": name,
                "behavior_mean": behavior_mean,
                "agent_mean": agent_mean,
                "mean_difference_agent_minus_behavior": float(agent_mean - behavior_mean),
                "behavior_std": behavior_std_raw,
                "agent_std": agent_std_raw,
                "standardized_behavior_mean": standardized_behavior_mean,
                "standardized_agent_mean": standardized_agent_mean,
                "standardized_difference_agent_minus_behavior": float(
                    standardized_agent_mean - standardized_behavior_mean
                ),
            }
        )
    return {
        "feature_names": names,
        "behavior_raw": behavior_raw,
        "agent_raw": agent_raw,
        "behavior_standardized": behavior_std,
        "agent_standardized": agent_std,
        "standardization": standardization,
        "shift_rows": shift_rows,
    }


def make_surface_only_data(base_data: RealData, surface: dict) -> RealData:
    zeros = np.zeros((base_data.n, 0), dtype=np.float32)
    return RealData(
        phi_x=zeros,
        phi_a_clinician=zeros,
        phi_a_agent=zeros,
        y_clinician=base_data.y_clinician,
        y_agent=base_data.y_agent,
        extra_clinician=surface["behavior_standardized"],
        extra_agent=surface["agent_standardized"],
    )


def summary_columns(columns) -> list[str]:
    required = [
        "dataset",
        "display_name",
        "input_path",
        "n",
        "v_behavior",
        "v_agent",
        "true_effect",
    ]
    quantile_cols = []
    for q in REWARD_QUANTILES:
        suffix = f"q{int(round(q * 100)):02d}"
        quantile_cols.extend([f"behavior_reward_{suffix}", f"agent_reward_{suffix}"])
    required.extend(quantile_cols)
    required.extend(
        [
            "required_topk_k",
            "required_topk_fraction",
            "required_topk_mean",
            "required_topk_achievable",
            "knn1_mean_behavior_reward",
            "knn5_mean_behavior_reward",
            "knn10_mean_behavior_reward",
            "knn5_gap_to_agent_mean",
            "knn10_gap_to_agent_mean",
            "frac_agent_reward_above_knn5_max",
            "frac_agent_reward_above_knn10_max",
            "nn_distance_min_mean",
            "nn_distance_min_p90",
            "support_severity",
        ]
    )
    seen = set()
    ordered = []
    for col in required:
        if col in columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def density_columns(columns) -> list[str]:
    required = [
        "dataset",
        "display_name",
        "input_path",
        "n",
        "embedding",
        "C",
        "clip",
        "clip_label",
        "calibrate",
        "v_behavior",
        "v_agent",
        "true_effect",
        "v_snips",
        "v_unnormalized_mips",
        "bias_snips",
        "bias_unnormalized",
        "abs_bias_snips",
        "abs_bias_unnormalized",
        "ess",
        "ess_fraction",
        "mean_w",
        "std_w",
        "raw_max_w",
        "clip_rate",
        "corr_w_y_behavior",
        "top1_weight_mass",
        "top5_weight_mass",
        "top10_weight_mass",
        "snips_minus_unnormalized_mips",
    ]
    for prefix in ("w", "raw_w"):
        for q in (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100):
            required.append(f"{prefix}_q{q:02d}")

    seen = set()
    ordered = []
    for col in required:
        if col in columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def feature_ablation_columns(columns) -> list[str]:
    required = [
        "dataset",
        "display_name",
        "input_path",
        "mode",
        "estimator",
        "primary_for_interpretation",
        "n",
        "v_behavior",
        "v_agent",
        "true_effect",
        "v_hat",
        "bias",
        "abs_bias",
        "rel_bias",
        "direction_correct",
        "feature_dim_behavior",
        "feature_dim_agent",
        "extra_dim",
        "surface_feature_count",
        "learned_output_embedding_dim",
        "learned_feature_dim_after_concat",
        "learned_uses_agent_rewards",
        "surface_standardized_on",
    ]
    return _ordered_columns(required, columns)


def surface_shift_columns(columns) -> list[str]:
    required = [
        "dataset",
        "display_name",
        "feature",
        "behavior_mean",
        "agent_mean",
        "mean_difference_agent_minus_behavior",
        "behavior_std",
        "agent_std",
        "standardized_behavior_mean",
        "standardized_agent_mean",
        "standardized_difference_agent_minus_behavior",
    ]
    return _ordered_columns(required, columns)


def _ordered_columns(required: list[str], columns) -> list[str]:
    seen = set()
    ordered = []
    for col in required:
        if col in columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    for col in columns:
        if col not in seen:
            ordered.append(col)
            seen.add(col)
    return ordered


def ensure_output_dir(path: Path) -> Path:
    root = (Path.cwd() / OUTPUT_ROOT).resolve()
    resolved = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Output path must be under {root}: {resolved}") from exc
    return resolved


def safe_json(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["support", "density", "features"],
        default=["support"],
        help="Diagnostic sections to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=OUTPUT_ROOT / "cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knn-rows-k", type=int, default=10)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    if "support" in args.sections:
        run_support_section(args)
    if "density" in args.sections:
        run_density_section(args)
    if "features" in args.sections:
        run_features_section(args)


if __name__ == "__main__":
    main()
