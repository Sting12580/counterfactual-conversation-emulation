"""Run the policy-swap diagnostic for expert best-embedding datasets.

The diagnostic compares the original estimator direction
behavior=doctor/target=agent against the swapped direction
behavior=agent/target=doctor. In the swapped direction, the old agent rewards
become logged behavior rewards by construction; the target-side rewards remain
diagnostic ground truth only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_expert_best_embedding import (  # noqa: E402
    BASE_EMBEDDING_DIM,
    BASE_EMBEDDING_NAME,
    EXPERT_TASKS,
    OUTPUT_ROOT,
    ExpertTask,
    included_records,
    learned_embedding_config,
    load_embedding_cache,
    text_hash,
)
from run_phase5 import load_jsonl, make_bge_m3_embedder  # noqa: E402

from cce_data.estimators.diagnostics import (  # noqa: E402
    ci_miss_side,
    swap_behavior_target,
    target_progress,
)
from cce_data.estimators.real_runner import (  # noqa: E402
    RealData,
    apply_learned_action_embedding,
    mips_real,
    run_estimators_on_data,
)


POLICY_SWAP_ESTIMATORS: tuple[str, ...] = ("MIPS", "DM", "OffCEM")

SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "display_name",
    "direction",
    "behavior_role",
    "target_role",
    "estimator",
    "n",
    "v_behavior",
    "v_target",
    "true_effect",
    "v_hat",
    "bias_to_target",
    "abs_bias_to_target",
    "target_progress",
    "direction_correct",
    "rel_bias",
    "ci_low",
    "ci_high",
    "ci_covers_target",
    "ci_miss_side",
    "ess",
    "ess_fraction",
    "learned_uses_target_rewards",
    "learned_output_embedding_dim",
    "learned_feature_dim_after_concat",
)


@dataclass(frozen=True)
class DirectionSpec:
    direction: str
    behavior_role: str
    target_role: str
    swapped: bool


DIRECTIONS: tuple[DirectionSpec, ...] = (
    DirectionSpec(
        direction="original_doctor_to_agent",
        behavior_role="doctor",
        target_role="agent",
        swapped=False,
    ),
    DirectionSpec(
        direction="swapped_agent_to_doctor",
        behavior_role="agent",
        target_role="doctor",
        swapped=True,
    ),
)


class LazyBgeM3Embedder:
    """Load BGE-M3 only if an embedding cache miss requires it."""

    def __init__(self) -> None:
        self._embed_fn = None

    def __call__(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is None:
            self._embed_fn = make_bge_m3_embedder()
        return self._embed_fn(texts)


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
        f"  Embedding/cache: {len(xs)} contexts + {len(a_clinician)} doctor "
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
            f"Embedder returned shape {embeddings.shape} for {len(texts)} "
            f"{dataset_name}/{role} texts."
        )
    if embeddings.shape[1] != BASE_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {BASE_EMBEDDING_DIM}-d {BASE_EMBEDDING_NAME} embeddings, "
            f"got {embeddings.shape[1]}."
        )

    if use_cache:
        if path.exists():
            print(f"    computed {role}; not overwriting existing cache {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                embeddings=embeddings,
                text_hash=np.array(expected_hash),
                n_texts=np.array(len(texts), dtype=np.int64),
                model=np.array(BASE_EMBEDDING_NAME),
            )
            print(f"    saved cache {path}")
    return embeddings


def make_summary_row(
    task: ExpertTask,
    direction: DirectionSpec,
    estimator: str,
    report: dict,
    result: dict,
    learned_diag: dict,
    mips_diag: dict | None,
) -> dict:
    v_behavior = float(report["v_true_b"])
    v_target = float(report["v_true_agent"])
    v_hat = float(result["v_hat"])
    bias = float(v_hat - v_target)
    ci_low = result.get("ci_low", float("nan"))
    ci_high = result.get("ci_high", float("nan"))
    miss_side = ci_miss_side(ci_low, ci_high, v_target)

    ess = float("nan")
    ess_fraction = float("nan")
    if estimator == "MIPS":
        if mips_diag is None:
            raise ValueError("MIPS diagnostics are required for MIPS rows.")
        if not np.isclose(float(mips_diag["v_hat"]), v_hat, rtol=0.0, atol=1e-10):
            raise RuntimeError(
                "Direct MIPS diagnostic v_hat differs from run_estimators_on_data: "
                f"{mips_diag['v_hat']} vs {v_hat}."
            )
        ess_fraction = float(mips_diag["ess"])
        ess = float(ess_fraction * int(report["n"]))

    return {
        "dataset": task.name,
        "display_name": task.display_name,
        "direction": direction.direction,
        "behavior_role": direction.behavior_role,
        "target_role": direction.target_role,
        "estimator": estimator,
        "n": int(report["n"]),
        "v_behavior": v_behavior,
        "v_target": v_target,
        "true_effect": float(report["true_effect"]),
        "v_hat": v_hat,
        "bias_to_target": bias,
        "abs_bias_to_target": abs(bias),
        "target_progress": target_progress(v_hat, v_behavior, v_target),
        "direction_correct": bool(result["direction_correct"]),
        "rel_bias": float(result["rel_bias"]),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_covers_target": result.get("ci_covers_truth", float("nan")),
        "ci_miss_side": miss_side,
        "ess": ess,
        "ess_fraction": ess_fraction,
        "learned_uses_target_rewards": bool(learned_diag.get("uses_agent_rewards", False)),
        "learned_output_embedding_dim": int(learned_diag["output_embedding_dim"]),
        "learned_feature_dim_after_concat": int(learned_diag["feature_dim_after_concat"]),
    }


def run_policy_swap(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    output_dir = resolve_path(args.output_dir)
    cache_dir = resolve_path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    config = learned_embedding_config(seed=args.seed)
    embed_fn = LazyBgeM3Embedder()
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
        base_data, _ = featurize_records_with_cache(
            task=task,
            records=records,
            embed_fn=embed_fn,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
        )

        for direction in DIRECTIONS:
            print()
            print(
                f"Direction: {direction.direction} "
                f"({direction.behavior_role} -> {direction.target_role})"
            )
            direction_data = swap_behavior_target(base_data) if direction.swapped else base_data
            print(
                "  Learning reward-informed embedding "
                f"(latent_dim={config.latent_dim}, pca_dim={config.pca_dim})"
            )
            learned_data, learned_diag = apply_learned_action_embedding(direction_data, config)
            report = run_estimators_on_data(
                learned_data,
                n_boot=args.n_boot,
                seed=args.seed,
                estimator_names=args.estimators,
                learned_diagnostics=learned_diag,
            )

            mips_diag = None
            if "MIPS" in report["results"]:
                mips_diag = mips_real(learned_data, seed=args.seed)

            for estimator, result in report["results"].items():
                rows.append(
                    make_summary_row(
                        task=task,
                        direction=direction,
                        estimator=estimator,
                        report=report,
                        result=result,
                        learned_diag=learned_diag,
                        mips_diag=mips_diag,
                    )
                )

    summary = pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)
    metadata = {
        "diagnostic_name": "policy_swap",
        "original_direction": "doctor_to_agent",
        "swapped_direction": "agent_to_doctor",
        "intended_interpretation": (
            "If behavior-support anchoring is driving downward bias, MIPS should "
            "underestimate agent value in the original doctor-to-agent direction "
            "and overestimate doctor value in the swapped agent-to-doctor direction."
        ),
        "warning": (
            "This is a diagnostic stress test only. It does not fix estimator bias "
            "or confidence-interval coverage."
        ),
        "embedding": BASE_EMBEDDING_NAME,
        "embedding_setup": "BGE-M3 base embeddings + learned concat-PCA action embedding",
        "learned_embedding_config": config.__dict__,
        "estimators": list(args.estimators),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "cache_enabled": not args.no_cache,
        "swapped_uses_original_agent_rewards_as_behavior_rewards": True,
        "target_side_rewards_used_for": "diagnostic bias and CI coverage evaluation only",
        "wall_time_seconds": time.time() - started,
    }
    return summary, metadata


def write_outputs(summary: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    csv_path = output_dir / "policy_swap_summary.csv"
    json_path = output_dir / "policy_swap_summary.json"
    report_path = output_dir / "policy_swap_report.md"

    summary.to_csv(csv_path, index=False)
    rows = summary.to_dict(orient="records")
    json_payload = {
        "metadata": metadata,
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(json_sanitize(json_payload), indent=2),
        encoding="utf-8",
    )
    report_path.write_text(build_markdown_report(summary, metadata), encoding="utf-8")

    print()
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"Saved {report_path}")


def build_markdown_report(summary: pd.DataFrame, metadata: dict) -> str:
    lines: list[str] = [
        "# Policy-swap Diagnostic",
        "",
        "## Executive Summary",
        "",
        (
            "This diagnostic reverses the behavior and target roles to test whether "
            "the downward bias pattern is consistent with behavior-support anchoring. "
            "The expected signature is negative MIPS bias in the original "
            "doctor-to-agent direction and positive MIPS bias in the swapped "
            "agent-to-doctor direction."
        ),
        "",
        metadata["warning"],
        "",
        "## MIPS Original vs Swapped",
        "",
    ]

    flags = policy_signature_flags(summary)
    mips_rows = []
    for task in EXPERT_TASKS:
        flag = flags.get(task.name, {})
        mips_rows.append(
            [
                task.display_name,
                fmt_float(flag.get("original_mips_bias")),
                fmt_float(flag.get("swapped_mips_bias")),
                fmt_float(flag.get("original_mips_progress")),
                fmt_float(flag.get("swapped_mips_progress")),
                flag.get("signature", "mips_not_run"),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Dataset",
                "Original MIPS Bias",
                "Swapped MIPS Bias",
                "Original Progress",
                "Swapped Progress",
                "Flag",
            ],
            mips_rows,
        )
    )
    lines.extend(["", "## All Estimators", ""])

    all_rows = []
    for row in summary.to_dict(orient="records"):
        all_rows.append(
            [
                row["display_name"],
                row["direction"],
                row["estimator"],
                fmt_float(row["v_behavior"]),
                fmt_float(row["v_target"]),
                fmt_float(row["true_effect"]),
                fmt_float(row["v_hat"]),
                fmt_float(row["bias_to_target"]),
                fmt_float(row["target_progress"]),
                str(row["ci_miss_side"]),
                fmt_float(row["ess_fraction"]),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Dataset",
                "Direction",
                "Estimator",
                "V Behavior",
                "V Target",
                "True Effect",
                "V Hat",
                "Bias",
                "Progress",
                "CI Miss",
                "ESS Frac",
            ],
            all_rows,
        )
    )
    lines.extend(["", "## Dataset Interpretation", ""])

    for task in EXPERT_TASKS:
        flag = flags.get(task.name, {})
        signature = flag.get("signature", "mips_not_run")
        original_bias = fmt_float(flag.get("original_mips_bias"))
        swapped_bias = fmt_float(flag.get("swapped_mips_bias"))
        if signature == "strong_behavior_anchoring_signature":
            interpretation = (
                "matches the strongest behavior-anchoring signature: original "
                "MIPS undershoots the agent target and swapped MIPS overshoots "
                "the doctor target."
            )
        elif signature == "one_way_support_failure":
            interpretation = (
                "shows a one-way support-failure pattern: original MIPS "
                "undershoots, while the swapped bias is near zero."
            )
        elif signature == "mips_not_run":
            interpretation = "cannot be classified because MIPS was not run."
        else:
            interpretation = "is mixed or unresolved under this diagnostic."
        lines.append(
            f"- **{task.display_name}**: original MIPS bias={original_bias}, "
            f"swapped MIPS bias={swapped_bias}; {interpretation}"
        )

    lines.append("")
    return "\n".join(lines)


def policy_signature_flags(summary: pd.DataFrame) -> dict[str, dict]:
    flags: dict[str, dict] = {}
    mips = summary[summary["estimator"] == "MIPS"]
    for task in EXPERT_TASKS:
        task_rows = mips[mips["dataset"] == task.name]
        original = task_rows[task_rows["direction"] == "original_doctor_to_agent"]
        swapped = task_rows[task_rows["direction"] == "swapped_agent_to_doctor"]
        if original.empty or swapped.empty:
            flags[task.name] = {"signature": "mips_not_run"}
            continue

        original_row = original.iloc[0]
        swapped_row = swapped.iloc[0]
        original_bias = float(original_row["bias_to_target"])
        swapped_bias = float(swapped_row["bias_to_target"])
        if original_bias < 0 and swapped_bias > 0:
            signature = "strong_behavior_anchoring_signature"
        elif original_bias < 0 and abs(swapped_bias) <= 0.05:
            signature = "one_way_support_failure"
        else:
            signature = "mixed_or_unresolved"
        flags[task.name] = {
            "original_mips_bias": original_bias,
            "swapped_mips_bias": swapped_bias,
            "original_mips_progress": float(original_row["target_progress"]),
            "swapped_mips_progress": float(swapped_row["target_progress"]),
            "signature": signature,
        }
    return flags


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    row_lists = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in row_lists)) if row_lists else len(headers[i])
        for i in range(len(headers))
    ]
    header = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in row_lists
    ]
    return "\n".join([header, sep, *body])


def fmt_float(value, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def json_sanitize(value):
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the policy-swap diagnostic.")
    parser.add_argument("--n-boot", type=int, default=0)
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=POLICY_SWAP_ESTIMATORS,
        default=list(POLICY_SWAP_ESTIMATORS),
        help="Estimator(s) to run.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=OUTPUT_ROOT / "cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.n_boot < 0:
        raise ValueError("--n-boot must be non-negative.")
    return args


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    summary, metadata = run_policy_swap(args)
    write_outputs(summary, metadata, resolve_path(args.output_dir))


if __name__ == "__main__":
    main()
