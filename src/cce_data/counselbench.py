from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from cce_data.build import write_jsonl
from cce_data.effect import compute_ground_truth_effect
from cce_data.schema import CANONICAL_COLUMNS, stable_example_id


COUNSELBENCH_EVAL_REPO = "izi-ano/CounselBench-Eval"
COUNSELBENCH_EVAL_PARQUET_URL = (
    "https://huggingface.co/datasets/izi-ano/CounselBench-Eval/resolve/"
    "refs%2Fconvert%2Fparquet/default/test/0000.parquet"
)

TARGET_RESPONDERS = ("gpt4", "llama3", "gemini")

COMPOSITE_WEIGHTS = {
    "overall_01": 0.30,
    "empathy_01": 0.20,
    "specificity_01": 0.20,
    "factual_01": 0.15,
    "toxicity_safe_01": 0.10,
    "medical_safe_01": 0.05,
}

RewardMode = Literal["composite", "overall"]


def _load_counselbench_eval_frame() -> pd.DataFrame:
    """Load CounselBench-Eval using optional HF tooling, with a parquet fallback."""
    try:
        from datasets import load_dataset

        return load_dataset(COUNSELBENCH_EVAL_REPO, split="test").to_pandas()
    except ImportError:
        try:
            return pd.read_parquet(COUNSELBENCH_EVAL_PARQUET_URL)
        except Exception as exc:
            raise RuntimeError(
                "Could not load CounselBench-Eval. Install optional dependencies with "
                "`pip install -e '.[hf]'` or install a parquet engine such as pyarrow."
            ) from exc


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _json_default(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
    except ImportError:
        pass
    return str(value)


def _mean_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _yes_rate(values: pd.Series) -> float | None:
    usable = values[values.isin(["Yes", "No"])]
    if usable.empty:
        return None
    return float((usable == "Yes").mean())


def _normalise_likert(mean_value: float | None, low: float, high: float) -> float | None:
    if mean_value is None:
        return None
    return max(0.0, min(1.0, (mean_value - low) / (high - low)))


def _weighted_available_score(components: dict[str, float | None]) -> float | None:
    weighted_sum = 0.0
    weight_sum = 0.0
    for name, weight in COMPOSITE_WEIGHTS.items():
        value = components.get(name)
        if value is None:
            continue
        weighted_sum += weight * value
        weight_sum += weight
    if weight_sum == 0:
        return None
    return weighted_sum / weight_sum


def aggregate_expert_scores(frame: pd.DataFrame, reward_mode: RewardMode = "composite") -> pd.DataFrame:
    """Aggregate five expert annotations per (questionID, responder)."""
    required = {
        "questionID",
        "questionTitle",
        "questionText",
        "response",
        "topic",
        "responder",
        "survey_id",
        "overall_score",
        "overall_reason",
        "empathy_score",
        "specificity_score",
        "medical_advice_score",
        "medical_copy",
        "medical_reason",
        "factual_consistency_score",
        "factual_copy",
        "factual_reason",
        "toxicity_score",
        "toxicity_copy",
        "toxicity_reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CounselBench-Eval frame is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for (question_id, responder), group in frame.groupby(["questionID", "responder"], sort=True):
        overall_mean = _mean_or_none(group["overall_score"])
        empathy_mean = _mean_or_none(group["empathy_score"])
        specificity_mean = _mean_or_none(group["specificity_score"])
        factual_mean = _mean_or_none(group["factual_consistency_score"])
        toxicity_mean = _mean_or_none(group["toxicity_score"])
        medical_yes_rate = _yes_rate(group["medical_advice_score"])

        components = {
            "overall_01": _normalise_likert(overall_mean, 1, 5),
            "empathy_01": _normalise_likert(empathy_mean, 1, 5),
            "specificity_01": _normalise_likert(specificity_mean, 1, 5),
            "factual_01": _normalise_likert(factual_mean, 1, 4),
            "toxicity_safe_01": (
                None
                if toxicity_mean is None
                else 1.0 - float(_normalise_likert(toxicity_mean, 1, 5))
            ),
            "medical_safe_01": None if medical_yes_rate is None else 1.0 - medical_yes_rate,
        }
        if reward_mode == "overall":
            expert_reward = components["overall_01"]
        elif reward_mode == "composite":
            expert_reward = _weighted_available_score(components)
        else:
            raise ValueError(f"Unsupported reward_mode: {reward_mode}")

        survey_ids = [int(value) for value in group["survey_id"].dropna().tolist()]
        rows.append(
            {
                "questionID": question_id,
                "responder": responder,
                "questionTitle": group["questionTitle"].iloc[0],
                "questionText": group["questionText"].iloc[0],
                "topic": group["topic"].iloc[0],
                "response": group["response"].iloc[0],
                "overall_mean": overall_mean,
                "empathy_mean": empathy_mean,
                "specificity_mean": specificity_mean,
                "factual_mean": factual_mean,
                "toxicity_mean": toxicity_mean,
                "medical_advice_yes_rate": medical_yes_rate,
                "expert_reward": expert_reward,
                "reward_mode": reward_mode,
                "n_annotations": int(len(group)),
                "survey_ids": survey_ids,
                "annotation_summary": {
                    "overall_reasons": group["overall_reason"].dropna().tolist(),
                    "medical_reasons": group["medical_reason"].dropna().tolist(),
                    "medical_spans": group["medical_copy"].dropna().tolist(),
                    "factual_reasons": group["factual_reason"].dropna().tolist(),
                    "factual_spans": group["factual_copy"].dropna().tolist(),
                    "toxicity_reasons": group["toxicity_reason"].dropna().tolist(),
                    "toxicity_spans": group["toxicity_copy"].dropna().tolist(),
                },
            }
        )
    return pd.DataFrame(rows)


def _context(row: pd.Series) -> str:
    return (
        f"Topic: {row['topic']}\n"
        f"Question title: {row['questionTitle']}\n\n"
        f"Patient question:\n{row['questionText']}"
    )


def _rubric(row: pd.Series, role: str) -> str:
    return json.dumps(
        {
            "rubric_version": "counselbench_expert_composite_v1",
            "role": role,
            "reward_mode": row["reward_mode"],
            "score": row["expert_reward"],
            "dimensions": {
                "overall_mean": row["overall_mean"],
                "empathy_mean": row["empathy_mean"],
                "specificity_mean": row["specificity_mean"],
                "factual_consistency_mean": row["factual_mean"],
                "toxicity_mean": row["toxicity_mean"],
                "medical_advice_yes_rate": row["medical_advice_yes_rate"],
            },
            "normalization": {
                "overall_01": _normalise_likert(_safe_float(row["overall_mean"]), 1, 5),
                "empathy_01": _normalise_likert(_safe_float(row["empathy_mean"]), 1, 5),
                "specificity_01": _normalise_likert(_safe_float(row["specificity_mean"]), 1, 5),
                "factual_01": _normalise_likert(_safe_float(row["factual_mean"]), 1, 4),
                "toxicity_safe_01": (
                    None
                    if _safe_float(row["toxicity_mean"]) is None
                    else 1.0 - float(_normalise_likert(_safe_float(row["toxicity_mean"]), 1, 5))
                ),
                "medical_safe_01": (
                    None
                    if _safe_float(row["medical_advice_yes_rate"]) is None
                    else 1.0 - float(row["medical_advice_yes_rate"])
                ),
            },
            "weights": COMPOSITE_WEIGHTS if row["reward_mode"] == "composite" else {"overall_01": 1.0},
            "n_annotations": row["n_annotations"],
            "survey_ids": row["survey_ids"],
            "annotation_summary": row["annotation_summary"],
        },
        ensure_ascii=False,
        default=_json_default,
    )


def _canonical_record(row: pd.Series) -> dict[str, Any]:
    record = {
        "example_id": stable_example_id("counselbench_eval", "test", row["questionID"]),
        "source": "counselbench_eval",
        "split": "test",
        "source_id": row["questionID"],
        "x_patient_context": _context(row),
        "a_clinician": row["response"],
        "e_action_repr": None,
        "y_score": row["expert_reward"],
        "y_rubric": _rubric(row, role="human_baseline"),
        "y_source": "counselbench:mental_health_expert_mean",
        "dialogue": "",
        "note": "",
        "section_header": "single_turn_mental_health_qa",
        "section_text": row["response"],
        "time_zero_policy": (
            "X contains only the public anonymous mental-health question and topic; "
            "the evaluated response is excluded from X."
        ),
        "extraction_method": "counselbench_eval_grouped_expert_annotations",
        "inclusion_status": "included",
        "exclusion_reasons": [],
        "metadata": {
            "topic": row["topic"],
            "question_title": row["questionTitle"],
            "responder": row["responder"],
            "n_annotations": row["n_annotations"],
            "dataset_url": "https://huggingface.co/datasets/izi-ano/CounselBench-Eval",
        },
    }
    return {column: record.get(column) for column in CANONICAL_COLUMNS}


def _paired_record(human: pd.Series, target: pd.Series) -> dict[str, Any]:
    record = _canonical_record(human)
    target_responder = str(target["responder"])
    record["a_agent"] = target["response"]
    record["y_agent_score"] = target["expert_reward"]
    record["y_agent_rubric"] = _rubric(target, role=f"target_{target_responder}")
    record["y_agent_source"] = "counselbench:mental_health_expert_mean"
    record["target_policy"] = f"counselbench_logged_responder:{target_responder}"
    record["target_responder"] = target_responder
    record["metadata"] = {
        **record["metadata"],
        "target_responder": target_responder,
        "human_response": human["response"],
        "target_response": target["response"],
    }
    return record


def build_counselbench_dataset(
    output_dir: Path = Path("data/counselbench"),
    reward_mode: RewardMode = "composite",
    bootstrap: int = 1000,
    seed: int = 20260509,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    raw = frame if frame is not None else _load_counselbench_eval_frame()
    aggregated = aggregate_expert_scores(raw, reward_mode=reward_mode)

    humans = aggregated[aggregated["responder"] == "human"].set_index("questionID", drop=False)
    if len(humans) != 100:
        raise ValueError(f"Expected 100 human baseline questions, found {len(humans)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    phase2_records = [_canonical_record(row) for _, row in humans.sort_index().iterrows()]
    write_jsonl(phase2_records, output_dir / "phase2_dataset.jsonl")

    aggregated_path = output_dir / "counselbench_eval_aggregated.csv"
    aggregated.drop(columns=["annotation_summary"]).to_csv(aggregated_path, index=False)

    target_files: dict[str, str] = {}
    effect_files: dict[str, str] = {}
    target_counts: dict[str, int] = {}
    for responder in TARGET_RESPONDERS:
        target = aggregated[aggregated["responder"] == responder].set_index("questionID", drop=False)
        common_ids = sorted(set(humans.index) & set(target.index))
        records = [_paired_record(humans.loc[qid], target.loc[qid]) for qid in common_ids]
        target_path = output_dir / f"phase3_{responder}_expert_scored.jsonl"
        write_jsonl(records, target_path)
        target_files[responder] = str(target_path)
        target_counts[responder] = len(records)

        effect_path = output_dir / f"ground_truth_effect_{responder}.json"
        compute_ground_truth_effect(
            input_path=target_path,
            output_path=effect_path,
            clinician_score_field="y_score",
            agent_score_field="y_agent_score",
            bootstrap=bootstrap,
            seed=seed,
        )
        effect_files[responder] = str(effect_path)

    responder_counts = Counter(aggregated["responder"].tolist())
    manifest = {
        "source": COUNSELBENCH_EVAL_REPO,
        "output_dir": str(output_dir),
        "reward_mode": reward_mode,
        "reward_formula": (
            "composite weighted normalized expert dimensions"
            if reward_mode == "composite"
            else "normalized expert overall score only"
        ),
        "composite_weights": COMPOSITE_WEIGHTS if reward_mode == "composite" else None,
        "raw_rows": int(len(raw)),
        "aggregated_rows": int(len(aggregated)),
        "human_baseline_rows": int(len(phase2_records)),
        "target_counts": target_counts,
        "responder_counts": dict(responder_counts),
        "files": {
            "phase2": str(output_dir / "phase2_dataset.jsonl"),
            "aggregated_csv": str(aggregated_path),
            "phase3_targets": target_files,
            "effects": effect_files,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
