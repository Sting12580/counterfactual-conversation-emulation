from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

from cce_data.build import write_jsonl
from cce_data.effect import compute_ground_truth_effect
from cce_data.schema import CANONICAL_COLUMNS, stable_example_id


AYERS_ASKDOCS_DATAVERSE_URL = "https://doi.org/10.7910/DVN/BZARC3"
AYERS_ASKDOCS_CSV_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/7001793?format=original"
)

QUALITY_COLUMNS = {
    "physician": [
        "Eval 1 Quality (Physician)",
        "Eval 2 Quality (Physician)",
        "Eval 3 Quality (Physician)",
    ],
    "chatgpt": [
        "Eval 1 Quality (ChatGPT)",
        "Eval 2 Quality (ChatGPT)",
        "Eval 3 Quality (ChatGPT)",
    ],
}

EMPATHY_COLUMNS = {
    "physician": [
        "Eval 1 Empathy (Physician)",
        "Eval 2 Empathy (Physician)",
        "Eval 3 Empathy (Physician)",
    ],
    "chatgpt": [
        "Eval 1 Empathy (ChatGPT)",
        "Eval 2 Empathy (ChatGPT)",
        "Eval 3 Empathy (ChatGPT)",
    ],
}

PREFERENCE_COLUMNS = [
    "Eval 1 Preference",
    "Eval 2 Preference",
    "Eval 3 Preference",
]

RewardMode = Literal["composite", "quality", "empathy"]


def _load_ayers_askdocs_frame(input_path: Path | None = None) -> pd.DataFrame:
    """Load Ayers et al. AskDocs replication data from a local CSV or Dataverse."""
    source: str | Path | BytesIO
    if input_path is None:
        response = requests.get(
            AYERS_ASKDOCS_CSV_URL,
            headers={"User-Agent": "counterfactual-conversation-emulation/0.1"},
            timeout=60,
        )
        response.raise_for_status()
        source = BytesIO(response.content)
    else:
        source = input_path
    try:
        return pd.read_csv(source, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        # The Dataverse CSV contains a few non-UTF-8 bytes in question text.
        return pd.read_csv(source, encoding="latin1")


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


def _mean_score(row: pd.Series, columns: list[str]) -> float | None:
    values = pd.to_numeric(row[columns], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _normalise_likert(mean_value: float | None, low: float = 1.0, high: float = 5.0) -> float | None:
    if mean_value is None:
        return None
    if math.isnan(mean_value):
        return None
    return max(0.0, min(1.0, (mean_value - low) / (high - low)))


def _reward(quality_mean: float | None, empathy_mean: float | None, reward_mode: RewardMode) -> float:
    quality_01 = _normalise_likert(quality_mean)
    empathy_01 = _normalise_likert(empathy_mean)
    if reward_mode == "quality":
        if quality_01 is None:
            raise ValueError("Missing quality score")
        return quality_01
    if reward_mode == "empathy":
        if empathy_01 is None:
            raise ValueError("Missing empathy score")
        return empathy_01
    if reward_mode != "composite":
        raise ValueError(f"Unsupported reward_mode: {reward_mode}")

    components = [score for score in [quality_01, empathy_01] if score is not None]
    if not components:
        raise ValueError("Missing quality and empathy scores")
    return float(sum(components) / len(components))


def _preference_counts(row: pd.Series) -> dict[str, Any]:
    preferences = [str(row[column]) for column in PREFERENCE_COLUMNS]
    chatgpt = sum(value == "ChatGPT" for value in preferences)
    physician = sum(value == "Physician" for value in preferences)
    return {
        "raw": preferences,
        "chatgpt_count": chatgpt,
        "physician_count": physician,
        "chatgpt_rate": chatgpt / len(PREFERENCE_COLUMNS),
        "physician_rate": physician / len(PREFERENCE_COLUMNS),
    }


def aggregate_human_scores(frame: pd.DataFrame, reward_mode: RewardMode = "composite") -> pd.DataFrame:
    """Aggregate the three healthcare-professional ratings already stored per row."""
    required = {
        "postID",
        "Question",
        "submissionID",
        "commentID",
        "Physician Response",
        "ChatGPT Response",
        "Physician Length",
        "ChatGPT Length",
        *PREFERENCE_COLUMNS,
        *QUALITY_COLUMNS["physician"],
        *QUALITY_COLUMNS["chatgpt"],
        *EMPATHY_COLUMNS["physician"],
        *EMPATHY_COLUMNS["chatgpt"],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Ayers AskDocs frame is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        physician_quality_mean = _mean_score(row, QUALITY_COLUMNS["physician"])
        physician_empathy_mean = _mean_score(row, EMPATHY_COLUMNS["physician"])
        chatgpt_quality_mean = _mean_score(row, QUALITY_COLUMNS["chatgpt"])
        chatgpt_empathy_mean = _mean_score(row, EMPATHY_COLUMNS["chatgpt"])

        rows.append(
            {
                "postID": str(row["postID"]),
                "submissionID": str(row["submissionID"]),
                "commentID": str(row["commentID"]),
                "Question": row["Question"],
                "Physician Response": row["Physician Response"],
                "ChatGPT Response": row["ChatGPT Response"],
                "Physician Length": int(row["Physician Length"]),
                "ChatGPT Length": int(row["ChatGPT Length"]),
                "reward_mode": reward_mode,
                "n_evaluators": 3,
                "physician_quality_mean": physician_quality_mean,
                "physician_empathy_mean": physician_empathy_mean,
                "physician_quality_01": _normalise_likert(physician_quality_mean),
                "physician_empathy_01": _normalise_likert(physician_empathy_mean),
                "physician_reward": _reward(
                    physician_quality_mean,
                    physician_empathy_mean,
                    reward_mode=reward_mode,
                ),
                "chatgpt_quality_mean": chatgpt_quality_mean,
                "chatgpt_empathy_mean": chatgpt_empathy_mean,
                "chatgpt_quality_01": _normalise_likert(chatgpt_quality_mean),
                "chatgpt_empathy_01": _normalise_likert(chatgpt_empathy_mean),
                "chatgpt_reward": _reward(
                    chatgpt_quality_mean,
                    chatgpt_empathy_mean,
                    reward_mode=reward_mode,
                ),
                "preference": _preference_counts(row),
                "raw_scores": {
                    "physician_quality": [int(row[column]) for column in QUALITY_COLUMNS["physician"]],
                    "physician_empathy": [int(row[column]) for column in EMPATHY_COLUMNS["physician"]],
                    "chatgpt_quality": [int(row[column]) for column in QUALITY_COLUMNS["chatgpt"]],
                    "chatgpt_empathy": [int(row[column]) for column in EMPATHY_COLUMNS["chatgpt"]],
                },
            }
        )
    return pd.DataFrame(rows)


def _context(row: pd.Series) -> str:
    return f"Patient question:\n{row['Question']}"


def _rubric(row: pd.Series, role: str) -> str:
    prefix = "physician" if role == "physician" else "chatgpt"
    preference = row["preference"]
    return json.dumps(
        {
            "rubric_version": "ayers_askdocs_healthcare_professional_quality_empathy_v1",
            "role": role,
            "reward_mode": row["reward_mode"],
            "score": row[f"{prefix}_reward"],
            "dimensions": {
                "quality_mean": row[f"{prefix}_quality_mean"],
                "empathy_mean": row[f"{prefix}_empathy_mean"],
            },
            "normalization": {
                "quality_01": row[f"{prefix}_quality_01"],
                "empathy_01": row[f"{prefix}_empathy_01"],
            },
            "weights": (
                {"quality_01": 0.5, "empathy_01": 0.5}
                if row["reward_mode"] == "composite"
                else {f"{row['reward_mode']}_01": 1.0}
            ),
            "n_evaluators": row["n_evaluators"],
            "raw_scores": row["raw_scores"],
            "preference": {
                "raw": preference["raw"],
                "chatgpt_count": preference["chatgpt_count"],
                "physician_count": preference["physician_count"],
                "selected_rate": (
                    preference["physician_rate"]
                    if role == "physician"
                    else preference["chatgpt_rate"]
                ),
            },
        },
        ensure_ascii=False,
        default=_json_default,
    )


def _canonical_record(row: pd.Series) -> dict[str, Any]:
    record = {
        "example_id": stable_example_id("ayers_askdocs_2023", "test", row["postID"]),
        "source": "ayers_askdocs_2023",
        "split": "test",
        "source_id": row["postID"],
        "x_patient_context": _context(row),
        "a_clinician": row["Physician Response"],
        "e_action_repr": None,
        "y_score": row["physician_reward"],
        "y_rubric": _rubric(row, role="physician"),
        "y_source": "ayers_askdocs_2023:healthcare_professional_mean",
        "dialogue": "",
        "note": "",
        "section_header": "single_turn_askdocs_medical_qa",
        "section_text": row["Physician Response"],
        "time_zero_policy": (
            "X contains only the anonymous patient question from Reddit AskDocs; "
            "the physician and ChatGPT responses are excluded from X."
        ),
        "extraction_method": "ayers_askdocs_replication_data_human_annotations",
        "inclusion_status": "included",
        "exclusion_reasons": [],
        "metadata": {
            "submission_id": row["submissionID"],
            "comment_id": row["commentID"],
            "physician_length": row["Physician Length"],
            "chatgpt_length": row["ChatGPT Length"],
            "n_evaluators": row["n_evaluators"],
            "dataset_url": AYERS_ASKDOCS_DATAVERSE_URL,
            "datafile_url": AYERS_ASKDOCS_CSV_URL,
        },
    }
    return {column: record.get(column) for column in CANONICAL_COLUMNS}


def _paired_record(row: pd.Series) -> dict[str, Any]:
    record = _canonical_record(row)
    record["a_agent"] = row["ChatGPT Response"]
    record["y_agent_score"] = row["chatgpt_reward"]
    record["y_agent_rubric"] = _rubric(row, role="chatgpt")
    record["y_agent_source"] = "ayers_askdocs_2023:healthcare_professional_mean"
    record["target_policy"] = "ayers_askdocs_logged_responder:chatgpt"
    record["target_responder"] = "chatgpt"
    record["metadata"] = {
        **record["metadata"],
        "preference": row["preference"],
        "human_response": row["Physician Response"],
        "target_response": row["ChatGPT Response"],
    }
    return record


def build_ayers_askdocs_dataset(
    output_dir: Path = Path("data/ayers_askdocs"),
    input_path: Path | None = None,
    reward_mode: RewardMode = "composite",
    bootstrap: int = 1000,
    seed: int = 20260509,
    frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    raw = frame if frame is not None else _load_ayers_askdocs_frame(input_path=input_path)
    aggregated = aggregate_human_scores(raw, reward_mode=reward_mode)

    if len(aggregated) != 195:
        raise ValueError(f"Expected 195 Ayers AskDocs rows, found {len(aggregated)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = aggregated.sort_values("postID")
    phase2_records = [_canonical_record(row) for _, row in ordered.iterrows()]
    paired_records = [_paired_record(row) for _, row in ordered.iterrows()]

    phase2_path = output_dir / "phase2_dataset.jsonl"
    target_path = output_dir / "phase3_chatgpt_expert_scored.jsonl"
    write_jsonl(phase2_records, phase2_path)
    write_jsonl(paired_records, target_path)

    aggregated_path = output_dir / "ayers_askdocs_aggregated.csv"
    aggregated.drop(columns=["preference", "raw_scores"]).to_csv(aggregated_path, index=False)

    effect_path = output_dir / "ground_truth_effect_chatgpt.json"
    effect = compute_ground_truth_effect(
        input_path=target_path,
        output_path=effect_path,
        clinician_score_field="y_score",
        agent_score_field="y_agent_score",
        bootstrap=bootstrap,
        seed=seed,
    )

    manifest = {
        "source": "Ayers et al. 2023 AskDocs physician vs ChatGPT replication data",
        "source_url": AYERS_ASKDOCS_DATAVERSE_URL,
        "datafile_url": AYERS_ASKDOCS_CSV_URL,
        "input_path": str(input_path) if input_path is not None else AYERS_ASKDOCS_CSV_URL,
        "output_dir": str(output_dir),
        "reward_mode": reward_mode,
        "reward_formula": (
            "mean(normalized quality, normalized empathy)"
            if reward_mode == "composite"
            else f"normalized {reward_mode} only"
        ),
        "normalization": "Likert 1-5 mapped to [0, 1] by (score - 1) / 4",
        "raw_rows": int(len(raw)),
        "paired_rows": int(len(paired_records)),
        "human_baseline_rows": int(len(phase2_records)),
        "target_counts": {"chatgpt": int(len(paired_records))},
        "effect": effect,
        "files": {
            "phase2": str(phase2_path),
            "aggregated_csv": str(aggregated_path),
            "phase3_targets": {"chatgpt": str(target_path)},
            "effects": {"chatgpt": str(effect_path)},
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
