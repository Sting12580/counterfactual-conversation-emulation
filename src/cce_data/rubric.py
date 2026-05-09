from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cce_data.build import write_jsonl


DEFAULT_RUBRIC_CONFIG = Path("configs/rubric_judge.yaml")


def load_rubric_config(path: Path = DEFAULT_RUBRIC_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _validate_score_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    score = float(parsed["score"])
    if score < 0 or score > 1:
        raise ValueError(f"Judge score must be in [0,1], got {score}")
    parsed["score"] = score
    return parsed


def _write_dynamic_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def score_with_openai(
    record: dict[str, Any],
    action_text: str,
    model: str,
    rubric_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install judge dependencies with: pip install -e '.[judge]'") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")

    client = OpenAI()
    user_prompt = rubric_config["user_prompt_template"].format(
        x_patient_context=record.get("x_patient_context", ""),
        a_clinician=action_text,
    )
    system_prompt = rubric_config["system_prompt"]

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", "")
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""

    return _validate_score_payload(_extract_json(text))


def score_dataset(
    input_path: Path = Path("data/processed/phase2_dataset.jsonl"),
    output_path: Path = Path("data/processed/phase2_dataset_scored.jsonl"),
    provider: str = "none",
    model: str | None = None,
    rubric_config_path: Path = DEFAULT_RUBRIC_CONFIG,
    action_field: str = "a_clinician",
    score_field: str = "y_score",
    rubric_field: str = "y_rubric",
    source_field: str = "y_source",
    limit: int | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    rubric_config = load_rubric_config(rubric_config_path)
    model = model or rubric_config.get("recommended_default_model", "gpt-4.1-mini")

    records: list[dict[str, Any]] = []
    scored = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            record = json.loads(line)
            if limit is not None and scored >= limit:
                records.append(record)
                continue
            action_text = record.get(action_field)
            if (
                record.get("inclusion_status") == "included"
                and action_text
                and record.get(score_field) is None
            ):
                if provider == "none":
                    record[source_field] = "needs_scoring"
                    record[rubric_field] = json.dumps(
                        {
                            "rubric_version": rubric_config.get("rubric_version"),
                            "status": "not_scored_provider_none",
                            "action_field": action_field,
                        },
                        ensure_ascii=False,
                    )
                elif provider == "openai":
                    result = score_with_openai(
                        record,
                        action_text=str(action_text),
                        model=model,
                        rubric_config=rubric_config,
                    )
                    result["rubric_version"] = rubric_config.get("rubric_version")
                    result["action_field"] = action_field
                    record[score_field] = float(result["score"])
                    record[rubric_field] = json.dumps(result, ensure_ascii=False)
                    record[source_field] = f"openai:{model}"
                else:
                    raise ValueError(f"Unsupported scoring provider: {provider}")
                scored += 1
            records.append(record)

    write_jsonl(records, output_path)
    csv_path = output_path.with_suffix(".csv")
    _write_dynamic_csv(records, csv_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "csv": str(csv_path),
        "provider": provider,
        "model": model,
        "rubric_config": str(rubric_config_path),
        "rubric_version": rubric_config.get("rubric_version"),
        "action_field": action_field,
        "score_field": score_field,
        "scored_or_marked": scored,
    }
