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


RUBRIC_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "dimensions": {
            "type": "object",
            "properties": {
                "diagnostic_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "management_quality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "safety_and_escalation": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "context_use_and_factuality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "communication_and_uncertainty": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": [
                "diagnostic_quality",
                "management_quality",
                "safety_and_escalation",
                "context_use_and_factuality",
                "communication_and_uncertainty",
            ],
        },
        "safety_override": {"type": "boolean"},
        "major_issues": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["score", "dimensions", "safety_override", "major_issues", "rationale"],
}


def _write_dynamic_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _build_prompts(
    record: dict[str, Any],
    action_text: str,
    rubric_config: dict[str, Any],
) -> tuple[str, str]:
    user_prompt = rubric_config["user_prompt_template"].format(
        x_patient_context=record.get("x_patient_context", ""),
        a_clinician=action_text,
    )
    return rubric_config["system_prompt"], user_prompt


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
    system_prompt, user_prompt = _build_prompts(record, action_text, rubric_config)

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


def _anthropic_text(response: Any) -> str:
    chunks: list[str] = []
    for block in getattr(response, "content", []) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
        else:
            text = getattr(block, "text", "")
            if text:
                chunks.append(str(text))
    return "".join(chunks).strip()


def score_with_anthropic(
    record: dict[str, Any],
    action_text: str,
    model: str,
    rubric_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("Install judge dependencies with: pip install -e '.[judge]'") from exc

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")

    client = Anthropic()
    system_prompt, user_prompt = _build_prompts(record, action_text, rubric_config)
    response = client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return _validate_score_payload(_extract_json(_anthropic_text(response)))


def _google_text(response: Any) -> str:
    text = getattr(response, "text", "")
    if text:
        return str(text).strip()
    chunks: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", "")
            if part_text:
                chunks.append(str(part_text))
    return "".join(chunks).strip()


def score_with_google(
    record: dict[str, Any],
    action_text: str,
    model: str,
    rubric_config: dict[str, Any],
) -> dict[str, Any]:
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install judge dependencies with: pip install -e '.[judge]'") from exc

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for provider=google")

    client = genai.Client()
    system_prompt, user_prompt = _build_prompts(record, action_text, rubric_config)
    response = client.models.generate_content(
        model=model,
        contents=f"{system_prompt}\n\n{user_prompt}",
        config={
            "response_mime_type": "application/json",
            "response_json_schema": RUBRIC_JSON_SCHEMA,
        },
    )
    return _validate_score_payload(_extract_json(_google_text(response)))


def _canonical_provider(provider: str) -> str:
    if provider == "gemini":
        return "google"
    return provider


def _default_model_for_provider(rubric_config: dict[str, Any], provider: str) -> str:
    provider_defaults = rubric_config.get("provider_defaults", {})
    canonical_provider = _canonical_provider(provider)
    if canonical_provider in provider_defaults:
        return str(provider_defaults[canonical_provider])
    return str(rubric_config.get("recommended_default_model", "gpt-4.1-mini"))


def _score_with_provider(
    provider: str,
    record: dict[str, Any],
    action_text: str,
    model: str,
    rubric_config: dict[str, Any],
) -> dict[str, Any]:
    canonical_provider = _canonical_provider(provider)
    if canonical_provider == "openai":
        return score_with_openai(
            record,
            action_text=action_text,
            model=model,
            rubric_config=rubric_config,
        )
    if canonical_provider == "anthropic":
        return score_with_anthropic(
            record,
            action_text=action_text,
            model=model,
            rubric_config=rubric_config,
        )
    if canonical_provider == "google":
        return score_with_google(
            record,
            action_text=action_text,
            model=model,
            rubric_config=rubric_config,
        )
    raise ValueError(f"Unsupported scoring provider: {provider}")


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
    canonical_provider = _canonical_provider(provider)
    model = model or _default_model_for_provider(rubric_config, provider)

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
                elif canonical_provider in {"openai", "anthropic", "google"}:
                    result = _score_with_provider(
                        canonical_provider,
                        record,
                        action_text=str(action_text),
                        model=model,
                        rubric_config=rubric_config,
                    )
                    result["rubric_version"] = rubric_config.get("rubric_version")
                    result["action_field"] = action_field
                    record[score_field] = float(result["score"])
                    record[rubric_field] = json.dumps(result, ensure_ascii=False)
                    record[source_field] = f"{canonical_provider}:{model}"
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
