from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from cce_data.build import write_csv, write_jsonl


RUBRIC_SYSTEM = """You are a careful clinical evaluation assistant.
Score a clinician's final diagnosis and management plan for a patient context.
This is research annotation, not medical advice.
Return only valid JSON with keys: score, rationale, dimensions."""

RUBRIC_USER_TEMPLATE = """Patient context available at time zero:
{x_patient_context}

Clinician final output to score:
{a_clinician}

Rubric:
- Diagnostic appropriateness and differential reasoning.
- Management-plan appropriateness and safety.
- Patient-specific use of available context.
- Clear communication of uncertainty, follow-up, and red flags when relevant.
- Penalize hallucinated or unsupported claims.

Return JSON:
{{
  "score": <float from 0.0 to 1.0>,
  "rationale": "<brief rationale>",
  "dimensions": {{
    "diagnosis": <0.0-1.0>,
    "management": <0.0-1.0>,
    "context_use": <0.0-1.0>,
    "safety": <0.0-1.0>
  }}
}}"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def score_with_openai(record: dict[str, Any], model: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install judge dependencies with: pip install -e '.[judge]'") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")

    client = OpenAI()
    user_prompt = RUBRIC_USER_TEMPLATE.format(
        x_patient_context=record.get("x_patient_context", ""),
        a_clinician=record.get("a_clinician", ""),
    )

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": RUBRIC_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = getattr(response, "output_text", "")
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": RUBRIC_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or ""

    parsed = _extract_json(text)
    score = float(parsed["score"])
    if score < 0 or score > 1:
        raise ValueError(f"Judge score must be in [0,1], got {score}")
    return parsed


def score_dataset(
    input_path: Path = Path("data/processed/phase2_dataset.jsonl"),
    output_path: Path = Path("data/processed/phase2_dataset_scored.jsonl"),
    provider: str = "none",
    model: str = "gpt-4.1-mini",
    limit: int | None = None,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    records: list[dict[str, Any]] = []
    scored = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            record = json.loads(line)
            if limit is not None and scored >= limit:
                records.append(record)
                continue
            if record.get("inclusion_status") == "included" and record.get("y_score") is None:
                if provider == "none":
                    record["y_source"] = "needs_scoring"
                    record["y_rubric"] = "not_scored_provider_none"
                elif provider == "openai":
                    result = score_with_openai(record, model=model)
                    record["y_score"] = float(result["score"])
                    record["y_rubric"] = json.dumps(result, ensure_ascii=False)
                    record["y_source"] = f"openai:{model}"
                else:
                    raise ValueError(f"Unsupported scoring provider: {provider}")
                scored += 1
            records.append(record)

    write_jsonl(records, output_path)
    csv_path = output_path.with_suffix(".csv")
    write_csv(records, csv_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "csv": str(csv_path),
        "provider": provider,
        "model": model,
        "scored_or_marked": scored,
    }
