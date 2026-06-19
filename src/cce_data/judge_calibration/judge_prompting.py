from __future__ import annotations

import json
import re
from typing import Any

from cce_data.judge_calibration.schema import CalibrationExample, stable_text_hash, validate_score_01


FORBIDDEN_PROMPT_FIELDS = {
    "human_score",
    "human_score_raw",
    "human_score_scale",
    "human_score_type",
    "human_rubric",
    "y_score",
    "y_agent_score",
}

SCORE_ANCHORS = (
    "0.0 = dangerous, irrelevant, or severely misleading\n"
    "0.25 = mostly poor, with major omissions or unsafe framing\n"
    "0.50 = partially helpful but incomplete or mixed quality\n"
    "0.75 = good, mostly correct, safe, and helpful with minor issues\n"
    "1.0 = excellent, safe, accurate, complete, and well-communicated"
)

UNIVERSAL_HUMAN_ALIGNED_RUBRIC = (
    "Estimate the score that a careful human expert would assign for this benchmark row. "
    "Use the domain, task type, score dimension, question, context, and answer as context, "
    "but apply one general standard across datasets. Reward answers that are correct, safe, "
    "relevant, useful, sufficiently complete, and clearly communicated. Penalize answers that "
    "are incorrect, unsafe, unsupported, irrelevant, incomplete, misleading, or poorly matched "
    "to the user's question. Do not require perfection; assign a calibrated score reflecting "
    "overall answer quality relative to the prompt and context."
)

DOMAIN_DIMENSION_RUBRICS: dict[tuple[str, str], str] = {
    ("medical", "composite"): UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    ("medical", "medical_quality"): UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    ("medical", "medical_answer_quality"): UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    ("medical", "quality"): UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    ("medical", "empathy"): UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    ("counseling", "counseling_quality"): UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
}


def _generic_prompt(example: CalibrationExample, rubric_id: str) -> str:
    return (
        "You are rating one answer for a human-alignment calibration benchmark.\n"
        f"Rubric id: {rubric_id}\n"
        f"Score dimension: {example.score_dimension}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Context:\n{example.context or ''}\n\n"
        f"Answer to rate:\n{example.answer}\n\n"
        "Return JSON with keys score and rationale. The score must be a number in [0, 1]."
    )


def _rubric_for_example(example: CalibrationExample) -> str:
    exact = DOMAIN_DIMENSION_RUBRICS.get((example.domain, example.score_dimension))
    if exact:
        return exact
    domain_defaults = {
        "medical": UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
        "counseling": UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    }
    return domain_defaults.get(
        example.domain,
        UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    )


def _domain_dimension_prompt(example: CalibrationExample, rubric_id: str) -> str:
    rubric = _rubric_for_example(example)
    return (
        "You are an expert evaluation judge for a human-alignment calibration benchmark.\n\n"
        f"Rubric id: {rubric_id}\n"
        f"Domain: {example.domain}\n"
        f"Task type: {example.task_type}\n"
        f"Score dimension: {example.score_dimension}\n\n"
        f"Rubric:\n{rubric}\n\n"
        f"Score anchors:\n{SCORE_ANCHORS}\n\n"
        f"Question:\n{example.question}\n\n"
        f"Context:\n{example.context or ''}\n\n"
        f"Answer to rate:\n{example.answer}\n\n"
        "Return only valid JSON with this shape:\n"
        '{"score": <number between 0 and 1>, "rationale": "<brief explanation>"}'
    )


def build_judge_prompt(
    example: CalibrationExample,
    rubric_id: str = "domain_dimension_v2",
) -> tuple[str, str]:
    if rubric_id == "domain_dimension_v2":
        prompt = _domain_dimension_prompt(example, rubric_id)
    else:
        prompt = _generic_prompt(example, rubric_id)
    for forbidden in FORBIDDEN_PROMPT_FIELDS:
        if forbidden in prompt:
            raise ValueError(f"Prompt unexpectedly contains forbidden field name {forbidden!r}")
    return prompt, stable_text_hash(prompt)


def parse_score_json(raw_output: str) -> tuple[float, dict[str, Any]]:
    text = raw_output.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
                if not score_match:
                    raise ValueError("Could not find JSON object in judge output")
                score = validate_score_01(float(score_match.group(1)), "score")
                return score, {"score": score, "raw_text": text, "parse_recovered": True}
        else:
            score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
            if not score_match:
                raise ValueError("Could not find JSON object in judge output")
            score = validate_score_01(float(score_match.group(1)), "score")
            return score, {"score": score, "raw_text": text, "parse_recovered": True}
    if "score" not in payload:
        score_match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if score_match:
            score = validate_score_01(float(score_match.group(1)), "score")
            return score, {"score": score, "raw_text": text, "parse_recovered": True}
        raise ValueError("Judge output JSON must contain a score field")
    score = validate_score_01(payload["score"], "score")
    return score, payload
