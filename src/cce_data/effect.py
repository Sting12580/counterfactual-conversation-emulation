from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("No values for quantile")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def compute_ground_truth_effect(
    input_path: Path,
    output_path: Path,
    clinician_score_field: str = "y_score",
    agent_score_field: str = "y_agent_score",
    bootstrap: int = 1000,
    seed: int = 20260509,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Scored dataset not found: {input_path}")

    paired: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            clinician_score = record.get(clinician_score_field)
            agent_score = record.get(agent_score_field)
            if clinician_score is None or agent_score is None:
                continue
            paired.append(
                {
                    "example_id": record.get("example_id"),
                    "source": record.get("source"),
                    "split": record.get("split"),
                    "clinician": float(clinician_score),
                    "agent": float(agent_score),
                    "diff": float(agent_score) - float(clinician_score),
                }
            )

    if not paired:
        raise ValueError(
            f"No paired rows with {clinician_score_field} and {agent_score_field} in {input_path}"
        )

    clinician_scores = [row["clinician"] for row in paired]
    agent_scores = [row["agent"] for row in paired]
    diffs = [row["diff"] for row in paired]
    rng = random.Random(seed)
    boot_diffs: list[float] = []
    n = len(paired)
    for _ in range(bootstrap):
        sample = [paired[rng.randrange(n)] for _ in range(n)]
        boot_diffs.append(statistics.mean(row["diff"] for row in sample))

    result = {
        "input": str(input_path),
        "n_paired": n,
        "clinician_score_field": clinician_score_field,
        "agent_score_field": agent_score_field,
        "v_true_pi_b": statistics.mean(clinician_scores),
        "v_true_pi_agent": statistics.mean(agent_scores),
        "true_effect": statistics.mean(diffs),
        "bootstrap": {
            "n": bootstrap,
            "seed": seed,
            "ci95": [_quantile(boot_diffs, 0.025), _quantile(boot_diffs, 0.975)],
        },
        "summary": {
            "clinician_min": min(clinician_scores),
            "clinician_max": max(clinician_scores),
            "agent_min": min(agent_scores),
            "agent_max": max(agent_scores),
            "direction_agent_better_rate": sum(diff > 0 for diff in diffs) / n,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
