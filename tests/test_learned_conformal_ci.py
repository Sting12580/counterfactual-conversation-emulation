from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from cce_data.estimators.learned_embedding import LearnedEmbeddingConfig

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compute_learned_conformal_ci.py"
)
SPEC = importlib.util.spec_from_file_location("compute_learned_conformal_ci", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
learned_conformal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = learned_conformal
SPEC.loader.exec_module(learned_conformal)

compute_learned_conformal_report = learned_conformal.compute_learned_conformal_report
remap_score_fields = learned_conformal.remap_score_fields


def test_remap_score_fields_uses_selected_judge_columns() -> None:
    records = [
        {
            "y_score": 0.1,
            "y_agent_score": None,
            "y_score_claude_sonnet46": 0.7,
            "y_agent_score_claude_sonnet46": 0.8,
        }
    ]

    remapped = remap_score_fields(
        records,
        score_field="y_score_claude_sonnet46",
        agent_score_field="y_agent_score_claude_sonnet46",
    )

    assert remapped[0]["y_score"] == 0.7
    assert remapped[0]["y_agent_score"] == 0.8
    assert records[0]["y_score"] == 0.1


def test_compute_learned_conformal_report_smoke() -> None:
    records = []
    for i in range(30):
        records.append(
            {
                "inclusion_status": "included",
                "x_patient_context": f"context {i % 5}",
                "a_clinician": f"clinician action {i % 7}",
                "a_agent": f"agent action {i % 7}",
                "y_score": 0.4 + 0.01 * (i % 10),
                "y_agent_score": 0.55 + 0.01 * (i % 10),
            }
        )

    def embed(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for row, text in enumerate(texts):
            seed = sum(ord(ch) for ch in text) % 10000
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=8).astype(np.float32)
            out[row] = vec / (np.linalg.norm(vec) + 1e-9)
        return out

    report = compute_learned_conformal_report(
        records,
        embed,
        LearnedEmbeddingConfig(
            latent_dim=4,
            hidden_dim=8,
            merge_strategy="concat-pca",
            pca_dim=3,
            max_epochs=8,
            validation_fraction=0.0,
            patience=0,
            seed=0,
        ),
        seed=0,
    )

    assert report["n"] == 30
    assert report["learned_embedding"]["uses_agent_rewards"] is False
    assert report["learned_embedding"]["merge_strategy"] == "concat-pca"
    assert set(report["results"]) == {"DM", "MIPS", "OffCEM"}
    for result in report["results"].values():
        assert result["conformal_ci_low"] <= result["v_hat"] <= result["conformal_ci_high"]
        assert result["conformal_half_width"] >= 0
