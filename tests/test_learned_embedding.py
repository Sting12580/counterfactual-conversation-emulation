from __future__ import annotations

import numpy as np

from cce_data.estimators.learned_embedding import (
    LearnedEmbeddingConfig,
    learn_reward_informed_embeddings,
)
from cce_data.estimators.real_runner import run_phase5_headline


def test_learned_embedding_shapes_and_normalization() -> None:
    rng = np.random.default_rng(0)
    n, dim = 32, 10
    phi_x = rng.normal(size=(n, dim)).astype(np.float32)
    phi_a_cl = rng.normal(size=(n, dim)).astype(np.float32)
    phi_a_ag = rng.normal(size=(n, dim)).astype(np.float32)
    y = 0.5 + 0.2 * (phi_x[:, 0] * phi_a_cl[:, 0])

    z_x, z_a_cl, z_a_ag, diag = learn_reward_informed_embeddings(
        phi_x,
        phi_a_cl,
        phi_a_ag,
        y,
        LearnedEmbeddingConfig(
            latent_dim=6,
            hidden_dim=8,
            max_epochs=25,
            validation_fraction=0.0,
            patience=0,
            seed=0,
        ),
    )

    assert z_x.shape == (n, 6)
    assert z_a_cl.shape == (n, 6)
    assert z_a_ag.shape == (n, 6)
    assert np.allclose(np.linalg.norm(z_x, axis=1), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(z_a_cl, axis=1), 1.0, atol=1e-5)
    assert diag["uses_agent_rewards"] is False
    assert diag["feature_dim_after_concat"] == 18


def test_concat_pca_merge_preserves_base_signal_dimension() -> None:
    rng = np.random.default_rng(1)
    n, dim = 36, 12
    phi_x = rng.normal(size=(n, dim)).astype(np.float32)
    phi_a_cl = rng.normal(size=(n, dim)).astype(np.float32)
    phi_a_ag = rng.normal(size=(n, dim)).astype(np.float32)
    y = 0.4 + 0.1 * phi_a_cl[:, 0] + 0.05 * phi_x[:, 1]

    z_x, z_a_cl, z_a_ag, diag = learn_reward_informed_embeddings(
        phi_x,
        phi_a_cl,
        phi_a_ag,
        y,
        LearnedEmbeddingConfig(
            latent_dim=5,
            hidden_dim=8,
            merge_strategy="concat-pca",
            pca_dim=4,
            max_epochs=20,
            validation_fraction=0.0,
            patience=0,
            seed=1,
        ),
    )

    assert z_x.shape == (n, 9)
    assert z_a_cl.shape == (n, 9)
    assert z_a_ag.shape == (n, 9)
    assert diag["merge_strategy"] == "concat-pca"
    assert diag["pca"]["n_components"] == 4
    assert diag["output_embedding_dim"] == 9
    assert diag["feature_dim_after_concat"] == 27
    assert diag["uses_agent_rewards"] is False


def test_phase5_runner_accepts_learned_action_embedding() -> None:
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

    report = run_phase5_headline(
        records,
        embed,
        n_boot=0,
        seed=0,
        learned_embedding=LearnedEmbeddingConfig(
            latent_dim=4,
            hidden_dim=8,
            max_epochs=20,
            validation_fraction=0.0,
            patience=0,
            seed=0,
        ),
    )

    assert report["n"] == 30
    assert set(report["results"]) == {"DM", "MIPS", "OffCEM"}
    assert report["learned_embedding"]["uses_agent_rewards"] is False
    assert report["learned_embedding"]["feature_dim_after_concat"] == 12
