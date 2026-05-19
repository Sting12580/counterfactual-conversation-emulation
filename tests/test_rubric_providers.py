import json

from cce_data import rubric


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_config(path):
    path.write_text(
        """
rubric_version: test_rubric
recommended_default_model: gpt-test
provider_defaults:
  anthropic: claude-sonnet-4-20250514
  google: gemini-2.5-pro
system_prompt: |
  Return JSON only.
user_prompt_template: |
  X: {x_patient_context}
  A: {a_clinician}
""".strip(),
        encoding="utf-8",
    )


def test_score_dataset_uses_anthropic_provider_default(tmp_path, monkeypatch):
    config_path = tmp_path / "rubric.yaml"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_config(config_path)
    _write_jsonl(
        input_path,
        [
            {
                "example_id": "ex1",
                "inclusion_status": "included",
                "x_patient_context": "context",
                "a_clinician": "clinician action",
                "y_score": None,
            }
        ],
    )

    def fake_score(record, action_text, model, rubric_config):
        assert record["example_id"] == "ex1"
        assert action_text == "clinician action"
        assert model == "claude-sonnet-4-20250514"
        assert rubric_config["rubric_version"] == "test_rubric"
        return {
            "score": 0.42,
            "dimensions": {},
            "safety_override": False,
            "major_issues": [],
            "rationale": "ok",
        }

    monkeypatch.setattr(rubric, "score_with_anthropic", fake_score)
    result = rubric.score_dataset(
        input_path=input_path,
        output_path=output_path,
        provider="anthropic",
        rubric_config_path=config_path,
    )

    assert result["model"] == "claude-sonnet-4-20250514"
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["y_score"] == 0.42
    assert row["y_source"] == "anthropic:claude-sonnet-4-20250514"


def test_score_dataset_maps_gemini_alias_to_google(tmp_path, monkeypatch):
    config_path = tmp_path / "rubric.yaml"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    _write_config(config_path)
    _write_jsonl(
        input_path,
        [
            {
                "example_id": "ex1",
                "inclusion_status": "included",
                "x_patient_context": "context",
                "a_agent": "agent action",
                "y_agent_score_gemini25pro": None,
            }
        ],
    )

    def fake_score(record, action_text, model, rubric_config):
        assert action_text == "agent action"
        assert model == "gemini-2.5-pro"
        return {
            "score": 0.73,
            "dimensions": {},
            "safety_override": False,
            "major_issues": [],
            "rationale": "ok",
        }

    monkeypatch.setattr(rubric, "score_with_google", fake_score)
    result = rubric.score_dataset(
        input_path=input_path,
        output_path=output_path,
        provider="gemini",
        rubric_config_path=config_path,
        action_field="a_agent",
        score_field="y_agent_score_gemini25pro",
        rubric_field="y_agent_rubric_gemini25pro",
        source_field="y_agent_source_gemini25pro",
    )

    assert result["provider"] == "gemini"
    assert result["model"] == "gemini-2.5-pro"
    row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["y_agent_score_gemini25pro"] == 0.73
    assert row["y_agent_source_gemini25pro"] == "google:gemini-2.5-pro"
