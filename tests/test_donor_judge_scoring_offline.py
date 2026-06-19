from cce_data.judge_calibration.judge_clients import (
    AnthropicJudgeClient,
    GeminiJudgeClient,
    MockJudgeClient,
    OpenAIChatJudgeClient,
    PHASE7_V2_DONOR_JUDGE_IDS,
    _redact_url,
    make_judge_client,
)
from cce_data.judge_calibration.judge_prompting import (
    DOMAIN_DIMENSION_RUBRICS,
    UNIVERSAL_HUMAN_ALIGNED_RUBRIC,
    build_judge_prompt,
    parse_score_json,
)
from cce_data.judge_calibration.schema import CalibrationExample, make_split_group


def _example() -> CalibrationExample:
    return CalibrationExample(
        calib_example_id="ex1",
        dataset_id="d",
        domain="medical",
        task_type="qa",
        score_dimension="quality",
        question_id="q1",
        question="question",
        context="context",
        answer="answer",
        answer_source="chatgpt",
        human_score=0.73,
        human_score_raw=0.73,
        human_score_scale={"min": 0, "max": 1},
        human_score_type="expert_scalar",
        human_rubric="secret label rubric",
        source_file=None,
        split_group=make_split_group("d", "q1", "quality"),
        metadata={},
    )


def _counseling_example() -> CalibrationExample:
    return CalibrationExample(
        calib_example_id="ex2",
        dataset_id="counselbench",
        domain="counseling",
        task_type="counseling_qa",
        score_dimension="counseling_quality",
        question_id="q2",
        question="I feel stuck and overwhelmed.",
        context="",
        answer="That sounds hard. Consider one small step and reach out for support.",
        answer_source="gpt4",
        human_score=0.61,
        human_score_raw=0.61,
        human_score_scale={"min": 0, "max": 1},
        human_score_type="expert_scalar",
        human_rubric="secret rubric",
        source_file=None,
        split_group=make_split_group("counselbench", "q2", "counseling_quality"),
        metadata={},
    )


def test_prompt_excludes_gold_label_fields() -> None:
    prompt, prompt_hash = build_judge_prompt(_example())
    assert "human_score" not in prompt
    assert "0.73" not in prompt
    assert prompt_hash


def test_domain_dimension_v2_prompt_has_universal_human_aligned_rubric_and_anchors() -> None:
    prompt, _ = build_judge_prompt(_example(), rubric_id="domain_dimension_v2")
    assert "Rubric id: domain_dimension_v2" in prompt
    assert "Domain: medical" in prompt
    assert "Estimate the score that a careful human expert would assign" in prompt
    assert "apply one general standard across datasets" in prompt
    assert "Score anchors:" in prompt
    assert "0.25 = mostly poor" in prompt
    assert "1.0 = excellent" in prompt


def test_medical_and_counseling_prompts_share_general_rubric_with_distinct_metadata() -> None:
    medical_prompt, _ = build_judge_prompt(_example(), rubric_id="domain_dimension_v2")
    counseling_prompt, _ = build_judge_prompt(
        _counseling_example(),
        rubric_id="domain_dimension_v2",
    )
    assert medical_prompt != counseling_prompt
    assert "Domain: medical" in medical_prompt
    assert "Domain: counseling" in counseling_prompt
    assert UNIVERSAL_HUMAN_ALIGNED_RUBRIC in medical_prompt
    assert UNIVERSAL_HUMAN_ALIGNED_RUBRIC in counseling_prompt


def test_registered_domain_dimension_rubrics_are_universal_for_v2_sources() -> None:
    assert set(DOMAIN_DIMENSION_RUBRICS.values()) == {UNIVERSAL_HUMAN_ALIGNED_RUBRIC}


def test_prompt_hash_changes_when_rubric_id_changes() -> None:
    _, generic_hash = build_judge_prompt(_example(), rubric_id="generic_v1")
    _, v2_hash = build_judge_prompt(_example(), rubric_id="domain_dimension_v2")
    assert generic_hash != v2_hash


def test_mock_client_output_parses_to_valid_score() -> None:
    prompt, _ = build_judge_prompt(_example())
    raw = MockJudgeClient(seed=1).score(prompt)
    score, payload = parse_score_json(raw)
    assert 0.0 <= score <= 1.0
    assert "rationale" in payload


def test_parse_score_recovers_from_truncated_rationale() -> None:
    score, payload = parse_score_json('{"score": 0.42, "rationale": "truncated')
    assert score == 0.42
    assert payload["parse_recovered"] is True


def test_parse_score_recovers_from_malformed_json_with_score() -> None:
    score, payload = parse_score_json('{"score": 0.33, "rationale": "bad "quote"}')
    assert score == 0.33
    assert payload["parse_recovered"] is True


def test_provider_factory_returns_real_client_types_without_network() -> None:
    assert isinstance(make_judge_client("openai", model="gpt-test"), OpenAIChatJudgeClient)
    assert isinstance(make_judge_client("anthropic", model="claude-test"), AnthropicJudgeClient)
    assert isinstance(make_judge_client("gemini", model="gemini-test"), GeminiJudgeClient)
    assert isinstance(make_judge_client("deepseek", model="deepseek-test"), OpenAIChatJudgeClient)
    assert isinstance(make_judge_client("cohere", model="cohere-test"), OpenAIChatJudgeClient)
    assert isinstance(make_judge_client("mistral", model="mistral-test"), OpenAIChatJudgeClient)
    assert isinstance(make_judge_client("qwen", model="qwen-test"), OpenAIChatJudgeClient)


def test_cohere_provider_uses_expected_openai_compatible_settings() -> None:
    client = make_judge_client("cohere", model="command-a-plus-05-2026")
    assert isinstance(client, OpenAIChatJudgeClient)
    assert client.model == "command-a-plus-05-2026"
    assert client.api_key_env == "COHERE_API_KEY"
    assert client.base_url == "https://api.cohere.ai/compatibility/v1"


def test_v2_required_judge_list_replaces_deepseek_with_cohere_and_uses_flash_gemini() -> None:
    assert PHASE7_V2_DONOR_JUDGE_IDS == [
        "openai_gpt55",
        "anthropic_sonnet46",
        "google_gemini25flash_think1024",
        "cohere_command_a_plus",
        "mistral_large",
    ]
    assert "deepseek_v4flash" not in PHASE7_V2_DONOR_JUDGE_IDS
    assert "google_gemini25pro" not in PHASE7_V2_DONOR_JUDGE_IDS


def test_redact_url_strips_query_strings() -> None:
    assert _redact_url("https://example.test/path?key=secret") == "https://example.test/path"
