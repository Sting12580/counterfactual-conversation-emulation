from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cce_data.build import write_jsonl


DEFAULT_AGENT_CONFIG = Path("configs/agent_policy.yaml")


def load_agent_config(path: Path = DEFAULT_AGENT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_dynamic_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def generate_with_openai(record: dict[str, Any], model: str, agent_config: dict[str, Any]) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install judge dependencies with: pip install -e '.[judge]'") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for provider=openai")

    client = OpenAI()
    user_prompt = agent_config["user_prompt_template"].format(
        x_patient_context=record.get("x_patient_context", "")
    )
    system_prompt = agent_config["system_prompt"]

    kwargs: dict[str, Any] = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if "max_output_tokens" in agent_config:
        kwargs["max_output_tokens"] = int(agent_config["max_output_tokens"])
    # Some Responses API models accept temperature; if a model rejects it, rerun
    # with a config that omits temperature.
    if agent_config.get("temperature") is not None:
        kwargs["temperature"] = float(agent_config["temperature"])

    response = client.responses.create(**kwargs)
    text = getattr(response, "output_text", "").strip()
    if not text:
        raise RuntimeError(f"Empty agent output for {record.get('example_id')}")
    return text


def generate_agent_actions(
    input_path: Path,
    output_path: Path,
    provider: str = "openai",
    model: str | None = None,
    agent_config_path: Path = DEFAULT_AGENT_CONFIG,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")
    agent_config = load_agent_config(agent_config_path)
    model = model or agent_config.get("recommended_default_model", "gpt-4.1")

    records: list[dict[str, Any]] = []
    generated = 0
    now = datetime.now(timezone.utc).isoformat()

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            should_generate = (
                record.get("inclusion_status") == "included"
                and (overwrite or not record.get("a_agent"))
                and (limit is None or generated < limit)
            )
            if should_generate:
                if provider == "openai":
                    action = generate_with_openai(record, model=model, agent_config=agent_config)
                else:
                    raise ValueError(f"Unsupported generation provider: {provider}")
                record["a_agent"] = action
                record["agent_source"] = f"{provider}:{model}"
                record["agent_model"] = model
                record["agent_policy_version"] = agent_config.get("policy_version")
                record["agent_generated_at"] = now
                generated += 1
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
        "agent_config": str(agent_config_path),
        "agent_policy_version": agent_config.get("policy_version"),
        "generated": generated,
    }
