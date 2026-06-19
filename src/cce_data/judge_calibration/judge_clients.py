from __future__ import annotations

import hashlib
import json
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


PHASE7_V2_DONOR_JUDGE_IDS = [
    "openai_gpt55",
    "anthropic_sonnet46",
    "google_gemini25flash_think1024",
    "cohere_command_a_plus",
    "mistral_large",
]
PHASE7_V2_REQUIRED_JUDGES = ",".join(PHASE7_V2_DONOR_JUDGE_IDS)


class JudgeClient(ABC):
    @abstractmethod
    def score(self, prompt: str) -> str:
        """Return a raw JSON string with a scalar score."""


class MockJudgeClient(JudgeClient):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def score(self, prompt: str) -> str:
        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        payload = {"score": rng.random(), "rationale": "mock deterministic score"}
        return json.dumps(payload, sort_keys=True)


class HttpJudgeClient(JudgeClient):
    def __init__(self, timeout: float = 120.0, max_retries: int = 2) -> None:
        self.timeout = timeout
        self.max_retries = max_retries

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        safe_url = _redact_url(url)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            retry_delay_seconds = float(2**attempt)
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400:
                    body = response.text[:1000]
                    retry_delay_seconds = _retry_delay_seconds(
                        attempt=attempt,
                        status_code=response.status_code,
                        retry_after=response.headers.get("Retry-After"),
                    )
                    raise RuntimeError(f"HTTP {response.status_code} from {safe_url}: {body}")
                return response.json()
            except Exception as exc:  # pragma: no cover - exercised in integration runs
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(retry_delay_seconds)
        raise RuntimeError(f"Request failed after retries: {last_error}") from last_error


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _retry_delay_seconds(
    *,
    attempt: int,
    status_code: int,
    retry_after: str | None,
) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 300.0)
        except ValueError:
            pass
    if status_code == 429:
        return min(120.0, 15.0 * (attempt + 1))
    return float(2**attempt)


class OpenAIChatJudgeClient(HttpJudgeClient):
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 512,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}

    def score(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required environment variable {self.api_key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge. Return only valid JSON with "
                        'a numeric "score" in [0, 1] and a short "rationale".'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        if self.api_key_env == "OPENAI_API_KEY":
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_body)
        response = self._post_json(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload=payload,
        )
        return response["choices"][0]["message"]["content"]


class AnthropicJudgeClient(HttpJudgeClient):
    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens

    def score(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required environment variable {self.api_key_env}")
        response = self._post_json(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            payload={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": (
                    "You are a strict evaluation judge. Return only valid JSON with "
                    'a numeric "score" in [0, 1] and a short "rationale".'
                ),
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        parts = response.get("content", [])
        text_parts = [part.get("text", "") for part in parts if part.get("type") == "text"]
        return "\n".join(text_parts).strip()


class GeminiJudgeClient(HttpJudgeClient):
    def __init__(
        self,
        model: str,
        api_key_env: str = "GEMINI_API_KEY",
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.model = model
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens

    def score(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing required environment variable {self.api_key_env}")
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
            "maxOutputTokens": self.max_tokens,
        }
        model_name = self.model.lower()
        if model_name == "gemini-2.5-flash":
            generation_config["thinkingConfig"] = {"thinkingBudget": 1024}
        elif model_name == "gemini-2.5-flash-lite":
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        elif model_name == "gemini-3.5-flash":
            generation_config["thinkingConfig"] = {"thinkingLevel": "low"}
        response = self._post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            payload={
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "Return only valid JSON with a numeric score in [0, 1] "
                                    "and a short rationale.\n\n"
                                    + prompt
                                )
                            }
                        ],
                    }
                ],
                "generationConfig": generation_config,
            },
        )
        candidates = response.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini response did not include candidates: {response}")
        parts = candidates[0].get("content", {}).get("parts", [])
        return "\n".join(part.get("text", "") for part in parts).strip()


class UnsupportedRealJudgeClient(JudgeClient):
    def __init__(self, provider: str, model: str | None = None) -> None:
        self.provider = provider
        self.model = model

    def score(self, prompt: str) -> str:
        del prompt
        raise RuntimeError(
            f"Real judge provider {self.provider!r} is not wired in this offline-safe adapter. "
            "Use --provider mock or add a provider client with explicit credentials."
        )


def make_judge_client(
    provider: str,
    model: str | None = None,
    seed: int = 0,
    base_url: str | None = None,
    timeout: float = 120.0,
    max_retries: int = 2,
    max_tokens: int = 512,
) -> JudgeClient:
    if provider == "mock":
        return MockJudgeClient(seed=seed)
    if provider == "openai":
        return OpenAIChatJudgeClient(
            model=model or "gpt-5.5",
            api_key_env="OPENAI_API_KEY",
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        return AnthropicJudgeClient(
            model=model or "claude-sonnet-4-6",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
    if provider == "gemini":
        return GeminiJudgeClient(
            model=model or "gemini-2.5-pro",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
    if provider == "deepseek":
        return OpenAIChatJudgeClient(
            model=model or "deepseek-v4-pro",
            api_key_env="DEEPSEEK_API_KEY",
            base_url=base_url or "https://api.deepseek.com",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
            extra_body={"thinking": {"type": "disabled"}},
        )
    if provider == "cohere":
        return OpenAIChatJudgeClient(
            model=model or "command-a-plus-05-2026",
            api_key_env="COHERE_API_KEY",
            base_url=base_url or "https://api.cohere.ai/compatibility/v1",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
    if provider == "mistral":
        return OpenAIChatJudgeClient(
            model=model or "mistral-large-latest",
            api_key_env="MISTRAL_API_KEY",
            base_url=base_url or "https://api.mistral.ai/v1",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
    if provider in {"qwen", "dashscope"}:
        return OpenAIChatJudgeClient(
            model=model or "qwen3-max",
            api_key_env="DASHSCOPE_API_KEY",
            base_url=base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            timeout=timeout,
            max_retries=max_retries,
            max_tokens=max_tokens,
        )
    return UnsupportedRealJudgeClient(provider=provider, model=model)
