"""Minimal OpenAI-compatible client for a local LM Studio server.

stdlib only (urllib).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 160,
        timeout: int = 60,
        retries: int = 2,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.reasoning_effort = reasoning_effort

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "LLMClient":
        """Build a client from the [llm] block of configs/policy.toml."""
        return cls(
            base_url=overrides.pop("base_url", cfg["base_url"]),
            model=overrides.pop("model", cfg["model"]),
            temperature=overrides.pop("temperature", cfg["temperature"]),
            max_tokens=overrides.pop("max_tokens", cfg["max_tokens"]),
            timeout=overrides.pop("timeout", cfg["timeout_sec"]),
            retries=overrides.pop("retries", cfg["retries"]),
            reasoning_effort=overrides.pop("reasoning_effort", cfg.get("reasoning_effort")),
            **overrides,
        )

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise LLMError(f"HTTP {exc.code} from {url}: {detail[:400]}") from exc
            except Exception as exc:  # connection refused, timeout, ...
                last = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (attempt + 1))
        raise LLMError(
            f"cannot reach {url} after {self.retries + 1} tries: {last}. "
            "Is LM Studio running with Start Server enabled on the configured port?"
        )

    def _get(self, endpoint: str) -> Any:
        url = f"{self.base_url}/{endpoint}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=min(self.timeout, 10)) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise LLMError(
                f"cannot reach {url}: {exc}. "
                "Is LM Studio running with Start Server enabled on the configured port?"
            ) from exc

    def health(self) -> tuple[bool, str]:
        try:
            self._get("models")
            return True, "ok"
        except LLMError as exc:
            return False, str(exc)

    def list_models(self) -> list[dict[str, Any]]:
        return self._get("models").get("data", [])

    def resolve_model(self) -> str:
        """Return the configured model, else auto-detect the loaded chat model."""
        if self.model:
            return self.model
        models = self.list_models()
        if not models:
            raise LLMError(
                "server is up but reports zero models. Load a chat model in "
                "LM Studio first (an embedding model cannot generate text)."
            )
        chat = [m for m in models if "embed" not in m.get("id", "").lower()]
        if not chat:
            raise LLMError(
                f"only embedding models loaded: {[m.get('id') for m in models]}. "
                "An embedding model cannot answer chat prompts."
            )
        return chat[0]["id"]

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": model or self.resolve_model(),
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return self._post("chat/completions", payload)

    @staticmethod
    def text_of(response: dict[str, Any]) -> str:
        choices = response.get("choices") or []
        if not choices:
            raise LLMError(f"response has no choices: {str(response)[:300]}")
        return (choices[0].get("message") or {}).get("content", "") or ""