"""Minimal OpenAI-compatible client for a local LM Studio server.

stdlib only (urllib).
"""

from __future__ import annotations

import json
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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

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