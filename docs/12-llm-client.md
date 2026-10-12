# 12 — LLM client

Source of truth: `src/qlab_rl/llm/client.py`.

Minimal OpenAI-compatible client for a local LM Studio server, stdlib
`urllib` only.

- `base_url` default `http://localhost:1234/v1`; `GET /models` lists models,
  `POST /chat/completions` chats.
- `resolve_model()`: returns the configured model, else auto-detects the
  single loaded chat model (empty `model = ""` in config).
- `reasoning_effort = "none"` stops the model spending the whole token budget
  on reasoning and returning empty content (verified by
  `scripts/probe_latency.py`). `"low"`/`"medium"` lets it reason, costing
  latency and tokens.
- `timeout_sec = 60`, `retries = 2` for episode runs.
- `text_of()` extracts message content; `reasoning_of()` exposes the
  discarded reasoning trace for debugging only.
