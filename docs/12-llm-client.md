# 12 — LLM client

Source of truth: `src/qlab_rl/llm/client.py`.

Generic OpenAI-compatible client, stdlib `urllib` only. One class talks to
local servers and cloud gateways; only `base_url`, auth, and model handling
differ. See [16-providers.md](16-providers.md) for the preset catalog.

- `GET /models` lists models, `POST /chat/completions` chats.
- `from_provider()` builds a client from a `resolve_provider()` dict:
  bearer auth from env key, `extra_headers`, `auto_detect` flag.
- `resolve_model()`: returns the configured model; with `auto_detect` it
  falls back to the loaded chat model (LM Studio, empty `model = ""`);
  cloud presets require an explicit model id.
- Retries: connection errors plus 429/5xx honouring `Retry-After`. 401 =
  bad/missing key (names `$VAR`, never the value), 402 = out of credit,
  404 = wrong model id.
- `reasoning_effort = "none"` stops the model spending the whole token budget
  on reasoning and returning empty content (verified by
  `scripts/probe_latency.py`). `extra_body` passes vendor-specific params.
- `timeout_sec = 60`, `retries = 2` for episode runs.
- `text_of()` extracts message content (string or list of text blocks);
  `reasoning_of()` exposes the discarded reasoning trace for debugging only.
