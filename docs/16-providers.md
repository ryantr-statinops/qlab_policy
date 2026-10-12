# 16 — Providers

Any OpenAI-compatible endpoint works: `POST {base_url}/chat/completions`
plus `GET {base_url}/models`. No SDK, stdlib `urllib` only.

## Local (no auth)

| Preset | `base_url` | `model` |
|---|---|---|
| `lmstudio` (default) | `http://localhost:1234/v1` | empty = auto-detect loaded chat model |
| `ollama` | `http://localhost:11434/v1` | required, e.g. `qwen3:4b` |
| `vllm` | `http://localhost:8000/v1` | required (`--model`) |
| `llamacpp` | `http://localhost:8080/v1` | required (`--model`) |

## Cloud (API key via env var only)

| Preset | `api_key_env` | `model` | Notes |
|---|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | required, `vendor/slug` | sends `HTTP-Referer` + `X-Title`; 458+ models |
| `openai` | `OPENAI_API_KEY` | required | reference shape |
| `custom` | your var | required | Together / Fireworks / Groq: fill base_url + key |

Keys never go into toml, CSV, or logs. Error messages name the variable
(`check $OPENROUTER_API_KEY`), never the value. `check_provider.py` shows
`(set, hidden)`.

## Behaviour differences

- `auto_detect = false` on cloud: a missing model is an error, not a scan
  (a gateway lists thousands of models).
- Retries: connection errors (old path) plus 429/5xx honouring `Retry-After`.
  401 = bad/missing key, 402 = out of credit, 404 = wrong model id.
- `run_llm.py` on a cloud provider without `--max-steps` stops with a cost
  estimate unless `--yes` is passed. One call per bar x 500-1400 bars adds
  up: trial with `--max-steps 40` first.

## Usage

```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
python scripts\check_provider.py --provider openrouter
python scripts\ask_model.py --provider openrouter --repeat 3
python scripts\run_llm.py --provider openrouter --max-steps 40
```
