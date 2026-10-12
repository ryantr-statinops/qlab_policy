# 13 — Config

Loader: `src/qlab_rl/config.py` (stdlib `tomllib`); `resolve()` resolves
relative paths against the project root. `load_providers()` reads
`configs/providers.toml`; `resolve_provider()` merges one preset with
precedence **CLI flag > preset > `[llm]` default**. Cloud keys come from
`os.environ[api_key_env]` only — never from a file.

`configs/env.toml`:

| Field | Meaning |
|---|---|
| `env.fee_bps` | commission + slippage in bps on traded notional |
| `env.lambda_risk` | risk aversion in `reward = net - lambda * vol20 * \|exposure\|` |
| `env.warmup` | bars before the first decision (must exceed max lookback) |
| `env.trading_days_per_year` | annualisation factor (252) |
| `env.liquidate_at_end` | forced flat at segment end |
| `split.train/val/test` | 0.60 / 0.20 / 0.20 chronological fractions |
| `state.ret_windows/vol_window/drawdown_window` | lookbacks in bars |

`configs/policy.toml`: `[llm]` provider, endpoint, model, temperature,
max_tokens, timeout, retries, reasoning_effort; `[policy]` n_actions,
system_prompt; `[prompt]` include_history, history_window.

`configs/providers.toml`: one table per preset (`lmstudio`, `ollama`,
`vllm`, `llamacpp`, `openrouter`, `openai`, `custom`) sharing the schema
`base_url / model / api_key_env / auto_detect / timeout_sec /
extra_headers`. See [16-providers.md](16-providers.md).

`configs/data.toml`: ticker, start, source, cache_dir. Pluggable for vnstock
later.
