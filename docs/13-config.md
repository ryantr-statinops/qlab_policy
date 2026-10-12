# 13 — Config

Loader: `src/qlab_rl/config.py` (stdlib `tomllib`); `resolve()` resolves
relative paths against the project root.

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

`configs/policy.toml`: `[llm]` endpoint, model, temperature, max_tokens,
timeout, retries, reasoning_effort; `[policy]` n_actions, system_prompt;
`[prompt]` include_history, history_window.

`configs/data.toml`: ticker, start, source, cache_dir. Pluggable for vnstock
later.
