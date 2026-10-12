# 03 — Architecture

Source of truth: `src/qlab_rl/runner.py`, `src/qlab_rl/policies/base.py`.

```
configs/          data.toml, env.toml, policy.toml
scripts/          check_lmstudio, ask_model, fetch_data,
                  run_baseline, run_llm, eval (+ probe_latency)
src/qlab_rl/
  config.py       tomllib loader
  runner.py       drives a policy through an env, writes records
  llm/client.py   LM Studio OpenAI-compatible client
  data/provider.py Yahoo Finance OHLCV + CSV cache
  env/state.py    observation builder (no lookahead)
  env/trading_env.py  step / equity / reward / metrics
  env/splits.py   60/20/20 chronological
  policies/       base, actions, rule, llm
tests/            57 tests, no LM Studio required
data/raw/         cached CSV (gitignored)
outputs/          per-episode records (gitignored)
```

`run_policy(env, policy, on_step, max_steps)` drives the loop: `env.reset()`,
then `policy.act(obs, ctx)` -> `env.step(action)` until done or `max_steps`.
Returns env metrics plus `parse_rate`, `steps`, `action_hist`, `truncated`.

`PolicyContext` is a read-only view handed to policies (builder, index,
episode counters, `date`/`price` properties). It keeps prompt building out
of the env.
