# docs

Single-source notes for `qlab_rl`. README stays the landing page; details live here.

| File | Question it answers | Source of truth |
|---|---|---|
| `00-overview.md` | What is this? | `README.md` |
| `01-setup.md` | LM Studio setup | `scripts/check_lmstudio.py` |
| `02-quickstart.md` | Commands to run | `scripts/*.py` |
| `03-architecture.md` | Modules + loop | `src/qlab_rl/runner.py` |
| `04-data.md` | Bars, cache | `src/qlab_rl/data/provider.py`, `scripts/fetch_data.py` |
| `05-splits.md` | Train/val/test | `src/qlab_rl/env/splits.py` |
| `06-state.md` | 6-number observation | `src/qlab_rl/env/state.py` |
| `07-trading-env.md` | Fee / reward / equity | `src/qlab_rl/env/trading_env.py` |
| `08-metrics.md` | Sharpe / maxDD / ... | `src/qlab_rl/env/trading_env.py:metrics()` |
| `09-actions.md` | Exposure sets 5/3/4 | `src/qlab_rl/policies/actions.py` |
| `10-rule-policies.md` | Baselines | `src/qlab_rl/policies/rule.py` |
| `11-llm-policy.md` | Prompt / parse / fallback | `src/qlab_rl/policies/llm.py` |
| `12-llm-client.md` | LM Studio HTTP | `src/qlab_rl/llm/client.py` |
| `13-config.md` | TOML fields | `configs/*.toml` |
| `14-outputs-eval.md` | CSV schema, reading results | `src/qlab_rl/runner.py`, `scripts/eval.py` |
| `15-limits.md` | Limits + next | `README.md`, code comments |
| `16-providers.md` | Provider presets, keys, cost | `configs/providers.toml`, `src/qlab_rl/llm/client.py` |
