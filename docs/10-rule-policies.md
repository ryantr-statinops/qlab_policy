# 10 — Rule policies (baselines)

Source of truth: `src/qlab_rl/policies/rule.py`, `scripts/run_baseline.py`.

Non-LLM reference policies. These define the bar the LLM has to clear:

| Name | Behaviour |
|---|---|
| `buy_hold` | always 1.0 — the one that matters |
| `fixed_half` | always 0.5 |
| `always_flat` | always 0.0 |
| `random` | uniform over the 5 exposures, seeded RNG(42) |
| `sma20_long_flat` | long when price > 20-bar SMA, else flat |
| `sma20_cross` | long above SMA20, short below (always-in trend follower) |

Run BEFORE trusting any LLM result:

```powershell
python scripts\run_baseline.py --split test
```

Reference numbers, SPY test split:

| policy | sharpe | ret | maxDD |
|---|---|---|---|
| buy_hold | **1.02** | +41.5% | -19.0% |
| fixed_half | **1.02** | +19.9% | -9.9% |
| sma20_long_flat | 0.20 | +3.2% | -11.8% |
| random | -1.71 | -36.2% | -41.2% |
