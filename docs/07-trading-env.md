# 07 — Trading env

Source of truth: `src/qlab_rl/env/trading_env.py`.

At bar `i` you observe state from `bars[0..i]` and choose exposure `e` in
`[-1, 1]`. Shorting is symmetric: a falling market with a short earns money.

Per-bar accounting:

```
cost    = fee_rate * |e - e_prev|              # fee_rate = fee_bps / 10000
gross   = e * (close[i+1]/close[i] - 1)
net     = gross - cost
reward  = net - lambda * vol_20(i) * |e|
equity *= (1 + net)
```

Defaults: `fee_bps = 10.0`, `lambda_risk = 0.5`, `warmup = 21`
(must exceed max lookback), `trading_days_per_year = 252`.

End of segment: `liquidate_at_end = true` forces flat, charging one final
`fee_rate * |exposure|` on equity. `reset()` restarts at `warmup` with
`equity = 1.0`, `exposure = 0.0`.
