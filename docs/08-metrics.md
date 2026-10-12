# 08 — Metrics

Source of truth: `TradingEnv.metrics()` in `src/qlab_rl/env/trading_env.py`.

| Metric | Definition |
|---|---|
| `total_return` | `equity - 1.0` |
| `sharpe` | mean(net) / sample-stdev(net) * sqrt(252); `nan` if < 2 bars or sd = 0 |
| `cagr` | `equity ** (252/n) - 1` |
| `max_drawdown` | worst `equity/peak - 1` over the equity curve (<= 0) |
| `profit_factor` | sum(gains) / sum(losses); `inf` if no losses, `nan` if flat |
| `calmar` | `cagr / abs(max_drawdown)`; `inf` if no drawdown |
| `mean_reward` | mean of risk-adjusted rewards |
| `flat_ratio` | fraction of actions == 0.0 |
| `mean_exposure` | mean of chosen exposures |
| `turnover` | sum of `|a - b|` over consecutive actions from 0.0 |
| `final_equity` | equity after liquidation |

`run_policy` adds `parse_rate`, `steps`, `action_hist`, `truncated`.
