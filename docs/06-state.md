# 06 — State

Source of truth: `src/qlab_rl/env/state.py`.

At bar `i` the observation uses `bars[0..i]` only; the chosen exposure earns
the return from bar `i` to `i+1`. No lookahead.

Layout with default lookbacks (`ret_windows=[1,5,20]`, `vol_window=20`,
`drawdown_window=20`):

```
0 return_1d     close[i]/close[i-1] - 1
1 return_5d     close[i]/close[i-5] - 1
2 return_20d    close[i]/close[i-20] - 1
3 vol_20d       annualised stdev of daily returns over vol_window bars
4 drawdown_20d  close / rolling_max - 1  (<= 0)
5 exposure      exposure currently held, in [-1, 1]
```

Vector length is `len(ret_windows) + 3`; `feature_names` mirrors the layout
exactly. `observe()` returns `None` while still warming up; `warmup =
max(ret_windows + [vol_window, drawdown_window])`.
