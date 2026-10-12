# 09 — Actions

Source of truth: `src/qlab_rl/policies/actions.py`.

| `n_actions` | Set |
|---|---|
| 5 (default) | `SHORT_ALL=-1.0, SHORT_HALF=-0.5, FLAT=0.0, LONG_HALF=0.5, LONG_ALL=1.0` |
| 3 (long-only) | `FLAT=0.0, LONG_HALF=0.5, LONG_ALL=1.0` |
| 4 | `SHORT_HALF=-0.5, FLAT=0.0, LONG_HALF=0.5, LONG_ALL=1.0` |

Set via `n_actions` in `configs/policy.toml` or `--actions` on `run_llm.py` /
`eval.py`. Swap 5 -> 3 to test whether shorting is what is hurting.

Helpers: `labels()` lists names, `value_of()` maps a label to exposure,
`label_for()` maps a value back, `nearest()` snaps an invalid target to the
closest legal exposure (used for the LLM fallback).
