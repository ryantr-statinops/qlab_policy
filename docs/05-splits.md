# 05 — Splits

Source of truth: `src/qlab_rl/env/splits.py`.

Chronological 60/20/20 split **by bar count**: train first 60%, val next 20%,
test last 20%. Splitting by index (not date) keeps the ratio exact regardless
of holidays; because bars are sorted by date, the split is still strictly
chronological.

`split_indices` validates fractions are in (0, 1) and sum to 1.0, then
rounds and clamps so train/val leave at least room for a test tail.
Default live range: train 2015-01 -> 2022-01, val 2022-01 -> 2024-05,
test 2024-05 -> 2026-10.
