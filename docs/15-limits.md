# 15 — Limits & next

Known limits:

- Single asset, daily bars, no corporate actions, no borrow fee on shorts.
- One LLM call per bar, strictly sequential — roughly 500–1,400 calls per
  episode. At ~1 s/call that is 10–25 minutes per episode.
- No weight training. 4 GB VRAM is not enough for GRPO/PPO on a 4B model.

Next: give the LLM context a numeric policy cannot consume — news text,
research notes, chart descriptions — not weight training. The `memories`
hook in `build_user_message` is a placeholder for that direction.
