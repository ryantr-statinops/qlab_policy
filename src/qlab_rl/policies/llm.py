"""LLM policy: the language model is the decision maker.

One chat completion per bar. The model reads a compact state table and answers
with a single exposure choice, e.g. `ACTION: LONG_HALF`.

Design notes
------------
* temperature 0 by default, because a policy that flips its mind between runs
  makes every downstream metric meaningless.
* The prompt is deliberately short. On a 4 GB card the model is partly on CPU;
  every extra token costs wall-clock on every single bar.
* Raw values are shown as percentages rather than z-scores. A 4B model reads
  "+1.20%" more reliably than it reads "0.83 sigma".
"""

from __future__ import annotations

from typing import Sequence

from .actions import action_set

EXPOSURE_FEATURE = "exposure"


def feature_names(
    ret_windows: Sequence[int] = (1, 5, 20),
    vol_window: int = 20,
    drawdown_window: int = 20,
) -> list[str]:
    """Mirror of StateBuilder.feature_names, for prompt building."""
    return (
        [f"return_{w}d" for w in ret_windows]
        + [f"vol_{vol_window}d", f"drawdown_{drawdown_window}d", EXPOSURE_FEATURE]
    )


def format_pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def build_user_message(
    obs: Sequence[float],
    date: str,
    price: float,
    n_actions: int,
    memories: Sequence[dict] | None = None,
    names: Sequence[str] | None = None,
) -> str:
    names = list(names or feature_names())
    if len(names) != len(obs):
        raise ValueError(f"obs has {len(obs)} values but {len(names)} feature names")

    rows = []
    for label, value in zip(names, obs):
        if label == EXPOSURE_FEATURE:
            rows.append(f"  {label:<20} {value:+.2f}")
        else:
            rows.append(f"  {label:<20} {format_pct(value)}")

    options = "\n".join(
        f"  {name:<12} = {value:+.2f} "
        f"({abs(value) * 100:.0f}% "
        f"{'short' if value < 0 else 'long' if value > 0 else 'flat'})"
        for name, value in action_set(n_actions)
    )

    parts: list[str] = []
    if memories:
        parts.append("Decisions that worked well in earlier sessions:")
        for m in memories:
            parts.append(f"  {m['state']}  ->  {m['action']}  ({m['outcome']})")
        parts.append("")

    parts.append(f"Date: {date}   Close: {price:.2f}")
    parts.append("")
    parts.append("Market state:")
    parts.extend(rows)
    parts.append("")
    parts.append("Choose exactly one position exposure:")
    parts.append(options)
    parts.append("")
    parts.append("Reply in exactly this format and nothing else:")
    parts.append("ACTION: <one of the option names>")
    parts.append("REASON: <one short sentence>")
    return "\n".join(parts)