"""Discrete exposure action sets."""

from __future__ import annotations

ACTION_SETS: dict[int, list[tuple[str, float]]] = {
    5: [
        ("SHORT_ALL", -1.0),
        ("SHORT_HALF", -0.5),
        ("FLAT", 0.0),
        ("LONG_HALF", 0.5),
        ("LONG_ALL", 1.0),
    ],
    3: [
        ("FLAT", 0.0),
        ("LONG_HALF", 0.5),
        ("LONG_ALL", 1.0),
    ],
    4: [
        ("SHORT_HALF", -0.5),
        ("FLAT", 0.0),
        ("LONG_HALF", 0.5),
        ("LONG_ALL", 1.0),
    ],
}


def action_set(n_actions: int) -> list[tuple[str, float]]:
    if n_actions not in ACTION_SETS:
        raise ValueError(f"n_actions must be one of {sorted(ACTION_SETS)}, got {n_actions}")
    return list(ACTION_SETS[n_actions])


def labels(n_actions: int) -> list[str]:
    return [name for name, _ in action_set(n_actions)]


def value_of(n_actions: int, label: str) -> float | None:
    for name, value in action_set(n_actions):
        if name == label:
            return value
    return None


def nearest(n_actions: int, target: float) -> float:
    """Closest legal exposure to `target`, used when the model says something invalid."""
    return min((v for _, v in action_set(n_actions)), key=lambda v: abs(v - target))


def label_for(n_actions: int, value: float) -> str:
    for name, v in action_set(n_actions):
        if abs(v - value) < 1e-9:
            return name
    return "UNKNOWN"