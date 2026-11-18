"""Episode action-source mixture.

The mix mirrors H-JEPA's explore-and-stitch data: mostly expert motion-planned
rollouts, a slice of epsilon-noisy copies of those experts, and a slice of
random workspace exploration. Failed and colliding episodes are kept.
Counts use the largest-remainder method so a short run (for example 20
episodes at 0.5/0.3/0.2) is exactly 10/6/4 rather than a noisy sample.
"""

from __future__ import annotations

import numpy as np

SOURCE_NAMES = ("expert", "noisy", "explore")
SOURCE_CODE = {name: index for index, name in enumerate(SOURCE_NAMES)}


def assign_sources(num_episodes: int, weights: dict, seed: int) -> list[str]:
    keys = [name for name in SOURCE_NAMES if name in weights]
    if not keys:
        raise ValueError("source_weights is empty")
    raw = np.array([float(weights[name]) for name in keys], dtype=np.float64)
    if np.any(raw < 0) or float(raw.sum()) <= 0:
        raise ValueError(f"source_weights must be non-negative and sum to > 0, got {weights}")
    raw = raw / raw.sum() * num_episodes
    counts = np.floor(raw).astype(int)
    leftover = int(num_episodes - counts.sum())
    order = np.argsort(-(raw - counts))
    for index in order[:leftover]:
        counts[index] += 1
    sequence: list[str] = []
    for name, count in zip(keys, counts):
        sequence.extend([name] * int(count))
    rng = np.random.default_rng(seed)
    rng.shuffle(sequence)
    return sequence


def source_counts(sources: list[str]) -> dict[str, int]:
    return {name: sum(item == name for item in sources) for name in SOURCE_NAMES}
