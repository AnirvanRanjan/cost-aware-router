"""Grouped train / cal / test splitting (DESIGN.md §2.5).

Split by group, never by row: every near-duplicate cluster shares one group_id,
and a group lands wholly in one split. Lane A extends this module with
leave-one-subject-out and nested grouped CV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_FRACTIONS = {"train": 0.60, "cal": 0.20, "test": 0.20}


def grouped_split(group_ids, fractions: dict[str, float] | None = None, seed: int = 0) -> np.ndarray:
    """Assign each row a split name so that no group spans two splits.

    Groups are visited in a seeded random order and each goes to whichever split is
    furthest below its target row count, so row fractions land close to target even
    when group sizes vary.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    names = list(fractions)
    probs = np.array([fractions[n] for n in names], dtype=float)
    if (probs < 0).any() or not np.isclose(probs.sum(), 1.0):
        raise ValueError(f"split fractions must be non-negative and sum to 1, got {fractions}")

    groups = pd.Series(np.asarray(group_ids), dtype=object)
    sizes = groups.value_counts().sort_index()          # deterministic base order
    order = sizes.index.to_numpy()[np.random.default_rng(seed).permutation(len(sizes))]

    targets = probs * len(groups)
    filled = np.zeros(len(names))
    assignment: dict = {}
    for g in order:
        k = int(np.argmax(targets - filled))
        assignment[g] = names[k]
        filled[k] += sizes[g]
    return groups.map(assignment).to_numpy()


def assert_no_leakage(df: pd.DataFrame, group_col: str = "group_id", split_col: str = "split") -> None:
    """Raise if any group appears in more than one split. Call this after every split."""
    spans = df.groupby(group_col)[split_col].nunique()
    leaked = spans[spans > 1]
    if len(leaked):
        raise AssertionError(
            f"{len(leaked)} groups span more than one split, e.g. {list(leaked.index[:5])}"
        )
