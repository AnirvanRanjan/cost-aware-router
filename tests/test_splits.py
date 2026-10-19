import numpy as np
import pandas as pd
import pytest

from router.splits import assert_no_leakage, grouped_split


def _groups(n_groups=2000, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 4, size=n_groups)              # groups of 1-3 rows
    return np.repeat([f"g{i}" for i in range(n_groups)], sizes)


def test_no_group_spans_two_splits():
    g = _groups()
    df = pd.DataFrame({"group_id": g, "split": grouped_split(g, seed=1)})
    assert_no_leakage(df)


def test_row_fractions_land_near_target():
    g = _groups()
    share = pd.Series(grouped_split(g, seed=1)).value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.60, abs=0.02)
    assert share["cal"] == pytest.approx(0.20, abs=0.02)
    assert share["test"] == pytest.approx(0.20, abs=0.02)


def test_same_seed_same_split_different_seed_different_split():
    g = _groups()
    assert (grouped_split(g, seed=3) == grouped_split(g, seed=3)).all()
    assert (grouped_split(g, seed=3) != grouped_split(g, seed=4)).any()


def test_bad_fractions_are_rejected():
    with pytest.raises(ValueError):
        grouped_split(["a", "b"], {"train": 0.7, "test": 0.7})


def test_leakage_check_catches_a_straddling_group():
    df = pd.DataFrame({"group_id": ["g1", "g1", "g2"], "split": ["train", "test", "cal"]})
    with pytest.raises(AssertionError, match="1 groups"):
        assert_no_leakage(df)
