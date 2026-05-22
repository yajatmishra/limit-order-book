"""
Tests for validation/walk_forward.py  — WalkForwardCV splitter.

Coverage:
  - Rolling mode: fixed train window size, correct step
  - Anchored (expanding) mode: train always starts at 0 and grows
  - No look-ahead: test_start > train_end for all folds
  - Gap: test_start = train_end + 1 + gap
  - n_splits() matches actual number of yielded splits
  - get_splits() returns list equivalent to split() iterator
  - Edge cases: step != test_size, min_train_size, short series
"""

import numpy as np
import pytest
from validation.walk_forward import WalkForwardCV, WalkForwardSplit


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_lookahead(fold: WalkForwardSplit) -> bool:
    """Test strictly follows training data."""
    return fold.test_start > fold.train_end


def _disjoint(fold: WalkForwardSplit) -> bool:
    """Train and test index sets do not overlap."""
    return len(set(fold.train_idx) & set(fold.test_idx)) == 0


# ── Rolling (non-expanding) mode ─────────────────────────────────────────────

class TestRollingMode:

    def test_basic_split_count(self):
        cv     = WalkForwardCV(train_size=100, test_size=25)
        n      = cv.n_splits(500)
        folds  = cv.get_splits(500)
        assert len(folds) == n
        assert n > 0

    def test_train_size_constant(self):
        """All training windows should be exactly train_size."""
        cv = WalkForwardCV(train_size=100, test_size=20, expanding=False)
        for fold in cv.split(300):
            assert len(fold.train_idx) == 100, (
                f"Fold {fold.fold}: expected train_size=100, got {len(fold.train_idx)}"
            )

    def test_test_size_constant(self):
        cv = WalkForwardCV(train_size=50, test_size=10)
        for fold in cv.split(200):
            assert len(fold.test_idx) == 10

    def test_no_lookahead(self):
        cv = WalkForwardCV(train_size=50, test_size=10)
        for fold in cv.split(200):
            assert _no_lookahead(fold), (
                f"Fold {fold.fold}: look-ahead violation "
                f"test_start={fold.test_start} train_end={fold.train_end}"
            )

    def test_train_test_disjoint(self):
        cv = WalkForwardCV(train_size=50, test_size=10)
        for fold in cv.split(200):
            assert _disjoint(fold), f"Fold {fold.fold}: train/test overlap"

    def test_step_advances_correctly(self):
        """With step=5, consecutive test windows advance by 5."""
        cv    = WalkForwardCV(train_size=50, test_size=10, step=5)
        folds = cv.get_splits(200)
        for i in range(1, len(folds)):
            delta = folds[i].test_start - folds[i - 1].test_start
            assert delta == 5, f"Expected step=5, got {delta} between folds {i-1},{i}"

    def test_default_step_is_test_size(self):
        """Default step should equal test_size (non-overlapping tests)."""
        cv    = WalkForwardCV(train_size=50, test_size=20)
        folds = cv.get_splits(300)
        for i in range(1, len(folds)):
            delta = folds[i].test_start - folds[i - 1].test_start
            assert delta == 20

    def test_fold_numbering_sequential(self):
        cv    = WalkForwardCV(train_size=50, test_size=10)
        folds = cv.get_splits(200)
        for i, f in enumerate(folds):
            assert f.fold == i

    def test_test_periods_do_not_overlap(self):
        """Non-overlapping test periods: each test window starts after previous ends."""
        cv    = WalkForwardCV(train_size=50, test_size=10)
        folds = cv.get_splits(200)
        for i in range(1, len(folds)):
            assert folds[i].test_start > folds[i - 1].test_end


# ── Anchored (expanding) mode ─────────────────────────────────────────────────

class TestExpandingMode:

    def test_train_starts_at_zero(self):
        cv = WalkForwardCV(train_size=50, test_size=10, expanding=True)
        for fold in cv.split(200):
            assert fold.train_start == 0, (
                f"Fold {fold.fold}: train should start at 0, got {fold.train_start}"
            )

    def test_train_grows_each_step(self):
        cv    = WalkForwardCV(train_size=50, test_size=10, expanding=True)
        folds = cv.get_splits(200)
        sizes = [len(f.train_idx) for f in folds]
        assert sizes == sorted(sizes), "Training window should monotonically grow"

    def test_no_lookahead_expanding(self):
        cv = WalkForwardCV(train_size=50, test_size=10, expanding=True)
        for fold in cv.split(200):
            assert _no_lookahead(fold)

    def test_disjoint_expanding(self):
        cv = WalkForwardCV(train_size=50, test_size=10, expanding=True)
        for fold in cv.split(200):
            assert _disjoint(fold)

    def test_more_folds_than_rolling_for_same_params(self):
        """Expanding typically produces same or more folds as rolling."""
        cv_roll = WalkForwardCV(train_size=50, test_size=20, expanding=False)
        cv_exp  = WalkForwardCV(train_size=50, test_size=20, expanding=True)
        n_roll  = cv_roll.n_splits(400)
        n_exp   = cv_exp.n_splits(400)
        assert n_exp >= n_roll


# ── Gap parameter ─────────────────────────────────────────────────────────────

class TestGap:

    def test_gap_zero_is_immediate(self):
        """With gap=0, test starts immediately after train."""
        cv = WalkForwardCV(train_size=50, test_size=10, gap=0)
        for fold in cv.split(200):
            assert fold.test_start == fold.train_end + 1

    def test_gap_nonzero_creates_buffer(self):
        """With gap=5, test_start = train_end + 5 + 1 (gap observations skipped)."""
        gap = 5
        cv  = WalkForwardCV(train_size=50, test_size=10, gap=gap)
        for fold in cv.split(300):
            expected_test_start = fold.train_end + gap + 1
            assert fold.test_start == expected_test_start, (
                f"Fold {fold.fold}: expected test_start={expected_test_start}, "
                f"got {fold.test_start}"
            )

    def test_gap_indices_not_in_train_or_test(self):
        """The gap observations should not appear in either split."""
        gap = 10
        cv  = WalkForwardCV(train_size=50, test_size=10, gap=gap)
        for fold in cv.split(300):
            gap_indices = set(range(fold.train_end + 1, fold.test_start))
            in_train = gap_indices & set(fold.train_idx.tolist())
            in_test  = gap_indices & set(fold.test_idx.tolist())
            assert not in_train, f"Gap indices found in train: {in_train}"
            assert not in_test,  f"Gap indices found in test: {in_test}"


# ── Edge cases and validation ─────────────────────────────────────────────────

class TestEdgeCases:

    def test_short_series_returns_no_splits(self):
        """If train + test > n, no splits are produced."""
        cv = WalkForwardCV(train_size=50, test_size=30)
        assert cv.n_splits(70) == 0
        assert cv.get_splits(70) == []

    def test_exact_fit(self):
        """train_size + test_size == n should produce exactly 1 split."""
        cv    = WalkForwardCV(train_size=50, test_size=50)
        folds = cv.get_splits(100)
        assert len(folds) == 1

    def test_n_splits_matches_iterator(self):
        cv = WalkForwardCV(train_size=60, test_size=20, step=10)
        n  = 500
        count = sum(1 for _ in cv.split(n))
        assert cv.n_splits(n) == count

    def test_get_splits_matches_iterator(self):
        cv     = WalkForwardCV(train_size=50, test_size=10)
        it     = list(cv.split(200))
        listed = cv.get_splits(200)
        assert len(it) == len(listed)
        for a, b in zip(it, listed):
            assert np.array_equal(a.train_idx, b.train_idx)
            assert np.array_equal(a.test_idx,  b.test_idx)

    def test_invalid_train_size(self):
        with pytest.raises(ValueError):
            WalkForwardCV(train_size=0, test_size=10)

    def test_invalid_test_size(self):
        with pytest.raises(ValueError):
            WalkForwardCV(train_size=50, test_size=0)

    def test_invalid_gap(self):
        with pytest.raises(ValueError):
            WalkForwardCV(train_size=50, test_size=10, gap=-1)

    def test_invalid_step(self):
        with pytest.raises(ValueError):
            WalkForwardCV(train_size=50, test_size=10, step=0)

    def test_repr(self):
        cv = WalkForwardCV(train_size=100, test_size=20)
        r  = repr(cv)
        assert "WalkForwardCV" in r
        assert "100" in r
        assert "20" in r


# ── WalkForwardSplit properties ───────────────────────────────────────────────

class TestSplitProperties:

    def _make_fold(self):
        cv    = WalkForwardCV(train_size=50, test_size=10)
        return cv.get_splits(100)[0]

    def test_train_start_end(self):
        fold = self._make_fold()
        assert fold.train_start == fold.train_idx[0]
        assert fold.train_end   == fold.train_idx[-1]

    def test_test_start_end(self):
        fold = self._make_fold()
        assert fold.test_start == fold.test_idx[0]
        assert fold.test_end   == fold.test_idx[-1]

    def test_repr_contains_fold_number(self):
        fold = self._make_fold()
        assert "WalkForwardSplit" in repr(fold)
        assert "fold=0" in repr(fold)
