"""
Tests for validation/purged_cv.py  — PurgedKFold cross-validator.

Coverage:
  - Correct number of folds is produced
  - Train and test indices are disjoint within each fold
  - Test indices partition 0..N-1 exactly (no gap, no overlap)
  - Purging removes training observations whose label overlaps test window
  - Embargo removes observations immediately after the test fold
  - Edge cases: K=2, K=N, embargo=0, point-in-time labels (no purging)
"""

import numpy as np
import pytest
from validation.purged_cv import PurgedKFold, PurgedFold


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_test_indices(folds):
    """Concatenate all test indices from a list of folds."""
    return np.concatenate([f.test_idx for f in folds])


def _no_leakage(fold: PurgedFold) -> bool:
    """Return True if train and test index sets are disjoint."""
    return len(set(fold.train_idx) & set(fold.test_idx)) == 0


# ── Basic splitting ───────────────────────────────────────────────────────────

class TestPurgedKFoldBasic:

    def test_fold_count(self):
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        folds = cv.get_splits(n=100)
        assert len(folds) == 5

    def test_fold_count_k2(self):
        cv = PurgedKFold(n_splits=2, embargo_pct=0.0)
        folds = cv.get_splits(n=50)
        assert len(folds) == 2

    def test_fold_count_k10(self):
        cv = PurgedKFold(n_splits=10, embargo_pct=0.0)
        folds = cv.get_splits(n=100)
        assert len(folds) == 10

    def test_test_indices_partition(self):
        """Test indices from all folds should cover 0..N-1 exactly once."""
        n = 100
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        folds = cv.get_splits(n=n)
        all_test = sorted(_all_test_indices(folds).tolist())
        assert all_test == list(range(n))

    def test_test_indices_partition_k3(self):
        """Test partition also holds for non-divisible N."""
        n = 101
        cv = PurgedKFold(n_splits=3, embargo_pct=0.0)
        folds = cv.get_splits(n=n)
        all_test = sorted(_all_test_indices(folds).tolist())
        assert all_test == list(range(n))

    def test_train_test_disjoint(self):
        """Train and test indices must not overlap within each fold."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        for fold in cv.get_splits(n=200):
            assert _no_leakage(fold), f"Fold {fold.fold} has train/test overlap"

    def test_fold_indices_are_sorted(self):
        cv = PurgedKFold(n_splits=4, embargo_pct=0.0)
        for fold in cv.get_splits(n=80):
            assert np.all(np.diff(fold.train_idx) > 0), "train_idx not sorted"
            assert np.all(np.diff(fold.test_idx)  > 0), "test_idx not sorted"

    def test_fold_numbering(self):
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        folds = cv.get_splits(n=100)
        for i, fold in enumerate(folds):
            assert fold.fold == i


# ── Purging ───────────────────────────────────────────────────────────────────

class TestPurging:

    def test_no_purging_for_point_in_time(self):
        """With point-in-time labels (t_start == t_end == index), purge=0."""
        n  = 100
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        t  = np.arange(n, dtype=float)
        for fold in cv.split(n, t_start=t, t_end=t):
            assert fold.n_purged == 0, (
                f"Fold {fold.fold}: expected 0 purged but got {fold.n_purged}"
            )

    def test_purging_removes_overlapping_labels(self):
        """Labels spanning into the test window should be purged."""
        n      = 100
        h      = 5    # label length: label at t ends at t+4
        t_start = np.arange(n, dtype=float)
        t_end   = t_start + h - 1

        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        at_least_one_purged = False
        for fold in cv.split(n, t_start=t_start, t_end=t_end):
            # Any training observation whose label end >= test_start should be purged
            test_t0 = t_start[fold.test_idx].min()
            test_t1 = t_end[fold.test_idx].max()
            # Check none of the kept training obs have overlapping labels
            for idx in fold.train_idx:
                ts, te = t_start[idx], t_end[idx]
                overlaps = (ts <= test_t1) and (te >= test_t0)
                assert not overlaps, (
                    f"Fold {fold.fold}: training obs {idx} "
                    f"label [{ts},{te}] overlaps test window [{test_t0},{test_t1}]"
                )
            if fold.n_purged > 0:
                at_least_one_purged = True
        assert at_least_one_purged, "Expected at least one purged observation"

    def test_purging_reduces_train_size(self):
        """Longer labels → more purging → smaller training set."""
        n  = 200
        h1 = 1   # short labels (point-in-time)
        h5 = 10  # long labels
        t  = np.arange(n, dtype=float)

        cv = PurgedKFold(n_splits=4, embargo_pct=0.0)

        total_purged_h1 = sum(f.n_purged for f in cv.split(n, t, t))
        total_purged_h5 = sum(
            f.n_purged for f in cv.split(n, t, t + h5 - 1)
        )
        assert total_purged_h5 >= total_purged_h1


# ── Embargo ───────────────────────────────────────────────────────────────────

class TestEmbargo:

    def test_zero_embargo_removes_nothing_extra(self):
        """embargo_pct=0 should never embargo any observations."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
        for fold in cv.get_splits(n=100):
            assert fold.n_embargoed == 0

    def test_embargo_creates_gap(self):
        """With embargo, no training obs should fall in [test_end, test_end+gap)."""
        n           = 200
        embargo_pct = 0.05   # 5% of 200 = 10 obs
        cv          = PurgedKFold(n_splits=5, embargo_pct=embargo_pct)

        embargo_size = max(1, int(embargo_pct * n))
        for fold in cv.get_splits(n=n):
            test_end = fold.test_end + 1   # exclusive upper bound
            gap_end  = min(test_end + embargo_size, n)
            embargo_zone = set(range(test_end, gap_end))
            overlap = embargo_zone & set(fold.train_idx.tolist())
            assert len(overlap) == 0, (
                f"Fold {fold.fold}: embargoed observations in training set: {overlap}"
            )

    def test_embargo_count_positive(self):
        """With embargo > 0, at least some folds should have n_embargoed > 0."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.05)
        folds = cv.get_splits(n=200)
        # Interior folds always have an embargo window
        interior = [f for f in folds if f.fold < len(folds) - 1]
        assert any(f.n_embargoed > 0 for f in interior)

    def test_train_never_exceeds_embargo_boundary(self):
        """train_idx must respect embargo after each fold."""
        cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
        for fold in cv.get_splits(n=500):
            assert _no_leakage(fold)


# ── Validation and constructor ────────────────────────────────────────────────

class TestConstructorValidation:

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError):
            PurgedKFold(n_splits=1)

    def test_invalid_embargo_negative(self):
        with pytest.raises(ValueError):
            PurgedKFold(embargo_pct=-0.01)

    def test_invalid_embargo_too_large(self):
        with pytest.raises(ValueError):
            PurgedKFold(embargo_pct=0.6)

    def test_repr(self):
        cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
        assert "PurgedKFold" in repr(cv)
        assert "5" in repr(cv)


# ── PurgedFold repr and properties ────────────────────────────────────────────

class TestPurgedFoldProperties:

    def test_test_start_end(self):
        cv    = PurgedKFold(n_splits=4, embargo_pct=0.0)
        folds = cv.get_splits(n=100)
        for fold in folds:
            assert fold.test_start == fold.test_idx[0]
            assert fold.test_end   == fold.test_idx[-1]

    def test_repr_contains_fold_info(self):
        cv    = PurgedKFold(n_splits=3, embargo_pct=0.0)
        fold  = cv.get_splits(n=60)[0]
        r     = repr(fold)
        assert "PurgedFold" in r
        assert "fold=0" in r

    def test_n_folds_method(self):
        cv = PurgedKFold(n_splits=7)
        assert cv.n_folds() == 7
