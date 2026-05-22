"""
Purged K-Fold Cross-Validation
================================
Implements Lopez de Prado's (2018) purged k-fold CV for financial ML, which
prevents label-overlap leakage that standard k-fold produces when labels span
multiple time periods (e.g. forward-looking returns).

Key concepts
------------
  Purging  : Remove training observations whose *label interval* overlaps with
             the test fold's observation interval.  Prevents information from
             the test period leaking into the training set via multi-period
             labels.

  Embargo  : After the test fold, remove an additional buffer of `embargo_pct`
             of the total series from the training set.  Guards against
             autocorrelated features carrying information forward.

If only point-in-time observations are used (i.e. each label corresponds to a
single time-step), purging has no effect and the method reduces to ordinary
time-series k-fold.

Reference:
  Lopez de Prado (2018). "Advances in Financial Machine Learning."
  Wiley, Chapter 7.  Algorithm 7.3.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


# ── Split container ───────────────────────────────────────────────────────────

@dataclass
class PurgedFold:
    """One fold from PurgedKFold.split()."""
    fold:       int
    train_idx:  np.ndarray   # sorted integer indices for training
    test_idx:   np.ndarray   # sorted integer indices for testing
    n_purged:   int          # observations removed from train by purging
    n_embargoed: int         # observations removed from train by embargo

    @property
    def test_start(self) -> int:
        return int(self.test_idx[0])

    @property
    def test_end(self) -> int:
        return int(self.test_idx[-1])

    def __repr__(self) -> str:
        return (f"PurgedFold(fold={self.fold}, "
                f"n_train={len(self.train_idx)}, "
                f"n_test={len(self.test_idx)}, "
                f"n_purged={self.n_purged}, "
                f"n_embargoed={self.n_embargoed})")


# ── PurgedKFold ───────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    Purged K-Fold cross-validator for financial time-series labels.

    Parameters
    ----------
    n_splits    : number of folds K (default 5).
    embargo_pct : fraction of observations to embargo after each test fold.
                  e.g. 0.01 embargoes 1% of total series length.  Default 0.01.

    Usage — point-in-time labels
    ----------------------------
    >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    >>> for fold in cv.split(n=1000):
    ...     X_train = X[fold.train_idx]
    ...     X_test  = X[fold.test_idx]

    Usage — overlapping labels (e.g. 5-day forward returns)
    --------------------------------------------------------
    >>> # t_start[i], t_end[i] are the start and end timestamps of label i
    >>> for fold in cv.split(n=1000, t_start=t0, t_end=t1):
    ...     X_train = X[fold.train_idx]

    When t_start and t_end are provided, any training observation whose
    label interval [t_start[j], t_end[j]] overlaps the test window
    [min(t_start[test]), max(t_end[test])] is purged.
    """

    def __init__(
        self,
        n_splits:    int   = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0.0 <= embargo_pct < 0.5):
            raise ValueError("embargo_pct must be in [0, 0.5)")
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        n:       int,
        t_start: Optional[np.ndarray] = None,
        t_end:   Optional[np.ndarray] = None,
    ) -> Iterator[PurgedFold]:
        """
        Generate purged k-fold splits.

        Parameters
        ----------
        n       : number of observations.
        t_start : array of shape (n,) — start time index of each label.
                  If None, uses integer indices (point-in-time assumed).
        t_end   : array of shape (n,) — end time index of each label.
                  If None, same as t_start (point-in-time).

        Yields
        ------
        PurgedFold for each fold k=0..K-1.
        """
        if t_start is None:
            t_start = np.arange(n, dtype=float)
        if t_end is None:
            t_end = t_start.copy()

        t_start = np.asarray(t_start, dtype=float)
        t_end   = np.asarray(t_end,   dtype=float)

        embargo_size = max(1, int(self.embargo_pct * n)) if self.embargo_pct > 0 else 0

        indices  = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1   # distribute remainder

        current = 0
        for k, fold_size in enumerate(fold_sizes):
            # Test fold: [current, current+fold_size)
            test_start = current
            test_end   = current + fold_size
            test_idx   = indices[test_start:test_end]

            # Test label window
            test_t0 = t_start[test_idx].min()
            test_t1 = t_end[test_idx].max()

            # All indices NOT in the test fold
            train_candidates = np.concatenate([indices[:test_start],
                                               indices[test_end:]])

            # Purge: remove training obs whose label interval overlaps test window
            purge_mask = (
                (t_start[train_candidates] <= test_t1) &
                (t_end[train_candidates]   >= test_t0)
            )
            n_purged = int(purge_mask.sum())
            after_purge = train_candidates[~purge_mask]

            # Embargo: remove indices immediately after test fold end
            if embargo_size > 0:
                embargo_end = min(test_end + embargo_size, n)
                embargo_mask = (after_purge >= test_end) & (after_purge < embargo_end)
                n_embargoed  = int(embargo_mask.sum())
                train_idx    = after_purge[~embargo_mask]
            else:
                n_embargoed = 0
                train_idx   = after_purge

            yield PurgedFold(
                fold        = k,
                train_idx   = np.sort(train_idx),
                test_idx    = np.sort(test_idx),
                n_purged    = n_purged,
                n_embargoed = n_embargoed,
            )

            current = test_end

    def get_splits(
        self,
        n:       int,
        t_start: Optional[np.ndarray] = None,
        t_end:   Optional[np.ndarray] = None,
    ) -> List[PurgedFold]:
        """Return all folds as a list."""
        return list(self.split(n, t_start, t_end))

    def n_folds(self) -> int:
        return self.n_splits

    def __repr__(self) -> str:
        return (f"PurgedKFold(n_splits={self.n_splits}, "
                f"embargo_pct={self.embargo_pct})")
