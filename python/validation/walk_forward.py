"""
Walk-Forward Cross-Validation
==============================
Produces train/test index splits that respect the temporal ordering of
financial time series data.  Two modes are supported:

  Rolling (expanding = False):
    Training window has fixed size `train_size`.
    Both train and test windows slide forward each step.

    | train_size | test_size | ←step→ | train_size | test_size | …

  Anchored (expanding = True):
    Training window always starts at index 0 and grows each step.
    Only the test window slides forward.

    |───── growing train ─────| test | …

In both modes each test period strictly follows its training period —
no look-ahead leakage is possible by construction.

An optional `gap` parameter drops observations between train end and test
start to account for signal publication delay or settlement lag.

Reference:
  Lopez de Prado (2018). "Advances in Financial Machine Learning." Wiley.
  Chapter 7 – Cross-Validation in Finance.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple


# ── Split container ───────────────────────────────────────────────────────────

@dataclass
class WalkForwardSplit:
    """A single train/test split produced by the walk-forward splitter."""
    fold:       int
    train_idx:  np.ndarray   # integer indices into the original series
    test_idx:   np.ndarray

    @property
    def train_start(self) -> int:
        return int(self.train_idx[0])

    @property
    def train_end(self) -> int:
        return int(self.train_idx[-1])

    @property
    def test_start(self) -> int:
        return int(self.test_idx[0])

    @property
    def test_end(self) -> int:
        return int(self.test_idx[-1])

    def __repr__(self) -> str:
        return (f"WalkForwardSplit(fold={self.fold}, "
                f"train=[{self.train_start}:{self.train_end+1}], "
                f"test=[{self.test_start}:{self.test_end+1}])")


# ── WalkForwardCV ─────────────────────────────────────────────────────────────

class WalkForwardCV:
    """
    Walk-forward cross-validation splitter.

    Parameters
    ----------
    train_size : number of observations in the training window.
    test_size  : number of observations in each test period.
    step       : how many observations to advance each fold.
                 Defaults to test_size (non-overlapping test periods).
    gap        : observations to skip between train end and test start
                 (e.g. to account for signal latency).  Default 0.
    expanding  : if True, use an anchored (expanding) training window;
                 if False (default), use a rolling fixed-size window.
    min_train_size : minimum training observations required
                     (only relevant when expanding=True).  Default train_size.

    Usage
    -----
    >>> cv = WalkForwardCV(train_size=252, test_size=63)
    >>> for split in cv.split(n=1000):
    ...     X_train = data[split.train_idx]
    ...     X_test  = data[split.test_idx]
    ...     model.fit(X_train); preds = model.predict(X_test)
    """

    def __init__(
        self,
        train_size:     int,
        test_size:      int,
        step:           Optional[int] = None,
        gap:            int = 0,
        expanding:      bool = False,
        min_train_size: Optional[int] = None,
    ) -> None:
        if train_size < 1:
            raise ValueError("train_size must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")

        self.train_size     = train_size
        self.test_size      = test_size
        self.step           = step if step is not None else test_size
        self.gap            = gap
        self.expanding      = expanding
        self.min_train_size = min_train_size if min_train_size is not None else train_size

        if self.step < 1:
            raise ValueError("step must be >= 1")

    def split(self, n: int) -> Iterator[WalkForwardSplit]:
        """
        Generate walk-forward splits for a series of length n.

        Parameters
        ----------
        n : total number of observations.

        Yields
        ------
        WalkForwardSplit for each fold.
        """
        fold = 0
        test_start = self.train_size + self.gap

        while test_start + self.test_size <= n:
            test_end = test_start + self.test_size   # exclusive

            if self.expanding:
                train_start = 0
                train_end   = test_start - self.gap   # exclusive
            else:
                train_start = test_start - self.gap - self.train_size
                train_end   = test_start - self.gap   # exclusive

            train_size = train_end - train_start
            if train_size < self.min_train_size:
                test_start += self.step
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx  = np.arange(test_start, test_end)

            yield WalkForwardSplit(
                fold      = fold,
                train_idx = train_idx,
                test_idx  = test_idx,
            )

            fold       += 1
            test_start += self.step

    def n_splits(self, n: int) -> int:
        """Number of splits that will be generated for a series of length n."""
        return sum(1 for _ in self.split(n))

    def get_splits(self, n: int) -> List[WalkForwardSplit]:
        """Return all splits as a list."""
        return list(self.split(n))

    def __repr__(self) -> str:
        mode = "expanding" if self.expanding else "rolling"
        return (f"WalkForwardCV(train={self.train_size}, test={self.test_size}, "
                f"step={self.step}, gap={self.gap}, mode={mode})")
