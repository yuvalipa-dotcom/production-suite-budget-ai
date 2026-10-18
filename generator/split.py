"""Production-level, deterministic data split and grouped cross-validation folds.

Every production belongs to exactly one group (train / test / demo) and every
snapshot of a production inherits that group. Assignment depends only on the
production ids and the seed.

* train -- model selection and hyper-parameter tuning, using production-grouped
  cross-validation only (all snapshots of a production stay in one fold).
* test  -- reserved for the single final evaluation.
* demo  -- reserved for the product demonstration. Never used for training,
  tuning, model selection or evaluation.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import config as C

SPLIT_STREAM = 1
FOLD_STREAM = 2


def assign_split(production_ids: Sequence[str], seed: int = C.SEED,
                 ratios: dict[str, float] = C.SPLIT_RATIOS) -> dict[str, str]:
    """Map each production id to exactly one of train / test / demo (exact counts)."""
    ids = sorted(set(production_ids))
    if len(ids) != len(production_ids):
        raise ValueError("production ids must be unique")
    rng = np.random.default_rng([seed, SPLIT_STREAM])
    order = [ids[i] for i in rng.permutation(len(ids))]
    n = len(ids)
    n_train = int(round(ratios["train"] * n))
    n_test = int(round(ratios["test"] * n))
    groups: dict[str, str] = {}
    for i, pid in enumerate(order):
        groups[pid] = "train" if i < n_train else ("test" if i < n_train + n_test else "demo")
    return groups


def assign_folds(production_ids: Sequence[str], n_folds: int = C.N_FOLDS,
                 seed: int = C.SEED) -> dict[str, int]:
    """Map each (train) production id to one CV fold in [0, n_folds). Grouped by production."""
    ids = sorted(set(production_ids))
    if len(ids) < n_folds:
        raise ValueError("fewer productions than folds")
    rng = np.random.default_rng([seed, FOLD_STREAM])
    order = [ids[i] for i in rng.permutation(len(ids))]
    return {pid: i % n_folds for i, pid in enumerate(order)}
