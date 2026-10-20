"""Group-disjoint resampling and its audit trail (Addendum P4-1, section 6; correction C2).

Every inner split is built explicitly from the production ids of the CURRENT training
partition and is recorded with that partition. Every fit records the rows it was given
(and, through ``RowSpy``, the rows that actually reached the estimator). ``audit`` fails
when a split puts one production on both sides, when a split or a fit uses a row outside
its partition, or when a fit's rows differ from the side of the split it declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FINAL_FOLD_CODE = 5              # seed code of the final refit on all of train (Addendum P4-1a)


class LeakageError(AssertionError):
    pass


def seed_fold(fold: int) -> int:
    """The outer-fold component of a seed list: folds 0-4 as is, the final refit (-1) as 5."""
    return FINAL_FOLD_CODE if fold == -1 else int(fold)


@dataclass
class Trail:
    """Splits and fits of one unit of work (an outer fold or the final refit)."""
    pid_of_row: dict[int, str]
    root: list[int]                  # the outer training partition: no split or fit may leave it
    splits: list[dict] = field(default_factory=list)
    fits: list[dict] = field(default_factory=list)
    spy: list[dict] = field(default_factory=list)

    def split(self, context: str, partition: np.ndarray, train: np.ndarray, held: np.ndarray) -> None:
        self.splits.append({"context": context, "partition": _ids(partition), "train": _ids(train),
                            "held_out": _ids(held)})

    def fit(self, context: str, kind: str, rows: np.ndarray, partition: np.ndarray, side: str) -> None:
        self.fits.append({"context": context, "kind": kind, "rows": _ids(rows), "partition": _ids(partition),
                          "side": side})

    def merge(self, other: "Trail") -> None:
        self.splits += other.splits
        self.fits += other.fits
        self.spy += other.spy


def _ids(a) -> list[int]:
    return sorted(int(x) for x in a)


def group_partition(pid_of_row: dict[int, str], rows: np.ndarray, k: int, seed: list[int]
                    ) -> list[tuple[np.ndarray, np.ndarray]]:
    """K(P, k, seed): sorted unique production ids of ``rows``, permuted with ``default_rng(seed)``; the id at
    permuted position i goes to part i mod k. Returns k (train, held-out) pairs of row ids."""
    rows = np.asarray(sorted(int(r) for r in rows))
    pids = sorted({pid_of_row[int(r)] for r in rows})
    if len(pids) < k:
        raise LeakageError("fewer productions than parts")
    perm = np.random.default_rng(seed).permutation(len(pids))
    part = {pids[p]: i % k for i, p in enumerate(perm)}
    labels = np.array([part[pid_of_row[int(r)]] for r in rows])
    return [(rows[labels != j], rows[labels == j]) for j in range(k)]


def audit(trail: Trail, forbidden_rows: set[int] = frozenset()) -> dict:
    """Check every recorded split and fit; raise ``LeakageError`` on the first violation.

    A partition is either the root (the outer training partition) or exactly the train side of an
    enclosing split (its context is a prefix), so a nested step can never reach a held-out side."""
    pid = trail.pid_of_row
    root = set(trail.root)
    parents = {s["context"]: set(s["train"]) for s in trail.splits}

    def enclosed(context: str, part: set[int]) -> bool:
        if part == root:
            return True
        return any(context.startswith(c + "/") and part == t for c, t in parents.items())

    for s in trail.splits:
        part, tr, ho = set(s["partition"]), set(s["train"]), set(s["held_out"])
        if not tr or not ho:
            raise LeakageError(f"{s['context']}: empty side")
        if not part <= root or not enclosed(s["context"].rsplit("/", 1)[0], part):
            raise LeakageError(f"{s['context']}: partition is not the training partition or an enclosing train side")
        if not (tr | ho) <= part:
            raise LeakageError(f"{s['context']}: split uses rows outside its partition")
        if tr & ho:
            raise LeakageError(f"{s['context']}: a row on both sides")
        shared = {pid[r] for r in tr} & {pid[r] for r in ho}
        if shared:
            raise LeakageError(f"{s['context']}: production {sorted(shared)[0]} on both sides")
        if (tr | ho) & forbidden_rows:
            raise LeakageError(f"{s['context']}: split touches a test or demo row")
    sides = {(s["context"], "train"): set(s["train"]) for s in trail.splits}
    sides |= {(s["context"], "held_out"): set(s["held_out"]) for s in trail.splits}
    for f in trail.fits:
        rows, part = set(f["rows"]), set(f["partition"])
        if not part <= root or not rows <= part:
            raise LeakageError(f"{f['context']}: {f['kind']} fit on rows outside its partition")
        if rows & forbidden_rows:
            raise LeakageError(f"{f['context']}: {f['kind']} fit on a test or demo row")
        if f["side"] != "partition" and sides.get((f["context"], f["side"])) != rows:
            raise LeakageError(f"{f['context']}: {f['kind']} fit rows differ from the declared split side")
    declared = {(f["context"], tuple(f["rows"])) for f in trail.fits if f["kind"] == "base"}
    for s in trail.spy:
        if (s["context"], tuple(s["rows"])) not in declared:
            raise LeakageError(f"{s['context']}: the estimator saw rows that no declared fit used")
    return {"splits_checked": len(trail.splits), "fits_checked": len(trail.fits),
            "estimator_fits_observed": len(trail.spy), "violations": 0}
