"""Production-level split and grouped cross-validation folds."""

import pytest

from generator import config as C
from generator.split import assign_folds, assign_split


def test_split_counts(generated):
    counts = generated["productions"]["split"].value_counts().to_dict()
    assert counts == {"train": 280, "test": 80, "demo": 40}


def test_no_production_in_more_than_one_group(snaps):
    groups_per_production = snaps.groupby("production_id")["split"].nunique()
    assert groups_per_production.eq(1).all()
    ids = {g: set(snaps.loc[snaps["split"] == g, "production_id"]) for g in ("train", "test", "demo")}
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["demo"])
    assert ids["test"].isdisjoint(ids["demo"])


def test_snapshot_split_matches_production_split(generated, snaps):
    by_prod = generated["productions"].set_index("production_id")["split"]
    assert (snaps["split"] == snaps["production_id"].map(by_prod)).all()


def test_split_is_deterministic_and_seed_dependent():
    ids = [f"SYN-{i:04d}" for i in range(1, 401)]
    assert assign_split(ids, 42) == assign_split(list(reversed(ids)), 42)
    assert assign_split(ids, 42) != assign_split(ids, 43)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError):
        assign_split(["SYN-0001", "SYN-0001"], 42)


def test_folds_only_on_train(snaps):
    train = snaps[snaps["split"] == "train"]
    other = snaps[snaps["split"] != "train"]
    assert set(train["cv_fold"]) == set(range(C.N_FOLDS))
    assert (other["cv_fold"] == -1).all()


def test_no_production_spans_two_folds(snaps):
    train = snaps[snaps["split"] == "train"]
    assert train.groupby("production_id")["cv_fold"].nunique().eq(1).all()


def test_folds_are_balanced(generated):
    train = generated["productions"][generated["productions"]["split"] == "train"]
    assert train["cv_fold"].value_counts().eq(280 // C.N_FOLDS).all()


def test_fold_function_is_grouped_and_deterministic():
    ids = [f"SYN-{i:04d}" for i in range(1, 281)]
    a, b = assign_folds(ids, 5, 42), assign_folds(ids, 5, 42)
    assert a == b
    assert set(a) == set(ids)
    assert set(a.values()) == set(range(5))
    # snapshot-level expansion keeps every production in a single fold
    rows = [(pid, stage) for pid in ids for stage in C.STAGES]
    folds_per_pid = {}
    for pid, _ in rows:
        folds_per_pid.setdefault(pid, set()).add(a[pid])
    assert all(len(f) == 1 for f in folds_per_pid.values())
