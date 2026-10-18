"""The frozen split reference contract (contracts/frozen_split.json). ALL DATA IS SYNTHETIC."""

import copy
import json
from collections import Counter

import pandas as pd
import pytest

from gate.checks import validate_frozen
from generator import config as C
from generator.split import assign_folds, assign_split
from tests.conftest import DATA, REPO
from tools.build_frozen_split import build

FROZEN = REPO / "contracts" / "frozen_split.json"


@pytest.fixture(scope="module")
def frozen():
    return json.loads(FROZEN.read_text(encoding="utf-8"))


def _assignments(doc):
    return {r["production_id"]: (r["split"], r["cv_fold"]) for r in doc["productions"]}


def test_rebuild_is_byte_identical():
    assert FROZEN.read_bytes() == build().encode("utf-8")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_matches_committed_assignments(frozen, version):
    df = pd.read_csv(DATA / version / "productions.csv", dtype=str, keep_default_na=False)
    assert _assignments(frozen) == {r.production_id: (r.split, int(r.cv_fold)) for r in df.itertuples()}


def test_matches_seeded_generation_rules(frozen):
    ids = [r["production_id"] for r in frozen["productions"]]
    groups = assign_split(ids, C.SEED)
    folds = assign_folds([p for p, g in groups.items() if g == "train"], C.N_FOLDS, C.SEED)
    assert _assignments(frozen) == {p: (groups[p], folds.get(p, -1)) for p in ids}


def test_counts_folds_and_ordering(frozen):
    a = _assignments(frozen)
    ids = [r["production_id"] for r in frozen["productions"]]
    assert ids == sorted(ids) and len(set(ids)) == 400
    assert Counter(g for g, _ in a.values()) == {"train": 280, "test": 80, "demo": 40}
    assert Counter(f for g, f in a.values() if g == "train") == {k: 56 for k in range(5)}
    assert all(f == -1 for g, f in a.values() if g != "train")
    assert frozen["version"] == "1.0.0" and frozen["synthetic"] is True
    assert set(frozen["provenance"]["source_files_sha256"]) == {"data/synthetic/v2/productions.csv",
                                                                "data/synthetic/v1/productions.csv"}


def test_fold_memberships_are_the_original_ones(frozen):
    v1 = pd.read_csv(DATA / "v1" / "productions.csv", dtype=str, keep_default_na=False)
    original = {k: set(v1.loc[v1["cv_fold"] == str(k), "production_id"]) for k in range(5)}
    a = _assignments(frozen)
    assert {k: {p for p, (g, f) in a.items() if f == k} for k in range(5)} == original


def test_gate_validator_accepts_the_contract(frozen):
    assert validate_frozen(frozen)[0] == []


def _mutate(doc, how):
    d = copy.deepcopy(doc)
    rows = d["productions"]
    train = next(r for r in rows if r["split"] == "train")
    test = next(r for r in rows if r["split"] == "test")
    if how == "test_gets_train_fold":
        test["cv_fold"] = 0
    elif how == "group_moved":
        test["split"], test["cv_fold"] = "train", 0
    elif how == "fold_reassigned":
        train["cv_fold"] = (train["cv_fold"] + 1) % 5
    elif how == "duplicate":
        rows.insert(1, dict(rows[0]))
    elif how == "unsorted":
        rows[0], rows[1] = rows[1], rows[0]
    elif how == "dropped":
        rows.pop()
    return d


@pytest.mark.parametrize("how", ["test_gets_train_fold", "group_moved", "fold_reassigned", "duplicate",
                                 "unsorted", "dropped"])
def test_gate_validator_rejects_invalid_variants(frozen, how):
    assert validate_frozen(_mutate(frozen, how))[0] != []
