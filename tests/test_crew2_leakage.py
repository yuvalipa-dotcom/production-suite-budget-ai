"""Crew 2 leakage tests (plan section f; corrections C2 and C5). Offline. ALL DATA IS SYNTHETIC.

1. Outcome-only columns never reach the model input (declared and dynamic trace).
2. Mutating or adding any ledger record dated after a snapshot leaves its features unchanged; the same
   change dated on the snapshot day moves them; a deliberately leaky builder is caught.
3. No production id in two split groups or two folds; groups and folds equal the frozen contract.
4. Test labels are not loaded before the final-evaluation step, and the test set opens once.
5. Every fit receives only rows of its training partition (declared fits and the RowSpy observation).
6. C2: every split actually used, at every tuning and calibration step and in the final refit, is
   recorded and group-disjoint within its partition; the audit catches each kind of violation.
"""

from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import pytest

from crew2.ml import data as D
from crew2.ml import features as F
from crew2.ml import leakage as L
from crew2.ml import protocol as PR
from crew2.ml.resample import LeakageError, Trail, audit, group_partition
from gate import policy_c as P
from tests.crew2_helpers import no_network, small_grids, small_session  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture(scope="module")
def handoff_frame():
    text, _ = D.build_handoff()
    return text, pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def ledgers():
    raw, _ = D.load_raw()
    led, _ = L.cleaned_ledgers(raw)
    return led


# ---------------------------------------------------------------- 1. outcome-only columns

def test_outcome_only_columns_never_reach_the_model_input(handoff_frame):
    outcome = F.outcome_only_columns()
    assert {"final_pre_vat", "financial_close_date", "is_overrun", "budget_gap_ils"} <= outcome
    text, _ = handoff_frame
    train = D.partition(text).train
    for fs in F.FEATURE_SETS:
        cols = list(F.model_input(train, fs).columns)
        assert cols == [F.ROW, *F.FEATURE_SETS[fs]]
        assert not set(cols) & outcome
        assert not set(cols) & {"split", "cv_fold", "production_id", "as_of_date"}
    assert L.declared_trace()["ok"]


def test_dynamic_trace_features_are_identical_without_any_outcome_only_column(ledgers, handoff_frame):
    _, hand = handoff_frame
    rep = L.dynamic_trace(ledgers, hand)
    assert rep == {"snapshots": 1538, "differing": 0, "ok": True}


def test_the_preprocessing_never_sees_an_outcome_column():
    from crew2.ml.models import base_pipeline
    for fs in F.FEATURE_SETS:
        pipe = base_pipeline("LR", {"C": 1.0}, fs, "t")
        used = {c for _, _, cols in pipe.named_steps["prep"].transformers for c in cols}
        assert used == set(F.FEATURE_SETS[fs]) and not used & F.outcome_only_columns()


# ---------------------------------------------------------------- 2. as-of invariance

def test_post_snapshot_ledger_changes_leave_features_unchanged_and_on_day_changes_move_them(ledgers, handoff_frame):
    _, hand = handoff_frame
    keys = L.sample_keys(hand, 8)
    rep = L.as_of_mutations(ledgers, keys)
    assert rep["ok"], {k: rep[k][:5] for k in ("post_dated_moved", "outcome_moved", "on_day_not_moved")}
    assert rep["snapshots"] == 32 and rep["mutations_checked"] >= 32 * 12


def test_the_as_of_check_catches_a_leaky_feature_builder(ledgers, handoff_frame):
    """Two-way proof: a builder that ignores the as-of date for budget revisions is detected."""
    _, hand = handoff_frame

    def leaky(led, pid, stage, as_of):
        late = copy.deepcopy(led)
        for r in late["budget_revisions"]:
            r["effective_date"] = "1900-01-01"                # every revision 'known' from the start
        return L.features_of(late, pid, stage, as_of)

    rep = L.as_of_mutations(ledgers, L.sample_keys(hand, 2), fn=leaky)
    assert not rep["ok"] and any("budget_revision_added" in x for x in rep["post_dated_moved"])


def test_a_final_cost_change_moves_labels_but_no_feature(ledgers):
    pid, as_of = "SYN-0001", "2024-09-04"
    ix0 = P.index({n: pd.DataFrame(v) for n, v in ledgers.items()})
    mutated = L.outcome_mutations(ledgers, pid)["final_pre_vat_changed"]
    ix1 = P.index({n: pd.DataFrame(v) for n, v in mutated.items()})
    assert P.snapshot_features(ix0, pid, "S0", as_of) == P.snapshot_features(ix1, pid, "S0", as_of)
    assert P.snapshot_labels(ix0, pid, as_of)["final_cost_pre_vat"] != P.snapshot_labels(ix1, pid, as_of)[
        "final_cost_pre_vat"]


# ---------------------------------------------------------------- 3. groups

def test_no_production_in_two_groups_or_two_folds(handoff_frame):
    text, _ = handoff_frame
    p = D.partition(text)
    rep = L.group_checks(p.train, p.test_features, p.demo_features, D.frozen_split())
    assert rep["ok"], rep
    assert sorted(p.train["cv_fold"].unique()) == [0, 1, 2, 3, 4]
    assert (p.test_features["cv_fold"] == -1).all() and (p.demo_features["cv_fold"] == -1).all()


def test_the_group_check_detects_a_moved_production(handoff_frame):
    text, _ = handoff_frame
    p = D.partition(text)
    test = p.test_features.copy()
    test.loc[test.index[0], "production_id"] = p.train["production_id"].iloc[0]
    assert not L.group_checks(p.train, test, p.demo_features, D.frozen_split())["ok"]


# ---------------------------------------------------------------- 4. sealed test labels

def test_test_labels_are_not_loaded_before_the_final_evaluation(small_grids, monkeypatch):
    s = PR.Session()
    opens = []
    real_open = D.TestVault.open
    monkeypatch.setattr(D.TestVault, "open", lambda self, shas: (opens.append(len(s.cv)), real_open(self, shas))[1])
    s.build_handoff()
    assert not set(D.LABEL_COLUMNS) & set(s.parts.test_features.columns)
    for run_id in PR.RUNS:
        s.run_cv(run_id)
    s.apply_selection_rule()
    s.fit_final_models()
    assert opens == [] and not s.parts.vault.opened
    with pytest.raises(PR.OrderError):
        PR.Session().evaluate_on_test_once()
    s.evaluate_on_test_once()
    assert opens == [len(PR.RUNS)] and s.test["opened"] == 1
    with pytest.raises(D.TestAlreadyOpened):
        s.evaluate_on_test_once()
    assert len(opens) == 2 and len(s.parts.vault.opens) == 1     # the second attempt was refused


def test_selection_never_receives_test_or_demo_rows(small_session):
    oof_rows = set().union(*(set(df[F.ROW]) for df in small_session.oof.values()))
    assert oof_rows == set(small_session.parts.train[F.ROW])
    assert not oof_rows & small_session.forbidden_rows


def test_the_threshold_is_fixed_at_one_half():
    from crew2.ml.metrics import THRESHOLD, scores
    assert THRESHOLD == 0.5
    s = scores([1, 0, 1, 0], [0.5, 0.49, 0.9, 0.51])
    assert (s["precision"], s["recall"], s["accuracy"]) == (2 / 3, 1.0, 0.75)


# ---------------------------------------------------------------- 5 and 6. fits and splits

@pytest.fixture(scope="module")
def trails(small_grids):
    """Every unit of fitting work with its recorded trail: all outer folds of all candidates, and the final
    refit on all of train."""
    s = PR.Session()
    s.build_handoff()
    train = s.parts.train
    out = []
    for kind in ("B0", "LR", "GB"):
        for f in range(PR.N_FOLDS):
            r = PR.outer_fold(train, "is_overrun", "F27", kind, f, s.forbidden_rows, keep_trail=True)
            out.append((f"fold{f}/{kind}", train[train["cv_fold"] != f], train[train["cv_fold"] == f], r["trail"]))
        _, _, _, _, trail = PR._fit_unit(train, "is_overrun", "F27", kind, -1, s.forbidden_rows)
        out.append((f"final/{kind}", train, train.iloc[0:0], trail))
    return s, out


def test_every_fit_uses_only_training_partition_rows(trails):
    s, units = trails
    forbidden = s.forbidden_rows
    for name, part, valid, trail in units:
        part_rows, valid_rows = set(part[F.ROW]), set(valid[F.ROW])
        for f in trail.fits:
            assert set(f["rows"]) <= part_rows and not set(f["rows"]) & (valid_rows | forbidden), name
        for spy in trail.spy:                                   # what actually reached each estimator
            assert set(spy["rows"]) <= part_rows and not set(spy["rows"]) & (valid_rows | forbidden), name
        if not name.endswith("B0"):
            assert trail.spy and len(trail.spy) == sum(1 for f in trail.fits if f["kind"] == "base"), name


def test_every_recorded_split_is_group_disjoint_inside_its_partition(trails):
    """C2, checked independently of ``audit``: every split used at every tuning and calibration step,
    including the final refit, keeps each production on one side and stays inside its partition."""
    s, units = trails
    pid = dict(zip(s.parts.train[F.ROW], s.parts.train["production_id"]))
    n_splits = 0
    for name, part, _, trail in units:
        part_rows = set(part[F.ROW])
        train_sides = {x["context"]: set(x["train"]) for x in trail.splits}
        for x in trail.splits:
            tr, ho, p = set(x["train"]), set(x["held_out"]), set(x["partition"])
            assert tr and ho and tr | ho <= p <= part_rows, (name, x["context"])
            assert not {pid[r] for r in tr} & {pid[r] for r in ho}, (name, x["context"])
            if "/cal" in x["context"] and "/tune" in x["context"]:   # calibration inside a tuning split
                parent = x["context"].split("/tune")[0] + "/tune" + x["context"].split("/tune")[1].split("/")[0]
                assert p == train_sides[parent], (name, x["context"])
            n_splits += 1
        assert audit(trail, s.forbidden_rows)["violations"] == 0
    assert n_splits > 0
    kinds = {x["context"].split("/")[-1][:3] for _, _, _, t in units for x in t.splits}
    assert {"tun", "cal"} <= kinds


def test_the_audit_catches_each_kind_of_violation(trails):
    s, units = trails
    _, _, _, good = next(u for u in units if u[0] == "fold0/LR")

    def broken(edit):
        t = copy.deepcopy(good)
        edit(t)
        return t

    pid_rows = {}
    for r, p in good.pid_of_row.items():
        pid_rows.setdefault(p, []).append(r)
    multi = next(rows for rows in pid_rows.values() if len(rows) > 1)

    def same_production_both_sides(t):
        x = t.splits[0]
        a, b = multi[0], multi[1]
        x["train"] = sorted(set(x["train"]) - {b} | {a})
        x["held_out"] = sorted(set(x["held_out"]) - {a} | {b})

    def outside_partition(t):
        t.splits[0]["held_out"].append(max(s.forbidden_rows))

    def fit_on_held_out(t):
        f = next(f for f in t.fits if f["kind"] == "base")
        side = next(x for x in t.splits if x["context"] == f["context"])
        f["rows"] = side["held_out"]

    def calibration_on_tuning_held_out(t):
        x = next(x for x in t.splits if "/tune" in x["context"] and "/cal" in x["context"])
        parent = next(p for p in t.splits if p["context"] == x["context"].rsplit("/", 2)[0])
        x["partition"] = parent["held_out"]

    def spy_saw_other_rows(t):
        t.spy[0]["rows"] = t.spy[0]["rows"][1:]

    for edit in (same_production_both_sides, outside_partition, fit_on_held_out, calibration_on_tuning_held_out,
                 spy_saw_other_rows):
        with pytest.raises(LeakageError):
            audit(broken(edit), s.forbidden_rows)


def test_group_partition_is_deterministic_and_group_disjoint():
    pid = {i: f"P{i // 4:03d}" for i in range(400)}
    rows = np.arange(400)
    a = group_partition(pid, rows, 3, [42, 0, 1])
    b = group_partition(pid, rows[::-1], 3, [42, 0, 1])
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all() for x, y in zip(a, b))
    held = np.concatenate([h for _, h in a])
    assert sorted(held) == list(rows)
    for tr, ho in a:
        assert not {pid[r] for r in tr} & {pid[r] for r in ho}
    assert any((x[1] != y[1]).any() for x, y in zip(a, group_partition(pid, rows, 3, [42, 1, 1])))
    t = Trail(pid, root=list(rows))
    for tr, ho in a:
        t.split("x", rows, tr, ho)
    assert audit(t)["splits_checked"] == 3
