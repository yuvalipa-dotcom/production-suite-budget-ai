"""Policy C (P1-R2): target integrity, snapshot eligibility, outcome-only isolation.

Expected outcomes are derived at runtime from the generated ground truth (never
hard-coded), so no tracked file reveals which records carry a data-quality problem.
"""

import copy
import json
from collections import Counter

import pandas as pd
import pytest

from generator import config as C
from generator.cli import RAW_FILES
from generator.simulate import simulate_all
from generator.snapshots import snapshot_row, stage_days
from tests.conftest import ACTIVE_DATA, REPO
from tests.raw_repair import COST_TABLES, STAGES, apply_policy_c, target_records_dropped
from tests.reconstruct import (FEATURE_RULES, FEATURE_SOURCES, LABELS, as_csv_text, index,
                               snapshot_features, snapshot_labels)

LEDGERS = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs")
CONTRACT = json.loads((REPO / "contracts" / "column_roles.json").read_text(encoding="utf-8"))
OUTCOME_ONLY = {f"{t}.{c}" for t, cols in CONTRACT["tables"].items() for c, s in cols.items()
                if s["role"] == "outcome-only"}


@pytest.fixture(scope="module")
def policy(raw_text):
    return apply_policy_c(raw_text)


@pytest.fixture(scope="module")
def clean_snaps():
    return pd.read_csv(ACTIVE_DATA / "snapshots.csv", dtype=str, keep_default_na=False)


def _reproduce(policy):
    idx = index({n: policy.tables[n] for n in LEDGERS})
    rows = []
    for r in policy.eligibility.to_dict("records"):
        labels = snapshot_labels(idx, r["production_id"], r["as_of_date"])
        feats = snapshot_features(idx, r["production_id"], r["stage"], r["as_of_date"]) if r["eligible"] else None
        rows.append({**r, "labels": labels, "features": feats})
    return rows


# ---------- R3.1 / R3.2: target integrity and the Policy A guard ----------

def test_target_integrity_all_1600(policy, clean_snaps):
    assert all(policy.target_reliable.values()) and len(policy.target_reliable) == 400
    rows = {(r["production_id"], r["stage"]): r for r in _reproduce(policy)}
    mismatches = [(k, c) for k, t in ((k, t) for k, t in clean_snaps.set_index(["production_id", "stage"]).iterrows())
                  for c in LABELS if as_csv_text(rows[k]["labels"][c]) != t[c]]
    assert len(rows) == 1600 and mismatches == []


def test_policy_a_guard(policy, raw_text):
    assert target_records_dropped(raw_text, policy.tables) == []
    # the guard fails when a cost record with a present amount is removed (two-way proof)
    broken = {n: df.copy() for n, df in policy.tables.items()}
    broken["commitments"] = broken["commitments"].iloc[1:]
    assert target_records_dropped(raw_text, broken) != []


def test_no_cost_record_is_excluded(policy, raw_text):
    for table in COST_TABLES:
        key = {"commitments": "commitment_id", "manual_costs": "manual_cost_id"}[table]
        assert set(policy.tables[table][key]) == set(raw_text[table][key].str.strip())


# ---------- R3.3 / R3.4 / R3.7: eligibility and full comparison ----------

def test_ineligible_set_and_reasons_match_expected(policy, expected_ineligible):
    got = {(r.production_id, r.stage): r.reason for r in policy.eligibility.itertuples() if not r.eligible}
    assert got == expected_ineligible


def test_full_comparison_of_all_1600_snapshots(policy, clean_snaps, expected_ineligible):
    """Every row has an explicit expected outcome: eligible-and-identical, or ineligible with its reason."""
    truth = clean_snaps.set_index(["production_id", "stage"], drop=False)
    outcomes = Counter()
    for r in _reproduce(policy):
        key = (r["production_id"], r["stage"])
        if key in expected_ineligible:
            assert not r["eligible"] and r["reason"] == expected_ineligible[key]
            assert r["features"] is None                      # nothing filled for an ineligible snapshot
            outcomes["ineligible"] += 1
        else:
            assert r["eligible"] and r["reason"] == ""
            t = truth.loc[key]
            assert {c: as_csv_text(r["features"][c]) for c in FEATURE_RULES} == {c: t[c] for c in FEATURE_RULES}, key
            outcomes["eligible_identical"] += 1
    assert sum(outcomes.values()) == 1600


def test_exclusion_counts(policy, generated):
    groups = generated["productions"].set_index("production_id")["split"]
    inel = policy.eligibility[~policy.eligibility["eligible"]]
    assert len(inel) == 62
    assert Counter(inel["production_id"].map(groups)) == {"train": 36, "test": 14, "demo": 12}
    assert Counter(inel["reason"]) == {"commitment_category_missing": 50, "manual_cost_undated": 12}
    assert sum(1 for v in policy.target_reliable.values() if not v) == 0


def test_cancelled_commitment_with_missing_field_does_not_block_later_snapshots(policy, raw_text):
    """The rule is 'active' (selected and not yet cancelled), not merely 'selected'."""
    c = raw_text["commitments"].apply(lambda s: s.str.strip())
    cancelled = c[(c["category"] == "") & (c["cancelled_date"] != "")]
    assert len(cancelled) > 0
    el = policy.eligibility.set_index(["production_id", "stage"])
    for row in cancelled.itertuples():
        last = el.loc[(row.production_id, "S3")]
        if last.as_of_date >= row.cancelled_date:
            assert bool(last.eligible)


# ---------- R3.5 / R3.6: no fabrication, frozen groups ----------

def test_no_value_is_filled(policy, raw_text):
    for table in COST_TABLES:
        key = {"commitments": "commitment_id", "manual_costs": "manual_cost_id"}[table]
        raw = raw_text[table].apply(lambda s: s.str.strip()).drop_duplicates(key).set_index(key)
        cleaned = policy.tables[table].set_index(key)
        for col in raw.columns:
            blank = raw.index[raw[col] == ""]
            assert (cleaned.loc[blank, col] == "").all(), (table, col)


def test_payments_are_excluded_without_eligibility_effect(policy, defect_log):
    expected = set(defect_log.loc[(defect_log["table"] == "payments") & (defect_log["recoverable"] == "False"),
                                  "record_id"])
    assert policy.excluded_payments == expected
    assert not policy.eligibility["reason"].str.contains("payment").any()


def test_groups_and_folds_frozen(policy, generated):
    v1 = pd.read_csv(REPO / "data" / "synthetic" / "v1" / "productions.csv")
    v2 = generated["productions"]
    assert v1[["production_id", "split", "cv_fold"]].equals(v2[["production_id", "split", "cv_fold"]])
    manifest = json.loads((ACTIVE_DATA / "manifest.json").read_text(encoding="utf-8"))
    assert v2["split"].value_counts().to_dict() == manifest["split_productions"]
    groups = v2.set_index("production_id")["split"]
    kept = policy.eligibility[policy.eligibility["eligible"]]
    retained = kept.assign(group=kept["production_id"].map(groups))
    assert retained["group"].notna().all()
    assert set(retained["production_id"]) <= set(groups.index)


# ---------- R3.8: outcome-only isolation ----------

def _strip_outcome_only(tables):
    out = {}
    for t, df in tables.items():
        drop = [c for c in df.columns if f"{t}.{c}" in OUTCOME_ONLY]
        out[t] = df.drop(columns=drop)
    return out


def test_8c_declared_sources_contain_no_outcome_only_column():
    assert set(FEATURE_SOURCES) == set(FEATURE_RULES)
    leaks = {f: s & OUTCOME_ONLY for f, s in FEATURE_SOURCES.items() if s & OUTCOME_ONLY}
    assert leaks == {}
    assert {"commitments.final_pre_vat", "productions.financial_close_date"} <= OUTCOME_ONLY


def test_8c_features_computable_without_any_outcome_only_column(generated):
    """Dynamic trace: features are rebuilt from tables with every outcome-only column removed."""
    full = index({n: generated[n] for n in LEDGERS})
    bare = index(_strip_outcome_only({n: generated[n] for n in LEDGERS}))
    for r in generated["snapshots"].head(400).itertuples():
        assert snapshot_features(bare, r.production_id, r.stage, r.as_of_date) == \
               snapshot_features(full, r.production_id, r.stage, r.as_of_date)


def test_8c_contract_covers_every_column_once(generated, raw_text):
    for t, df in {**{n: generated[n] for n in generated if isinstance(generated[n], pd.DataFrame)},
                  "snapshot_schedule": raw_text["snapshot_schedule"]}.items():
        assert set(df.columns) == set(CONTRACT["tables"][t]), t
    for t, cols in CONTRACT["tables"].items():
        for c, spec in cols.items():
            assert spec["role"] in ("feature", "outcome-only", "metadata"), (t, c)


def _perturbed_final(tables, cid, delta):
    t = {n: df.copy() for n, df in tables.items()}
    c = t["commitments"]
    c.loc[c["commitment_id"] == cid, "final_pre_vat"] = c.loc[c["commitment_id"] == cid, "final_pre_vat"] + delta
    return t


def test_8a_final_cost_perturbation(generated):
    base_tables = {n: generated[n] for n in LEDGERS}
    base = index(base_tables)
    snaps = generated["snapshots"]
    com = generated["commitments"]
    cases = Counter()
    for pid in sorted(snaps["production_id"].unique())[:60]:
        live = com[(com["production_id"] == pid) & (com["cancelled_date"] == "")]
        cid = live["commitment_id"].iloc[0]
        rows = snaps[snaps["production_id"] == pid]
        gap0 = {r.stage: snapshot_labels(base, pid, r.as_of_date)["budget_gap_ils"] for r in rows.itertuples()}
        s0 = gap0["S0"]
        # one delta that crosses the S0 threshold, and one that does not
        cross = (-(s0 + 1000.0)) if s0 > 0.5 else (1000.0 - s0)
        keep = (-s0 / 2) if s0 > 2 else (-(abs(s0) + 500.0) if s0 <= 0.5 else 0.25)
        for delta, kind in ((cross, "crossing"), (keep, "non_crossing")):
            idx = index(_perturbed_final(base_tables, cid, delta))
            for r in rows.itertuples():
                assert snapshot_features(idx, pid, r.stage, r.as_of_date) == \
                       snapshot_features(base, pid, r.stage, r.as_of_date)
                new = snapshot_labels(idx, pid, r.as_of_date)
                assert abs(new["budget_gap_ils"] - (gap0[r.stage] + delta)) < 1e-6
                old_label, new_label = gap0[r.stage] > 0.5, new["budget_gap_ils"] > 0.5
                assert new["is_overrun"] == new_label
                if old_label != new_label:
                    cases["to_no_overrun" if old_label else "to_overrun"] += 1
                else:
                    cases["unchanged"] += 1
            cases[kind] += 1
    assert cases["to_no_overrun"] > 0 and cases["to_overrun"] > 0 and cases["unchanged"] > 0


def test_8a_generator_features_ignore_final_costs():
    for p in simulate_all(C.SEED)[:40]:
        q = copy.deepcopy(p)
        for c in q.commitments:
            c.final_pre_vat = (c.final_pre_vat or 0.0) * 3 + 7
        for stage, day in stage_days(p).items():
            a, b = snapshot_row(p, stage, day), snapshot_row(q, stage, day)
            assert {k: a[k] for k in FEATURE_RULES} == {k: b[k] for k in FEATURE_RULES}


def test_8b_close_date_perturbation_changes_nothing(generated):
    base_tables = {n: generated[n] for n in LEDGERS}
    moved = {n: df.copy() for n, df in base_tables.items()}
    moved["productions"]["financial_close_date"] = "2099-12-31"
    a, b = index(base_tables), index(moved)
    for r in generated["snapshots"].itertuples():
        assert snapshot_features(a, r.production_id, r.stage, r.as_of_date) == \
               snapshot_features(b, r.production_id, r.stage, r.as_of_date)
        assert snapshot_labels(a, r.production_id, r.as_of_date) == snapshot_labels(b, r.production_id, r.as_of_date)
    for p in simulate_all(C.SEED)[:40]:
        q = copy.deepcopy(p)
        q.close_after_event_days += 200
        for stage, day in stage_days(p).items():
            assert snapshot_row(p, stage, day) == snapshot_row(q, stage, day)


# ---------- R3.9: determinism ----------

def test_cleaning_is_deterministic(raw_text):
    a, b = apply_policy_c(raw_text), apply_policy_c(raw_text)
    assert a.eligibility.equals(b.eligibility)
    assert a.target_reliable == b.target_reliable and a.excluded_payments == b.excluded_payments
    assert a.issues == b.issues
    for n in RAW_FILES:
        assert a.tables[n].equals(b.tables[n])
