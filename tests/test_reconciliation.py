"""Snapshot-time reconciliation (D1, D2) and future-data mutation tests (D3)."""

import copy
import json

import pandas as pd

from generator import config as C
from generator.simulate import Commitment, ManualCost, simulate_all
from generator.snapshots import snapshot_row, stage_days
from tests.reconstruct import D1, D2, FEATURE_RULES, as_csv_text, features_at, index

LEDGERS = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs")


def _schema_features(generated):
    schema = json.loads((generated["dir"] / "schema.json").read_text(encoding="utf-8"))
    return [c for c, s in schema["columns"].items() if s["role"] == "feature"]


def test_every_feature_has_a_reconstruction_or_validation_rule(generated):
    features = _schema_features(generated)
    missing = [c for c in features if c not in FEATURE_RULES]
    assert missing == [], f"feature columns without a D1/D2 rule: {missing}"
    assert set(FEATURE_RULES) == set(features)
    assert set(D1).isdisjoint(D2)


def test_coverage_check_fails_for_an_uncovered_feature(generated):
    features = _schema_features(generated) + ["new_unreviewed_feature"]
    assert [c for c in features if c not in FEATURE_RULES] == ["new_unreviewed_feature"]


def test_all_features_match_the_independent_reconstruction_exactly(generated):
    idx = index({n: generated[n] for n in LEDGERS})
    snaps_text = pd.read_csv(generated["dir"] / "snapshots.csv", dtype=str, keep_default_na=False)
    mismatches = []
    for r in snaps_text.to_dict("records"):
        f = features_at(idx, r["production_id"], r["stage"], r["as_of_date"])
        for col in FEATURE_RULES:
            if as_csv_text(f[col]) != r[col]:
                mismatches.append((r["production_id"], r["stage"], col, as_csv_text(f[col]), r[col]))
    assert mismatches == [], mismatches[:5]


# ---------- D3: mutation tests ----------

def _features(row: dict) -> dict:
    return {k: row[k] for k in FEATURE_RULES}


def test_generator_ignores_records_dated_after_the_snapshot():
    productions = simulate_all(C.SEED)
    for p in productions[:80]:
        for stage in ("S0", "S1", "S2", "S3"):
            day = stage_days(p)[stage]
            base = _features(snapshot_row(p, stage, day))
            q = copy.deepcopy(p)
            q.commitments.append(Commitment("X-C99", p.production_id, next(iter(p.category_plans_entered)),
                                            "includes_vat", "company", 99_000.0, 83_898.31, day + 1,
                                            final_pre_vat=90_000.0))
            q.manual_costs.append(ManualCost("X-M99", p.production_id, next(iter(p.category_plans_entered)),
                                             "unexpected", "no_vat", 5_000.0, 5_000.0, day + 1))
            for c in q.commitments:                     # alter / cancel future-dated records
                if c.selected_day > day:
                    c.original_pre_vat *= 3
                if c.cancelled_day is None and c.selected_day <= day:
                    c.cancelled_day = day + 2
                    break
            q.manual_costs = [m for m in q.manual_costs if m.created_day <= day or m.manual_cost_id == "X-M99"]
            assert _features(snapshot_row(q, stage, day)) == base, (p.production_id, stage)


def test_generator_reacts_to_records_dated_on_or_before_the_snapshot():
    p = copy.deepcopy(simulate_all(C.SEED)[0])
    day = stage_days(p)["S2"]
    base = _features(snapshot_row(p, "S2", day))
    p.manual_costs.append(ManualCost("X-M98", p.production_id, next(iter(p.category_plans_entered)),
                                     "unexpected", "no_vat", 5_000.0, 5_000.0, day))
    after = _features(snapshot_row(p, "S2", day))
    assert after["committed_pre_vat"] == base["committed_pre_vat"] + 5_000.0
    assert after["n_manual_costs"] == base["n_manual_costs"] + 1


def _mutated(generated, pid, as_of, after: bool):
    """Copy of the source tables with records inserted/altered/removed after (or on/before) as_of."""
    t = {n: generated[n].copy() for n in LEDGERS}
    shift = (pd.Timestamp(as_of) + pd.Timedelta(days=1 if after else 0)).date().isoformat()
    c = t["commitments"]
    new = c[c["production_id"] == pid].iloc[[0]].copy()
    new["commitment_id"], new["selected_date"], new["cancelled_date"] = "X-C97", shift, ""
    t["commitments"] = pd.concat([c, new], ignore_index=True)
    m = t["manual_costs"]
    later = (m["production_id"] == pid) & (m["created_date"] > as_of)
    t["manual_costs"] = m[~later] if after else m
    rev = t["budget_revisions"]
    extra = rev[rev["production_id"] == pid].iloc[[-1]].copy()
    extra["revision_no"] = int(extra["revision_no"].iloc[0]) + 1
    extra["effective_date"] = shift
    extra["approved_entered_amount"] = extra["approved_entered_amount"] * 1.4
    t["budget_revisions"] = pd.concat([rev, extra], ignore_index=True)
    return t


def test_reconstruction_ignores_future_records(generated):
    snaps = generated["snapshots"]
    for r in snaps[snaps["stage"].isin(["S1", "S2"])].head(120).to_dict("records"):
        base = features_at(index({n: generated[n] for n in LEDGERS}), r["production_id"], r["stage"], r["as_of_date"])
        mut = features_at(index(_mutated(generated, r["production_id"], r["as_of_date"], after=True)),
                          r["production_id"], r["stage"], r["as_of_date"])
        assert {k: base[k] for k in FEATURE_RULES} == {k: mut[k] for k in FEATURE_RULES}


def test_reconstruction_reacts_to_records_on_or_before_the_snapshot(generated):
    snaps = generated["snapshots"]
    r = snaps[snaps["stage"] == "S2"].iloc[0].to_dict()
    base = features_at(index({n: generated[n] for n in LEDGERS}), r["production_id"], r["stage"], r["as_of_date"])
    mut = features_at(index(_mutated(generated, r["production_id"], r["as_of_date"], after=False)),
                      r["production_id"], r["stage"], r["as_of_date"])
    assert mut["n_commitments_active"] == base["n_commitments_active"] + 1
    assert mut["budget_pre_vat"] != base["budget_pre_vat"]
    assert mut["budget_revisions_to_date"] == base["budget_revisions_to_date"] + 1
