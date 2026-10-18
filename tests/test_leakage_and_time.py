"""Prediction-time features only, and chronological consistency of snapshots."""

import json

import pytest

from generator import config as C
from generator.schema import FORBIDDEN_FEATURE_SUBSTRINGS, SNAPSHOT_COLUMNS, feature_columns


def _features_from(schema_cols):
    return [c for c, s in schema_cols.items() if s["role"] == "feature"]


def check_no_leakage(schema_cols):
    """Raise AssertionError if any target / future / forbidden column is marked as a feature."""
    for col, spec in schema_cols.items():
        if spec["role"] != "feature":
            continue
        assert spec["available_at"] == "as_of", f"{col} is a feature but known only at {spec['available_at']}"
        for bad in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert bad not in col, f"{col} looks target-derived but is marked as a feature"


def test_schema_has_no_leakage():
    check_no_leakage(SNAPSHOT_COLUMNS)


def test_leakage_check_catches_a_target_marked_as_feature():
    """The leakage gate must fail when it should (proven in both directions)."""
    broken = json.loads(json.dumps(SNAPSHOT_COLUMNS))
    broken["budget_gap_ils"]["role"] = "feature"
    with pytest.raises(AssertionError):
        check_no_leakage(broken)
    broken = json.loads(json.dumps(SNAPSHOT_COLUMNS))
    broken["final_cost_pre_vat"].update(role="feature", available_at="as_of")
    with pytest.raises(AssertionError):
        check_no_leakage(broken)


def test_every_target_is_known_only_at_close():
    for col, spec in SNAPSHOT_COLUMNS.items():
        if spec["role"] == "target":
            assert spec["available_at"] == "financial_close", col


def test_schema_covers_exactly_the_snapshot_columns(snaps, generated):
    assert list(snaps.columns) == list(SNAPSHOT_COLUMNS)
    written = json.loads((generated["dir"] / "schema.json").read_text(encoding="utf-8"))
    assert set(written["columns"]) == set(SNAPSHOT_COLUMNS)
    assert written["synthetic"] is True


def test_feature_list(snaps):
    feats = feature_columns()
    assert feats == _features_from(SNAPSHOT_COLUMNS)
    assert "is_overrun" not in feats and "budget_gap_ils" not in feats and "split" not in feats
    assert set(feats) <= set(snaps.columns)


def test_stage_dates_are_strictly_ordered(snaps):
    order = {s: i for i, s in enumerate(C.STAGES)}
    s = snaps.assign(o=snaps["stage"].map(order)).sort_values(["production_id", "o"])
    for _, g in s.groupby("production_id"):
        dates = g["as_of_date"].tolist()
        assert dates == sorted(dates) and len(set(dates)) == 4
        assert g["days_to_event"].is_monotonic_decreasing
        assert (g["as_of_date"] < g["event_date"]).all()
        assert g["n_commitments_active"].is_monotonic_increasing
        assert g["n_manual_costs"].is_monotonic_increasing
        assert g["budget_revisions_to_date"].is_monotonic_increasing


def test_snapshots_only_see_records_created_by_as_of(generated, snaps):
    """No commitment selected, and no manual cost created, after as_of is counted."""
    c, m = generated["commitments"], generated["manual_costs"]
    s0 = snaps[snaps["stage"] == "S0"].set_index("production_id")
    # at contract day only commitments selected on day 0 can be active
    first = c.groupby("production_id")["selected_date"].min()
    has_day0 = (first.reindex(s0.index) <= s0["as_of_date"])
    assert ((s0["n_commitments_active"] > 0) <= has_day0).all()
    late = m.merge(snaps[snaps["stage"] == "S3"][["production_id", "as_of_date"]], on="production_id")
    late = late[late["created_date"] > late["as_of_date"]]
    assert len(late) > 0, "some costs must arrive after the last snapshot (future information)"


def test_close_after_event(generated):
    p = generated["productions"]
    assert (p["financial_close_date"] > p["event_date"]).all()
    assert (p["event_date"] > p["contract_date"]).all()
