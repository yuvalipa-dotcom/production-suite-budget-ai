"""v1 -> v2 conformance (A3, A4) and independent label recomputation (A5)."""

import copy
import json

import pandas as pd

from generator import config as C
from generator.simulate import BudgetRevision, simulate_all
from generator.snapshots import snapshot_row, stage_days
from tests.conftest import DATA
from tests.reconstruct import LABELS, as_csv_text, features_at, index

LEDGERS = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs")


def _v1(name):
    return pd.read_csv(DATA / "v1" / f"{name}.csv", keep_default_na=False)


def test_ids_groups_and_folds_identical_across_versions(generated):
    for name in ("productions", "snapshots"):
        v1, v2 = _v1(name), generated[name]
        cols = ["production_id", "split", "cv_fold"] + (["stage"] if name == "snapshots" else [])
        assert v1[cols].equals(v2[cols]), name


def test_v2_keeps_every_v1_file_byte_identical(generated):
    for f in ("snapshots.csv", "productions.csv", "commitments.csv", "manual_costs.csv", "payments.csv",
              "schema.json"):
        assert (DATA / "v1" / f).read_bytes() == (generated["dir"] / f).read_bytes(), f


def test_v2_manifest_declares_version_and_reason(generated):
    m = json.loads((generated["dir"] / "manifest.json").read_text(encoding="utf-8"))
    assert m["dataset_version"] == "v2" and m["supersedes"] == "v1"
    assert "revision ledger" in m["reason"] and "unchanged" in m["reason"]


def test_revision_ledger_preserves_original_and_every_revision(generated):
    rev, prods = generated["budget_revisions"], generated["productions"].set_index("production_id")
    by = rev.groupby("production_id")
    assert by.size().eq(prods["n_budget_revisions"] + 1).all()
    first = rev[rev["revision_no"] == 0].set_index("production_id")
    assert (first["effective_date"] == prods.loc[first.index, "contract_date"]).all()
    assert (first["approved_entered_amount"] == prods.loc[first.index, "initial_budget_entered"]).all()
    last = rev.loc[by["revision_no"].idxmax()].set_index("production_id")
    assert (last["approved_entered_amount"] == prods.loc[last.index, "final_budget_entered"]).all()
    for _, g in by:
        assert g["revision_no"].tolist() == list(range(len(g)))
        assert g["effective_date"].is_monotonic_increasing
    pre = rev["approved_entered_amount"].where(rev["vat_mode"] != "includes_vat",
                                               rev["approved_entered_amount"] / 1.18)
    assert (pre - rev["approved_pre_vat_amount"]).abs().max() <= 0.005


def test_every_label_and_gap_recomputed_independently(generated):
    """A5: labels and signed gaps rebuilt from final costs, fee terms and the revision ledger."""
    idx = index({n: generated[n] for n in LEDGERS})
    snaps_text = pd.read_csv(generated["dir"] / "snapshots.csv", dtype=str, keep_default_na=False)
    mismatches = []
    for r in snaps_text.to_dict("records"):
        f = features_at(idx, r["production_id"], r["stage"], r["as_of_date"])
        for col in LABELS:
            if as_csv_text(f[col]) != r[col]:
                mismatches.append((r["production_id"], r["stage"], col))
    assert mismatches == []


def test_label_changes_only_through_effective_budget_revisions(snaps, generated):
    changed = snaps.groupby("production_id")["is_overrun"].nunique()
    changed = set(changed[changed > 1].index)
    revised = set(generated["productions"].loc[generated["productions"]["n_budget_revisions"] > 0, "production_id"])
    assert changed and changed <= revised


def test_budget_revision_after_a_snapshot_changes_nothing_in_it():
    productions = simulate_all(C.SEED)
    checked = 0
    for p in productions[:60]:
        days = stage_days(p)
        for stage in ("S0", "S1", "S2"):
            day = days[stage]
            before = snapshot_row(p, stage, day)
            q = copy.deepcopy(p)
            q.revisions.append(BudgetRevision(day=day + 1, budget_entered=q.budget_entered_at(day) * 1.5,
                                              guest_count=q.guest_count_at(day) + 100))
            q.revisions.sort(key=lambda r: r.day)
            after = snapshot_row(q, stage, day)
            assert before == after, (p.production_id, stage)
            checked += 1
    assert checked == 180


def test_budget_revision_before_a_snapshot_does_change_it():
    p = copy.deepcopy(simulate_all(C.SEED)[0])
    day = stage_days(p)["S2"]
    before = snapshot_row(p, "S2", day)
    p.revisions.append(BudgetRevision(day=day, budget_entered=p.budget_entered_at(day) * 1.5,
                                      guest_count=p.guest_count_at(day)))
    p.revisions.sort(key=lambda r: r.day)
    after = snapshot_row(p, "S2", day)
    assert after["budget_pre_vat"] != before["budget_pre_vat"]
    assert after["budget_gap_ils"] != before["budget_gap_ils"]
