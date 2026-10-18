"""Snapshot totals reconcile with the underlying synthetic ledgers."""

import pytest

from generator.vat import pre_vat

RATE = 0.18


def _is_true(series):
    return series.astype(str) == "True"


def test_stored_pre_vat_matches_typed_amounts(generated):
    c = generated["commitments"]
    recomputed = [pre_vat(a, m, RATE) for a, m in zip(c["original_entered"], c["vat_mode"])]
    assert (c["original_pre_vat"] - recomputed).abs().max() <= 0.01
    m = generated["manual_costs"]
    recomputed = [pre_vat(a, v, RATE) for a, v in zip(m["entered"], m["vat_mode"])]
    assert (m["pre_vat"] - recomputed).abs().max() <= 0.01


def test_committed_reconciles_with_ledgers(generated, snaps):
    c, m = generated["commitments"], generated["manual_costs"]
    sample = snaps.sample(200, random_state=0)
    for _, row in sample.iterrows():
        pc = c[c["production_id"] == row["production_id"]]
        active = pc[(pc["selected_date"] <= row["as_of_date"]) &
                    ((pc["cancelled_date"] == "") | (pc["cancelled_date"] > row["as_of_date"]))]
        pm = m[(m["production_id"] == row["production_id"]) & (m["created_date"] <= row["as_of_date"])]
        expected = active["original_pre_vat"].sum() + pm["pre_vat"].sum()
        assert row["committed_pre_vat"] == pytest.approx(expected, abs=0.05)
        assert row["n_commitments_active"] == len(active)
        assert row["n_manual_costs"] == len(pm)


def test_final_cost_reconciles_with_ledgers(generated):
    c, m, p = generated["commitments"], generated["manual_costs"], generated["productions"]
    active_final = c[c["cancelled_date"] == ""].groupby("production_id")["final_pre_vat"].sum()
    manual = m.groupby("production_id")["pre_vat"].sum()
    expected = active_final.add(manual, fill_value=0.0)
    got = p.set_index("production_id")["final_cost_pre_vat"]
    assert (got - expected.reindex(got.index)).abs().max() <= 0.05


def test_cancelled_commitments_are_replaced_and_cost_nothing_directly(generated):
    c = generated["commitments"]
    cancelled = c[c["cancelled_date"] != ""]
    assert len(cancelled) > 0
    assert (cancelled["final_pre_vat"] == 0).all()
    replaced = set(c.loc[c["replaces_commitment_id"] != "", "replaces_commitment_id"])
    assert set(cancelled["commitment_id"]) == replaced


def test_payments_follow_obligations(generated):
    c = generated["commitments"].set_index("commitment_id")
    pay = generated["payments"]
    prods = generated["productions"].set_index("production_id")
    assert (pay["direction"] == "out").all()
    paid_for = c.loc[pay["commitment_id"]]
    # only company-paid, non-cancelled commitments are paid by the company
    assert (paid_for["payment_responsibility"] == "company").all()
    assert (paid_for["cancelled_date"] == "").all()
    # a payment never precedes the selection, nor follows financial close
    assert (pay["paid_date"].values >= paid_for["selected_date"].values).all()
    assert (pay["paid_date"].values <= prods.loc[pay["production_id"], "financial_close_date"].values).all()
    # total paid per commitment never exceeds its final gross cost
    totals = pay.groupby("commitment_id")["amount_gross"].sum()
    cc = c.loc[totals.index]
    final_gross = cc["final_pre_vat"].where(cc["vat_mode"] == "no_vat", cc["final_pre_vat"] * (1 + RATE))
    assert (totals - final_gross).max() <= 0.05


def test_client_direct_commitments_have_no_company_payments(generated):
    c = generated["commitments"]
    direct = set(c.loc[c["payment_responsibility"] == "client_direct", "commitment_id"])
    assert len(direct) > 0
    assert direct.isdisjoint(set(generated["payments"]["commitment_id"]))


def test_payments_are_not_costs(generated, snaps):
    """Cash flow never enters the cost: no payment-derived column exists in the snapshots
    (the final cost itself is reconciled against commitments + manual costs only, above)."""
    assert not any("paid" in col or "payment" in col for col in snaps.columns)


def test_position_feature_matches_definition(snaps):
    fee = snaps["fee_contract_pre_vat"].where(_is_true(snaps["fee_included"]), 0.0)
    pos = snaps["committed_pre_vat"] + fee - snaps["budget_pre_vat"]
    assert (pos - snaps["current_position_ils"]).abs().max() <= 0.02
