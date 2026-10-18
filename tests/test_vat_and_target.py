"""VAT semantics, target definition and the separation of overrun from profit."""

import pytest

from generator import config as C
from generator.vat import fee_entered, gross_of, pre_vat

RATE = 0.18


def test_locked_reference_example():
    # 300,000 typed including VAT; fixed fee 30,000 including VAT, included in the budget.
    assert pre_vat(300_000, "includes_vat", RATE) == pytest.approx(254_237.288, abs=0.01)
    assert round(pre_vat(30_000, "includes_vat", RATE)) == 25_424
    assert round(pre_vat(30_000, "includes_vat", RATE) / pre_vat(300_000, "includes_vat", RATE) * 100) == 10


@pytest.mark.parametrize("mode,pre,gross", [
    ("includes_vat", 1000 / 1.18, 1000),
    ("excludes_vat", 1000, 1180),
    ("no_vat", 1000, 1000),
])
def test_vat_modes(mode, pre, gross):
    assert pre_vat(1000, mode, RATE) == pytest.approx(pre)
    assert gross_of(1000, mode, RATE) == pytest.approx(gross)


def test_unknown_vat_mode_is_rejected():
    with pytest.raises(ValueError):
        pre_vat(1, "gross", RATE)


def test_fee_modes():
    assert fee_entered(200_000, "percent", 10) == pytest.approx(20_000)
    assert fee_entered(200_000, "amount", 15_000) == 15_000


def test_gap_formula(snaps):
    fee = snaps["fee_contract_pre_vat"].where(snaps["fee_included"].astype(str) == "True", 0.0)
    gap = snaps["final_cost_pre_vat"] + fee - snaps["budget_pre_vat"]
    assert (gap - snaps["budget_gap_ils"]).abs().max() <= 0.02


def test_any_positive_excess_is_an_overrun(snaps):
    expected = snaps["budget_gap_ils"] > C.OVERRUN_TOLERANCE_ILS
    assert (snaps["is_overrun"].astype(str) == "True").equals(expected)


def test_materiality_is_a_separate_warning_level(snaps):
    material = snaps["is_material_overrun"].astype(str) == "True"
    overrun = snaps["is_overrun"].astype(str) == "True"
    assert not (material & ~overrun).any()          # material implies overrun
    assert (overrun & ~material).any()              # small overruns still count as overruns


def test_final_cost_is_constant_per_production(snaps):
    assert snaps.groupby("production_id")["final_cost_pre_vat"].nunique().eq(1).all()


def test_fee_concession_never_changes_the_label(generated, snaps):
    prods = generated["productions"].set_index("production_id")
    conceded = prods.index[prods["fee_concession_pre_vat"] > 0]
    assert len(conceded) > 0
    s = snaps[snaps["production_id"].isin(conceded)]
    fee = s["fee_contract_pre_vat"].where(s["fee_included"].astype(str) == "True", 0.0)
    assert ((s["final_cost_pre_vat"] + fee - s["budget_pre_vat"]) - s["budget_gap_ils"]).abs().max() <= 0.02
    # profit is reduced by the concession; the overrun label is not
    p = prods.loc[conceded]
    diff = p["final_fee_contract_pre_vat"] - p["fee_concession_pre_vat"] - p["final_company_profit_pre_vat"]
    assert diff.abs().max() <= 0.02


def test_overrun_rate_is_in_the_designed_band(snaps):
    rate = (snaps["is_overrun"].astype(str) == "True").mean()
    assert 0.30 <= rate <= 0.50
    for stage, r in (snaps.assign(o=snaps["is_overrun"].astype(str) == "True")
                     .groupby("stage")["o"].mean().items()):
        assert 0.20 <= r <= 0.65, stage
