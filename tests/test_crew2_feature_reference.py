"""C4: the independent feature reference, and D4: fields without a revision ledger.

All 1,538 eligible snapshot feature rows are reconstructed from the CLEANED Policy C ledgers with
the as-of rules, and compared with the independently frozen v2 snapshot table
(``data/synthetic/v2/snapshots.csv``, written by the generator, never by the cleaning or
reconstruction code). If the reference is missing or disagrees, these tests fail: the frozen
data are never modified to force agreement. Offline. ALL DATA IS SYNTHETIC.

Numeric tolerance, stated explicitly: the reference is stored with 2 decimals, so
* money amounts (ILS pre-VAT) must agree within 0.01 ILS;
* ratios and shares must agree within 0.0051 (half of the last stored digit, plus float margin);
* integers, codes, booleans and the key must agree exactly.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from crew2.ml import data as D
from crew2.ml.features import F29
from gate import policy_c as P
from gate.checks import LEDGERS
from gate.handoff_spec import COLUMNS
from tests.conftest import ACTIVE_DATA
from tests.crew2_helpers import no_network  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")

REFERENCE = ACTIVE_DATA / "snapshots.csv"
MONEY_TOL = 0.01
RATIO_TOL = 0.0051
STATIC_FIELDS = ("archetype", "event_month", "lead_time_days", "budget_vat_mode", "fee_included", "fee_mode")


@pytest.fixture(scope="module")
def reconstruction():
    """Features rebuilt from the cleaned ledgers (as-of rules), keyed by (production_id, stage)."""
    raw, _ = D.load_raw()
    result = P.apply_policy_c(raw)
    ix = P.index({n: result.tables[n] for n in LEDGERS})
    elig = result.eligibility[result.eligibility["eligible"]]
    rows = [{"production_id": r.production_id, "stage": r.stage, **P.snapshot_features(ix, r.production_id, r.stage,
                                                                                        r.as_of_date)}
            for r in elig.itertuples()]
    return pd.DataFrame(rows).sort_values(["production_id", "stage"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def reference(reconstruction):
    if not REFERENCE.is_file():
        pytest.fail("STOP: the independent frozen snapshot reference is unavailable")
    ref = pd.read_csv(REFERENCE, dtype=str, keep_default_na=False)
    keys = set(zip(reconstruction["production_id"], reconstruction["stage"]))
    ref = ref[[k in keys for k in zip(ref["production_id"], ref["stage"])]]
    return ref.sort_values(["production_id", "stage"]).reset_index(drop=True)


def _tolerance(column: str) -> float:
    return MONEY_TOL if COLUMNS[column][2] == "ILS pre-VAT" else RATIO_TOL


def test_row_identity(reconstruction, reference):
    assert len(reconstruction) == 1538
    assert list(zip(reconstruction["production_id"], reconstruction["stage"])) == \
        list(zip(reference["production_id"], reference["stage"]))


def test_values_match_the_independent_reference_within_the_stated_tolerance(reconstruction, reference):
    worst = {}
    for c in F29:
        dtype = COLUMNS[c][0]
        if dtype == "number":
            diff = (reconstruction[c].astype(float) - reference[c].astype(float)).abs()
            worst[c] = float(diff.max())
            assert worst[c] <= _tolerance(c), (c, worst[c])
        elif dtype == "integer":
            assert (reconstruction[c].astype(int) == reference[c].astype(int)).all(), c
        elif dtype == "boolean":
            assert (reconstruction[c].map(str) == reference[c]).all(), c
        else:
            assert (reconstruction[c] == reference[c]).all(), c


def test_dtypes_and_missingness(reconstruction, reference):
    for c in F29:
        assert (reference[c] != "").all(), f"reference has a missing {c}"
        assert reconstruction[c].notna().all(), f"reconstruction has a missing {c}"
        dtype = COLUMNS[c][0]
        if dtype == "integer":
            assert reconstruction[c].map(lambda v: isinstance(v, int) and not isinstance(v, bool)).all(), c
            pd.to_numeric(reference[c], downcast="integer")
        elif dtype == "number":
            assert reconstruction[c].map(lambda v: isinstance(v, float)).all(), c
        elif dtype == "boolean":
            assert reconstruction[c].map(lambda v: isinstance(v, bool)).all(), c
            assert set(reference[c]) <= {"True", "False"}, c
        else:
            assert reconstruction[c].map(lambda v: isinstance(v, str)).all(), c


def test_as_of_budget_matches_the_revision_in_effect_and_differs_from_the_final_state(reference):
    """As-of behaviour against the frozen reference: where a later revision exists, the snapshot keeps the
    budget in effect at its date, not the final approved budget."""
    raw, _ = D.load_raw()
    result = P.apply_policy_c(raw)
    revs = result.tables["budget_revisions"]
    prods = result.tables["productions"].set_index("production_id")
    checked = later = 0
    for r in reference.itertuples():
        pr = revs[revs["production_id"] == r.production_id]
        in_effect = pr[pr["effective_date"] <= r.as_of_date].sort_values("revision_no", key=lambda s: s.astype(int))
        entered = float(in_effect.iloc[-1]["approved_entered_amount"])
        pre = entered / 1.18 if prods.loc[r.production_id, "budget_vat_mode"] == "includes_vat" else entered
        assert abs(pre - float(r.budget_pre_vat)) <= MONEY_TOL, (r.production_id, r.stage)
        if (pr["effective_date"] > r.as_of_date).any():
            later += 1
        checked += 1
    assert checked == 1538 and later > 0


def test_the_handoff_feature_columns_equal_the_reference(reference):
    text, _ = D.build_handoff()
    got = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False).sort_values(["production_id", "stage"])
    got = got.reset_index(drop=True)
    for c in F29:
        if COLUMNS[c][0] == "number":
            assert (got[c].astype(float) - reference[c].astype(float)).abs().max() <= _tolerance(c) + 1e-9, c
        else:
            assert (got[c] == reference[c]).all(), c


# ---------------------------------------------------------------- D4

def test_fields_without_a_revision_ledger_are_constant_across_a_productions_snapshots(reference):
    full = pd.read_csv(REFERENCE, dtype=str, keep_default_na=False)
    for frame in (full, reference):
        per = frame.groupby("production_id")[list(STATIC_FIELDS)].nunique()
        assert (per == 1).all().all(), per[(per != 1).any(axis=1)].head()
    raw, _ = D.load_raw()
    prods = P.apply_policy_c(raw).tables["productions"]
    assert prods["production_id"].is_unique                     # one master record each: no history to take
    for c in ("archetype", "contract_date", "event_date", "budget_vat_mode", "fee_mode", "fee_value",
              "fee_vat_mode", "fee_included"):
        assert c in prods.columns
