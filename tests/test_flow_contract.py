"""P5-B: the Flow contract flow-1 and the pinned monetary scale reference. Offline. ALL DATA IS SYNTHETIC.

The historical contract (v2) and its schema are unchanged. A x100 change to a monetary column fails with a
named ``UNIT_SCALE_ANOMALY``, also when a matching contract has been regenerated from the tampered file, and
the anomaly is reported without claiming to identify the underlying unit or currency.
"""

from __future__ import annotations

import csv
import io
import json

import pandas as pd
import pytest

from budget_flow import contract as C
from budget_flow import handoff as FH
from budget_flow import reference as R
from tests.conftest import REPO

APPROVED = REPO / "artifacts" / "crew1" / "p3-live-005"
CLEAN_BYTES = (APPROVED / "clean_data.csv").read_bytes()
BASE_BYTES = (APPROVED / "dataset_contract.json").read_bytes()


def rows_of(data: bytes) -> tuple[list[str], list[dict]]:
    r = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return list(r.fieldnames), list(r)


def csv_bytes(header, rows) -> bytes:
    out = io.StringIO()
    w = csv.DictWriter(out, header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode("utf-8")


def mutate(fn) -> bytes:
    header, rows = rows_of(CLEAN_BYTES)
    header, rows = fn(header, rows) or (header, rows)
    return csv_bytes(header, rows)


def scale(column: str, factor: float) -> bytes:
    def fn(header, rows):
        for r in rows:
            if r[column] != "":
                r[column] = repr(round(float(r[column]) * factor, 2))
    return mutate(fn)


class Run:
    """Writes a delivered handoff (and optionally its flow-1 contract) to a folder and validates it."""

    def __init__(self, tmp_path):
        self.dir = tmp_path

    def __call__(self, clean: bytes = CLEAN_BYTES, base: bytes = BASE_BYTES, flow: dict | None | str = None):
        (self.dir / "clean_data.csv").write_bytes(clean)
        (self.dir / "dataset_contract.json").write_bytes(base)
        flow_path = None
        if flow is not None:
            if flow == "regenerate":
                flow = C.build_flow_contract(clean, base)
            flow_path = self.dir / "flow_contract.json"
            flow_path.write_text(json.dumps(flow), encoding="utf-8")
        return FH.validate_handoff(self.dir / "clean_data.csv", self.dir / "dataset_contract.json", flow_path)


@pytest.fixture
def run(tmp_path):
    return Run(tmp_path)


def by_code(report, code):
    return [v for v in report.violations if v.code == code]


# ---------------------------------------------------------------- unchanged input passes

def test_the_rewriting_helper_is_byte_exact():
    assert mutate(lambda h, r: None) == CLEAN_BYTES


def test_the_approved_handoff_passes_with_and_without_its_flow_contract(run):
    for flow in (None, C.build_flow_contract(CLEAN_BYTES, BASE_BYTES)):
        rep = run(flow=flow)
        assert rep.ok, rep.violations
        assert all(r == 1.0 for r in rep.scale_ratios.values()) and set(rep.scale_ratios) == set(C.MONEY_COLUMNS)


def test_the_historical_contract_and_its_schema_are_unchanged():
    assert C.check_base_contract(json.loads(BASE_BYTES)) == []
    assert json.loads(BASE_BYTES)["version"] == "v2"
    from tests.test_crew2_frozen import PUBLISHED_MAIN, _git
    assert _git("diff", "--name-only", PUBLISHED_MAIN, "--", "contracts", "artifacts").split() == []


# ---------------------------------------------------------------- x100

def test_x100_with_the_original_contracts_fails_with_a_named_scale_violation(run):
    rep = run(clean=scale("budget_pre_vat", 100), flow=C.build_flow_contract(CLEAN_BYTES, BASE_BYTES))
    assert not rep.ok
    anomaly = by_code(rep, "UNIT_SCALE_ANOMALY")
    assert [v.column for v in anomaly] == ["budget_pre_vat"]
    assert rep.scale_ratios["budget_pre_vat"] == pytest.approx(100)
    assert {"CONTENT_DIGEST_MISMATCH", "PROFILE_DECLARATION_MISMATCH"} <= rep.codes   # in addition, not instead


def test_x100_is_named_even_without_any_flow_contract(run):
    rep = run(clean=scale("budget_pre_vat", 100))
    assert "UNIT_SCALE_ANOMALY" in rep.codes and "CONTENT_DIGEST_MISMATCH" not in rep.codes


def test_a_regenerated_matching_contract_does_not_bypass_the_pinned_reference(run):
    tampered = scale("budget_pre_vat", 100)
    rep = run(clean=tampered, flow="regenerate")                # contract rebuilt from the tampered file
    assert rep.codes == {"UNIT_SCALE_ANOMALY"}
    with pytest.raises(C.ContractViolationError, match="UNIT_SCALE_ANOMALY"):
        FH.load_handoff(run.dir / "clean_data.csv", run.dir / "dataset_contract.json", run.dir / "flow_contract.json")


@pytest.mark.parametrize("column", C.MONEY_COLUMNS)
def test_x100_is_detected_on_every_monetary_column(run, column):
    rep = run(clean=scale(column, 100), flow="regenerate")
    assert [v.column for v in by_code(rep, "UNIT_SCALE_ANOMALY")] == [column]


def test_the_anomaly_does_not_claim_to_identify_the_unit(run):
    (v,) = by_code(run(clean=scale("budget_pre_vat", 100)), "UNIT_SCALE_ANOMALY")
    assert "100 x the pinned reference" in v.detail and "not identified" in v.detail
    assert "agorot" in v.detail and "another currency" in v.detail


# ---------------------------------------------------------------- the documented band [0.5, 2.0]

@pytest.mark.parametrize("factor", [0.6, 0.95, 1.05, 1.5, 1.99])
def test_allowed_variation_passes(run, factor):
    rep = run(clean=scale("committed_pre_vat", factor), flow="regenerate")
    assert rep.ok, rep.violations


@pytest.mark.parametrize("factor", [0.01, 0.3, 2.5, 10])
def test_variation_outside_the_band_fails(run, factor):
    rep = run(clean=scale("committed_pre_vat", factor), flow="regenerate")
    assert [v.column for v in by_code(rep, "UNIT_SCALE_ANOMALY")] == ["committed_pre_vat"]


def test_the_band_is_the_documented_initial_policy():
    ref = json.loads(C.REFERENCE_FILE.read_text(encoding="utf-8"))
    assert C.SCALE_BAND == (0.5, 2.0) and ref["band"] == [0.5, 2.0]


# ---------------------------------------------------------------- declarations: units, currency, roles, types

def _base_with(column: str, **changes) -> bytes:
    base = json.loads(BASE_BYTES)
    for c in base["columns"]:
        if c["name"] == column:
            c.update(changes)
    return json.dumps(base, indent=2).encode("utf-8")


def test_a_changed_unit_declaration_fails(run):
    rep = run(base=_base_with("budget_pre_vat", unit="ILS agorot pre-VAT"))
    assert [(v.code, v.column) for v in rep.violations] == [("UNIT_DECLARATION_MISMATCH", "budget_pre_vat")]


def test_a_changed_currency_or_monetary_unit_in_the_flow_contract_fails(run):
    flow = C.build_flow_contract(CLEAN_BYTES, BASE_BYTES)
    flow["currency"]["code"] = "USD"
    flow["monetary_columns"]["fee_contract_pre_vat"]["unit"] = "USD pre-VAT"
    rep = run(flow=flow)
    assert {"CURRENCY_DECLARATION_MISMATCH", "UNIT_DECLARATION_MISMATCH"} == rep.codes


def test_role_restrictions(run):
    rep = run(base=_base_with("final_cost_pre_vat", role="feature"))
    assert [(v.code, v.column) for v in rep.violations] == [("ROLE_DECLARATION_MISMATCH", "final_cost_pre_vat")]
    base = json.loads(BASE_BYTES)
    base["outcome_only_exclusions"].append("budget_pre_vat")      # an outcome-only field declared as a column
    assert "ROLE_RESTRICTION" in run(base=json.dumps(base).encode()).codes


def test_type_and_range_declarations_must_match_the_approved_specification(run):
    assert run(base=_base_with("guest_count", dtype="number")).codes == {"TYPE_DECLARATION_MISMATCH"}
    assert run(base=_base_with("no_vat_share", max=100)).codes == {"RANGE_DECLARATION_MISMATCH"}
    assert run(base=_base_with("stage", allowed=["S0", "S1", "S2", "S3", "S9"])).codes == {"ALLOWED_DECLARATION_MISMATCH"}


def _set(i: int, column: str, value: str) -> bytes:
    def fn(header, rows):
        rows[i][column] = value
    return mutate(fn)


def _first(split: str) -> int:
    return next(i for i, r in enumerate(rows_of(CLEAN_BYTES)[1]) if r["split"] == split)


@pytest.mark.parametrize("column,value,code", [
    ("guest_count", "12.5", "TYPE_VIOLATION"),
    ("as_of_date", "2026-13-40", "TYPE_VIOLATION"),
    ("fee_included", "yes", "TYPE_VIOLATION"),
    ("budget_pre_vat", "", "NULL_VIOLATION"),
    ("stage", "S9", "ALLOWED_VALUE_VIOLATION"),
    ("no_vat_share", "1.5", "RANGE_VIOLATION"),
    ("committed_pre_vat", "-5.0", "RANGE_VIOLATION"),
])
def test_value_violations_are_named(run, column, value, code):
    rep = run(clean=_set(_first("train"), column, value), flow="regenerate")
    assert (code, column) in {(v.code, v.column) for v in rep.violations}


def test_a_missing_train_label_is_a_violation_but_a_missing_test_label_is_not(run):
    assert ("NULL_VIOLATION", "is_overrun") in {(v.code, v.column) for v in
                                                run(clean=_set(_first("train"), "is_overrun", "")).violations}
    assert run(clean=_set(_first("test"), "is_overrun", "")).ok      # test labels are redacted on reading anyway


def test_structure_violations_are_named(run):
    dropped = mutate(lambda h, rows: ([c for c in h if c != "guest_count"],
                                      [{k: v for k, v in r.items() if k != "guest_count"} for r in rows]))
    assert "COLUMN_SET_MISMATCH" in run(clean=dropped).codes
    dup = mutate(lambda h, rows: (h, rows + [dict(rows[0])]))
    assert {"PRIMARY_KEY_DUPLICATE", "ROW_COUNT_MISMATCH"} <= run(clean=dup).codes


def test_a_production_moved_out_of_its_frozen_group_fails(run):
    i = _first("train")
    pid = rows_of(CLEAN_BYTES)[1][i]["production_id"]

    def fn(header, rows):
        for r in rows:
            if r["production_id"] == pid:
                r["split"], r["cv_fold"] = "test", "-1"
    assert "SPLIT_POLICY_VIOLATION" in run(clean=mutate(fn), flow="regenerate").codes


def test_a_changed_delivered_contract_is_detected_by_the_flow_contract(run):
    flow = C.build_flow_contract(CLEAN_BYTES, BASE_BYTES)
    base = json.loads(BASE_BYTES)
    base["row_count"] = 1537
    rep = run(base=json.dumps(base).encode(), flow=flow)
    assert {"ROW_COUNT_MISMATCH", "BASE_CONTRACT_DIGEST_MISMATCH"} <= rep.codes


def test_a_malformed_flow_contract_fails_its_schema(run):
    flow = C.build_flow_contract(CLEAN_BYTES, BASE_BYTES)
    del flow["monetary_columns"]["budget_gap_ils"]
    assert run(flow=flow).codes == {"FLOW_CONTRACT_SCHEMA"}


# ---------------------------------------------------------------- the pinned reference

def test_the_reference_is_independently_reproduced_from_the_approved_handoff():
    """pandas/numpy (not the module's pure-Python statistics) on the TRAIN rows of the approved handoff."""
    from crew2.ml.data import EXPECTED_CLEAN_SHA256
    assert C.sha256_hex(CLEAN_BYTES) == EXPECTED_CLEAN_SHA256
    ref = json.loads(C.REFERENCE_FILE.read_text(encoding="utf-8"))
    train = pd.read_csv(io.BytesIO(CLEAN_BYTES)).query("split == 'train'")
    assert ref["source"]["train_rows"] == len(train) == 1084
    for c in C.MONEY_COLUMNS:
        a = train[c].abs()
        median, p90 = float(a.median()), float(a.quantile(0.9, interpolation="higher"))
        want_stat = "median_abs" if median > 0 else "p90_abs"
        col = ref["columns"][c]
        assert col["unit"] == C.MONEY_UNIT and col["statistic"] == want_stat
        assert col["median_abs"] == pytest.approx(median, rel=1e-12) and col["p90_abs"] == pytest.approx(p90, rel=1e-12)
        assert col["value"] == col[want_stat] > 0


def test_the_reference_matches_a_fresh_build_and_is_pinned():
    assert R.main(["--check"]) == 0
    assert C.sha256_hex(C.REFERENCE_FILE.read_bytes()) == C.REFERENCE_SHA256


def test_a_changed_reference_file_fails(tmp_path):
    ref = json.loads(C.REFERENCE_FILE.read_text(encoding="utf-8"))
    ref["columns"]["budget_pre_vat"]["value"] *= 100              # an attempt to accept x100 data
    fake = tmp_path / "money_scale_reference.v1.json"
    fake.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    header, rows = C.parse_and_redact(scale("budget_pre_vat", 100))
    rep = C.validate(header, rows, "0" * 64, BASE_BYTES, reference_path=fake)
    assert "REFERENCE_DIGEST_MISMATCH" in rep.codes and not rep.ok


def test_the_reference_holds_no_test_or_demo_outcome():
    """It is built from TRAIN rows only: removing every non-train row leaves it unchanged."""
    header, rows = rows_of(CLEAN_BYTES)
    train_only = [r for r in rows if r["split"] == "train"]
    _, parsed = C.parse_and_redact(csv_bytes(header, train_only))
    _, full = C.parse_and_redact(CLEAN_BYTES)
    assert C.scale_profile(parsed) == C.scale_profile(full)
