"""P2 handoff quality gate: reference handoff, 19 approved corruption cases, determinism,
read boundary and Policy C equivalence. ALL DATA IS SYNTHETIC.

Corruptions are applied to copies of an independent reference handoff. Where it
applies, the copied contract is edited too, so it stays internally consistent and
the gate must catch the error through its own reconstruction from raw.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from gate import checks as G
from gate.run import main
from generator import config as C
from tests.conftest import ACTIVE_DATA, REPO
from tests.handoff_reference import reference_frames, write_handoff
from tests.raw_repair import apply_policy_c as oracle_policy_c
from tests.reconstruct import FEATURE_RULES, LABELS, as_csv_text, index, snapshot_features, snapshot_labels

RAW = ACTIVE_DATA / "raw"
HARD_IDS = [c[0] for c in G.HANDOFF_CHECKS if c[1] == G.HARD]


@pytest.fixture(scope="module")
def reference(expected_ineligible):
    return reference_frames(expected_ineligible)


def _gate(tmp: Path, data: pd.DataFrame | None, contract: dict, drop_data: bool = False):
    data_path, contract_path = write_handoff(data, contract, tmp / "handoff")
    if drop_data:
        data_path.unlink()
    out = tmp / "report"
    code = main(["--stage", "handoff", "--raw", str(RAW), "--out", str(out),
                 "--data", str(data_path), "--contract", str(contract_path)])
    res = json.loads((out / "gate_result.json").read_text(encoding="utf-8"))
    return code, res, out


def _failing(res) -> list[str]:
    return [c["id"] for c in res["checks"] if c["severity"] == G.HARD and c["status"] != G.PASS]


# ---------------------------------------------------------------- reference handoff

def test_reference_handoff_passes_every_hard_check(tmp_path, reference):
    data, contract = reference
    code, res, _ = _gate(tmp_path, data, contract)
    assert code == 0 and res["status"] == "PASS" and _failing(res) == []
    assert [c["id"] for c in res["checks"]] == [c[0] for c in G.HANDOFF_CHECKS]
    assert len(data) == 1538 and contract["exclusion_counts"] == {"commitment_category_missing": 50,
                                                                  "manual_cost_undated": 12}


def test_source_gate_passes(tmp_path):
    assert main(["--stage", "source", "--raw", str(RAW), "--out", str(tmp_path)]) == 0
    res = json.loads((tmp_path / "gate_result.json").read_text(encoding="utf-8"))
    assert [(c["id"], c["status"]) for c in res["checks"]] == [
        ("G-SRC-MANIFEST", "PASS"), ("G-SRC-ROLES", "PASS"), ("G-SRC-OUTCOME", "PASS"),
        ("G-SRC-FROZEN-SPLIT", "PASS"), ("I-SYNTHETIC", "PASS")]


def test_prevalence_warning_is_computed_and_labelled(tmp_path, reference):
    _, res, _ = _gate(tmp_path, *reference)
    w = next(c for c in res["checks"] if c["id"] == "W-PREVALENCE-IMBALANCE")
    assert w["status"] == "WARN" and w["severity"] == "WARN"
    assert "before Policy C eligibility filtering (1600 snapshots): train 44.0% (493/1120), test 59.1% (189/320)" \
        in w["details"]
    assert "after Policy C eligibility filtering (1538 snapshots): train 44.5% (482/1084), test 59.2% (181/306)" \
        in w["details"]
    assert "historical context, not recomputed" in w["details"]


def test_handoff_gate_is_deterministic(tmp_path, reference):
    _, _, a = _gate(tmp_path / "a", *reference)
    _, _, b = _gate(tmp_path / "b", *reference)
    for name in ("gate_result.json", "gate_report.md"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
    assert b"\r\n" not in (a / "gate_result.json").read_bytes()


# ---------------------------------------------------------------- the 19 corruption cases

def _num(series: pd.Series, f) -> pd.Series:
    return series.map(lambda v: f"{f(float(v)):.2f}")


def _gap(d):
    return d["budget_gap_ils"].astype(float)


def _final_and_original_budgets() -> tuple[dict, dict]:
    revs = pd.read_csv(ACTIVE_DATA / "budget_revisions.csv", dtype=str, keep_default_na=False)
    revs["n"] = revs["revision_no"].astype(int)
    last = revs.sort_values("n").groupby("production_id").tail(1).set_index("production_id")["approved_pre_vat_amount"]
    first = revs[revs["n"] == 0].set_index("production_id")["approved_pre_vat_amount"]
    total = revs.groupby("production_id")["n"].max()
    return {p: (float(last[p]), int(total[p])) for p in last.index}, {p: float(v) for p, v in first.items()}


def c01(d, c, ctx):
    d["committed_pre_vat"] = _num(d["committed_pre_vat"], lambda v: v * 1000)


def c02(d, c, ctx):
    for col in ("budget_pre_vat", "committed_pre_vat"):
        d[col] = _num(d[col], lambda v: v * 1000)


def c03(d, c, ctx):
    m = d["budget_vat_mode"] == "includes_vat"
    d.loc[m, "budget_pre_vat"] = _num(d.loc[m, "budget_pre_vat"], lambda v: v * (1 + C.VAT_RATE))


def c04(d, c, ctx):
    m = (_gap(d) > 0.5) & (d["is_material_overrun"] == "True")
    d.loc[m, "budget_gap_ils"] = _num(d.loc[m, "budget_gap_ils"], lambda v: v + 1000)


def c05(d, c, ctx):
    i = d.index[(_gap(d) > 1000) & (d["is_material_overrun"] == "True")][0]
    d.loc[i, ["budget_gap_ils", "is_overrun", "is_material_overrun"]] = [f"{-float(d.loc[i, 'budget_gap_ils']):.2f}",
                                                                          "False", "False"]


def c06(d, c, ctx):
    i = d.index[_gap(d).abs() > 1000][0]
    d.loc[i, "is_overrun"] = "False" if d.loc[i, "is_overrun"] == "True" else "True"


def c07(d, c, ctx):
    final, _ = _final_and_original_budgets()
    later = d["budget_revisions_to_date"].astype(int) < d["production_id"].map(lambda p: final[p][1])
    d.loc[later, "budget_pre_vat"] = d.loc[later, "production_id"].map(lambda p: f"{final[p][0]:.2f}")
    assert later.any()


def c08(d, c, ctx):
    _, original = _final_and_original_budgets()
    revised = d["budget_revisions_to_date"].astype(int) > 0
    d.loc[revised, "budget_pre_vat"] = d.loc[revised, "production_id"].map(lambda p: f"{original[p]:.2f}")
    assert revised.any()


def c09(d, c, ctx):
    key, reason = next((k, r) for k, r in sorted(ctx["ineligible"].items()) if "+" not in r)
    snaps = pd.read_csv(ACTIVE_DATA / "snapshots.csv", dtype=str, keep_default_na=False)
    row = snaps[(snaps["production_id"] == key[0]) & (snaps["stage"] == key[1])][list(d.columns)]
    d.loc[len(d)] = row.iloc[0].tolist()
    c["row_count"] += 1
    c["exclusion_counts"][reason] -= 1


def c10(d, c, ctx):
    d.drop(index=d.index[0], inplace=True)
    c["row_count"] -= 1
    c["exclusion_counts"]["commitment_category_missing"] += 1


def c11(d, c, ctx):
    pid = d.loc[d["split"] == "test", "production_id"].iloc[0]
    d.loc[d["production_id"] == pid, ["split", "cv_fold"]] = ["train", "0"]


def c12(d, c, ctx):
    pid = d.loc[d["split"] == "train", "production_id"].iloc[0]
    m = d["production_id"] == pid
    d.loc[m, "cv_fold"] = d.loc[m, "cv_fold"].map(lambda f: str((int(f) + 1) % 5))


def c13(d, c, ctx):
    d["final_pre_vat"] = d["final_cost_pre_vat"]
    c["columns"].append({"name": "final_pre_vat", "dtype": "number", "role": "feature", "unit": "ILS pre-VAT",
                         "nullable": False})
    c["outcome_only_exclusions"].remove("commitments.final_pre_vat")


def c14(d, c, ctx):
    d.rename(columns={"committed_pre_vat": "committed_amount"}, inplace=True)
    next(x for x in c["columns"] if x["name"] == "committed_pre_vat")["name"] = "committed_amount"


def c15(d, c, ctx):
    c["columns"] = [x for x in c["columns"] if x["name"] != "guest_count"]


def c16(d, c, ctx):
    c["raw_manifest_sha256"] = "0" * 64


def c17(d, c, ctx):
    d.loc[len(d)] = d.iloc[0].tolist()
    c["row_count"] += 1


def c18(d, c, ctx):
    d.loc[d.index[0], "committed_pre_vat"] = ""
    next(x for x in c["columns"] if x["name"] == "committed_pre_vat")["nullable"] = True


def c19(d, c, ctx):
    ctx["drop_data"] = True


# (case number, description, corruption, required failing check IDs, "all" or "any")
CASES = [
    (1, "one financial amount column x1000", c01, ["G-FIN-RECON"], "all"),
    (2, "budget and committed x1000, ratios preserved", c02, ["G-FIN-RECON"], "all"),
    (3, "pre-VAT amount replaced by its VAT-inclusive value", c03, ["G-FIN-RECON"], "all"),
    (4, "signed gaps shifted, labels kept consistent", c04, ["G-TARGET-GAP"], "all"),
    (5, "label and gap sign flipped together", c05, ["G-TARGET-LABEL", "G-TARGET-GAP"], "all"),
    (6, "label flipped alone", c06, ["G-TARGET-CONSISTENCY"], "all"),
    (7, "as-of budget replaced by the final approved budget", c07, ["G-FIN-ASOF-BUDGET"], "all"),
    (8, "as-of budget replaced by the original budget after a revision", c08, ["G-FIN-ASOF-BUDGET"], "all"),
    (9, "ineligible snapshot inserted, contract counts updated", c09, ["G-ELIGIBILITY-SET"], "all"),
    (10, "eligible snapshot removed, contract counts updated", c10, ["G-ELIGIBILITY-SET"], "all"),
    (11, "production moved to another split group", c11, ["G-SPLIT-GROUP"], "all"),
    (12, "training fold reassigned", c12, ["G-SPLIT-FOLD"], "all"),
    (13, "final_pre_vat added as a prediction feature", c13, ["G-LEAKAGE-OUTCOME"], "all"),
    (14, "column renamed (data and contract)", c14, ["G-SCHEMA-COLUMNS"], "all"),
    (15, "column removed from the contract only", c15, ["G-SCHEMA-COLUMNS", "G-CONTRACT-SCHEMA"], "any"),
    (16, "incorrect raw provenance hash", c16, ["G-CONTRACT-PROVENANCE"], "all"),
    (17, "duplicate (production_id, stage) key", c17, ["G-KEY-UNIQUE"], "all"),
    (18, "null in a required feature", c18, ["G-NULLS"], "all"),
    (19, "required input file removed", c19, ["G-IN-FILES"], "all"),
]


def run_case(tmp: Path, reference, ineligible, case) -> dict:
    num, desc, fn, want, mode = case
    data, contract = reference
    d, c = data.copy(), json.loads(json.dumps(contract))
    ctx = {"ineligible": ineligible, "drop_data": False}
    fn(d, c, ctx)
    code, res, _ = _gate(tmp, d, c, drop_data=ctx["drop_data"])
    failing = _failing(res)
    hit = all(w in failing for w in want) if mode == "all" else any(w in failing for w in want)
    return {"case": num, "corruption": desc, "expected": want, "mode": mode, "observed_failing": failing,
            "gate_failed": res["status"] == "FAIL", "exit_code": code, "detected": hit}


@pytest.mark.parametrize("case", CASES, ids=[f"case{c[0]:02d}" for c in CASES])
def test_corruption_is_detected(tmp_path, reference, expected_ineligible, case):
    r = run_case(tmp_path, reference, expected_ineligible, case)
    print("CORRUPTION_RESULT " + json.dumps(r, sort_keys=True))
    assert r["gate_failed"] and r["exit_code"] == 1 and r["detected"], r
    if case[0] != 19:
        assert "G-IN-FILES" not in r["observed_failing"]           # the intended check really was evaluated


def test_corrupted_report_is_deterministic(tmp_path, reference, expected_ineligible):
    a = run_case(tmp_path / "a", reference, expected_ineligible, CASES[0])
    b = run_case(tmp_path / "b", reference, expected_ineligible, CASES[0])
    assert a == b
    assert (tmp_path / "a" / "report" / "gate_result.json").read_bytes() == \
           (tmp_path / "b" / "report" / "gate_result.json").read_bytes()


def test_missing_input_marks_dependent_checks_not_run(tmp_path, reference):
    code, res, _ = _gate(tmp_path, *reference, drop_data=True)
    status = {c["id"]: c["status"] for c in res["checks"]}
    assert code == 1 and status["G-IN-FILES"] == "FAIL"
    assert status["G-SCHEMA-COLUMNS"] == "NOT_RUN" and status["G-CONTRACT-SCHEMA"] == "PASS"
    assert status["G-CONTRACT-PROVENANCE"] == "PASS"


def test_unparsable_values_fail_without_crashing(tmp_path, reference):
    d, c = reference[0].copy(), json.loads(json.dumps(reference[1]))
    d.loc[d.index[0], ["budget_pre_vat", "budget_gap_ils"]] = ["abc", "nan"]
    code, res, _ = _gate(tmp_path, d, c)
    failing = _failing(res)
    assert code == 1 and {"G-SCHEMA-DTYPES", "G-FIN-RECON", "G-TARGET-GAP", "G-RANGES"} <= set(failing)
    assert not any(c["details"].startswith("check error") for c in res["checks"])


# ---------------------------------------------------------------- read boundary

def test_runtime_gate_never_reads_test_only_sources(tmp_path, reference, monkeypatch, generated):
    data_path, contract_path = write_handoff(*reference, tmp_path / "handoff")
    seen: list[Path] = []
    real = builtins.open

    def tracing_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            seen.append(Path(file).resolve())
        return real(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracing_open)
    monkeypatch.setattr(io, "open", tracing_open)
    assert main(["--stage", "source", "--raw", str(RAW), "--out", str(tmp_path / "s")]) == 0
    assert main(["--stage", "handoff", "--raw", str(RAW), "--out", str(tmp_path / "h"),
                 "--data", str(data_path), "--contract", str(contract_path)]) == 0
    monkeypatch.undo()
    # tracing works for both pandas reads and pathlib reads
    assert (RAW / "productions.csv").resolve() in seen and data_path.resolve() in seen
    assert (REPO / "contracts" / "frozen_split.json").resolve() in seen and contract_path.resolve() in seen
    raw_dir = RAW.resolve()
    forbidden_roots = [(REPO / p).resolve() for p in ("data/synthetic/v1", "ground_truth", "private",
                                                      "scan_output", "tests")] + [Path(generated["gt_dir"]).resolve()]
    for p in seen:
        assert not any(p.is_relative_to(f) for f in forbidden_roots), p
        if p.is_relative_to(ACTIVE_DATA.resolve()):
            assert p.is_relative_to(raw_dir), p                   # never the clean v2 tables


def test_gate_package_imports_neither_generator_nor_tests():
    for path in (REPO / "gate").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "import generator" not in text and "from generator" not in text, path.name
        assert "import tests" not in text and "from tests" not in text, path.name


# ---------------------------------------------------------------- Policy C equivalence (all 1,600)

@pytest.fixture(scope="module")
def gate_expected():
    return G.expected(G.load_raw(RAW))


def test_policy_c_gate_equals_oracle_on_all_1600(gate_expected):
    raw = G.load_raw(RAW)
    p, rows = gate_expected
    o = oracle_policy_c(raw)
    assert p.eligibility.equals(o.eligibility) and p.issues == o.issues and p.target_reliable == o.target_reliable
    ix = index({n: o.tables[n] for n in G.LEDGERS})
    assert len(rows) == 1600
    for (pid, stage), r in rows.items():
        assert r["labels"] == snapshot_labels(ix, pid, r["as_of_date"])
        if r["eligibility"]:
            assert r["features"] == snapshot_features(ix, pid, stage, r["as_of_date"])


def test_policy_c_matches_independent_truth_on_all_1600(gate_expected, expected_ineligible):
    _, rows = gate_expected
    snaps = pd.read_csv(ACTIVE_DATA / "snapshots.csv", dtype=str, keep_default_na=False)
    truth = snaps.set_index(["production_id", "stage"], drop=False)
    excluded = {k: r["reason"] for k, r in rows.items() if not r["eligibility"]}
    assert excluded == expected_ineligible and len(excluded) == 62
    assert Counter(excluded.values()) == {"commitment_category_missing": 50, "manual_cost_undated": 12}
    for key, r in rows.items():
        t = truth.loc[key]
        assert {c: as_csv_text(r["labels"][c]) for c in LABELS} == {c: t[c] for c in LABELS}, key
        if r["eligibility"]:
            assert {c: as_csv_text(r["features"][c]) for c in FEATURE_RULES} == {c: t[c] for c in FEATURE_RULES}, key
        else:
            assert r["features"] is None


# ---------------------------------------------------------------- committed data untouched

def _hashes():
    roots = [REPO / "data", REPO / "contracts"]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in roots for p in sorted(root.rglob("*")) if p.is_file()}


def test_gate_runs_leave_committed_files_unchanged(tmp_path, reference, expected_ineligible):
    before = _hashes()
    _gate(tmp_path / "ref", *reference)
    run_case(tmp_path / "c", reference, expected_ineligible, CASES[8])
    assert _hashes() == before
