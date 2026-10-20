"""P5-E: the whole workflow as one CrewAI Flow. Offline: scripted LLM responses, no key, no network.
ALL DATA IS SYNTHETIC.

* Happy path: Crew 1 kickoff -> gate PASS -> Crew 2 kickoff -> all 8 required outputs, TEST never touched.
* Four deliberate breaks of THIS run's Crew 1 handoff (on the Flow's isolated copy): each stops at the gate
  with named violation codes; Crew 2's kickoff, every grouped-CV run and every model fit are called 0 times;
  no Crew 2 output exists; Crew 1's outputs are unchanged.
* Fail closed: a gate exception stops the run; a Crew 1 or Crew 2 runtime error is FAILED_RUNTIME, never PASS.
"""

from __future__ import annotations

import json

import pytest

from budget_flow import crew2_kickoff as K2
from budget_flow import p4_reference as R
from crew2.ml import data as D
from crew2.ml import metrics as MT
from crew2.ml import protocol as P
from crew2.ml.data import EXPECTED_CLEAN_SHA256
from crew_common.workspace import sha256_bytes
from orchestrator import flow as F
from orchestrator import guard as G
from tests.conftest import REPO

CREW1 = [f"crew1/{n}" for n in F.CREW1_REQUIRED]
CREW2 = [f"crew2/{n}" for n in F.CREW2_REQUIRED]
EXPECTED_CODES = {"column_rename": {"COLUMN_SET_MISMATCH"}, "money_x100": {"UNIT_SCALE_ANOMALY"},
                  "row_deletion": {"ROW_COUNT_MISMATCH"},
                  "contract_column_deletion": {"BASE_CONTRACT_COLUMNS"},
                  "label_leakage": {"COLUMN_SET_MISMATCH", "BASE_CONTRACT_COLUMNS"}}
CANONICAL = REPO / "flow_evidence" / "runs"


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _request_digests(crew_dir) -> list[str]:
    return [e["request_sha256"] for e in _json(crew_dir / "crew_kickoff.json")["llm_transport"]["exchanges"]]


def _instrument(mp) -> dict:
    """Count Crew 2 kickoffs and every fit; make every TEST entry point a counting refusal."""
    calls = {k: 0 for k in ("crew2_kickoff", "run_cv", "fit_final_models", "TestVault.open",
                            "Session.evaluate_on_test_once", "Session.run_all", "Session.metrics", "bootstrap_ci")}

    def counted(name, real):
        def f(*a, **k):
            calls[name] += 1
            return real(*a, **k)
        return f

    def refused(name):
        def f(*a, **k):
            calls[name] += 1
            raise RuntimeError(f"{name} is forbidden in the Flow")
        return f

    mp.setattr(K2, "run_crew2_kickoff", counted("crew2_kickoff", K2.run_crew2_kickoff))
    mp.setattr(P.Session, "run_cv", counted("run_cv", P.Session.run_cv))
    mp.setattr(P.Session, "fit_final_models", counted("fit_final_models", P.Session.fit_final_models))
    mp.setattr(D.TestVault, "open", refused("TestVault.open"))
    for name in ("evaluate_on_test_once", "run_all", "metrics"):
        mp.setattr(P.Session, name, refused(f"Session.{name}"))
    mp.setattr(P, "bootstrap_ci", refused("bootstrap_ci"))
    mp.setattr(MT, "bootstrap_ci", refused("bootstrap_ci"))
    return calls


@pytest.fixture(scope="module")
def happy(tmp_path_factory):
    out = tmp_path_factory.mktemp("flow_happy") / "run"
    with pytest.MonkeyPatch.context() as mp:
        calls = _instrument(mp)
        manifest = F.run_flow(out, n_jobs=5)
    return manifest, calls, out


# ---------------------------------------------------------------- happy path

def test_happy_flow_runs_every_step_through_real_flow_routing(happy):
    m, _, out = happy
    assert m["status"] == "PASS" and m["exit_code"] == 0 and m["error"] is None
    assert m["flow"]["methods_executed"] == ["run_crew1", "prepare_handoff", "contract_gate", "run_crew2",
                                             "verify_outputs", "finish_pass", "write_manifest"]
    assert m["flow"]["routes"] == {"contract_gate": "PASS", "verify_outputs": "COMPLETE"}
    events = [json.loads(line) for line in (out / "flow_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[0]["event"] == "flow_started" and events[-1]["event"] == "flow_finished"
    assert all(e.get("flow") == "BudgetFlow" for e in events)
    assert "scripted" in m["disclosure"] and "no model inference" in m["disclosure"]


def test_happy_flow_kicks_off_both_crews_with_three_agents_each(happy):
    m, calls, _ = happy
    for crew in ("crew1", "crew2"):
        ev = m[crew]["crewai_events"]
        assert ev["crew_kickoff_started"] == ev["crew_kickoff_completed"] == 1
        assert ev["agent_execution_started"] == ev["task_completed"] == 3 and len(m[crew]["agents_executed"]) == 3
    assert m["crew2"]["kickoffs"] == calls["crew2_kickoff"] == 1


def test_happy_flow_writes_and_hashes_all_eight_required_outputs(happy):
    m, _, out = happy
    assert sorted(m["required_outputs_sha256"]) == sorted(CREW1 + CREW2)
    for rel, sha in m["required_outputs_sha256"].items():
        assert sha256_bytes((out / rel).read_bytes()) == sha
    assert m["crew1_outputs_unchanged_by_the_flow"]


def test_happy_flow_uses_this_runs_crew1_handoff(happy):
    m, _, out = happy
    delivered = m["handoff"]["as_delivered_sha256"]
    assert delivered["clean_data.csv"] == m["crew1"]["outputs_sha256"]["clean_data.csv"] == EXPECTED_CLEAN_SHA256
    assert delivered["dataset_contract.json"] == m["crew1"]["outputs_sha256"]["dataset_contract.json"]
    assert m["handoff"]["as_gated_sha256"] == delivered and m["gate"]["validated_files_sha256"] == delivered
    assert m["crew2"]["input_sha256"]["dataset_contract.json"] == delivered["dataset_contract.json"]


def test_happy_flow_keeps_test_sealed_and_the_approved_model(happy):
    m, calls, _ = happy
    assert {k: v for k, v in calls.items() if k not in ("crew2_kickoff", "run_cv", "fit_final_models")} == {
        k: 0 for k in calls if k not in ("crew2_kickoff", "run_cv", "fit_final_models")}
    assert calls["run_cv"] == len(P.RUNS) and calls["fit_final_models"] == 1
    c2 = m["crew2"]
    assert c2["test_vault"] == {"class": "SealedVault", "opened": False, "opens": 0, "test_metrics_computed": False}
    assert c2["historical_test"]["recomputed"] is False and c2["boundary_violations"] == []
    assert c2["selected_model"]["candidate"] == "LR" and c2["selected_model"]["params"] == {"C": 10.0}
    assert c2["reproducibility"]["ok"] and c2["reproducibility"]["selection_equal"]
    if c2["reproducibility"]["mode"] == "exact":
        assert c2["selected_model"]["model_sha256"] == R.HIST_MODEL_SHA256


def test_happy_flow_reproduces_the_canonical_request_digests(happy):
    """P5-F: the scripted-LLM request digests spared by the scanner exemption are reproduced by a fresh run."""
    m, _, out = happy
    assert _request_digests(out / "crew1") == _request_digests(CANONICAL / "p5e-run-001" / "crew1")
    if m["crew2"]["reproducibility"]["mode"] == "exact":        # Crew 2's tool results carry the CV numbers
        assert _request_digests(out / "crew2") == _request_digests(CANONICAL / "p5e-run-001" / "crew2")


def test_happy_flow_evidence_holds_no_row_data_path_or_secret(happy):
    _, _, out = happy
    for name in ("flow_manifest.json", "flow_events.jsonl", "handoff/gate_report.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert "SYN-" not in text and "sk-ant-" not in text, name
    assert str(REPO).lower() not in (out / "flow_events.jsonl").read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------- deliberate breaks

@pytest.mark.parametrize("scenario", sorted(EXPECTED_CODES))
def test_a_contract_break_stops_the_flow_before_crew2(tmp_path, scenario):
    out = tmp_path / "run"
    with pytest.MonkeyPatch.context() as mp:
        calls = _instrument(mp)
        m = F.run_flow(out, scenario=scenario)
    assert m["status"] == "STOPPED_CONTRACT" and m["exit_code"] == 3
    codes = set(m["gate"]["violation_codes"])
    assert EXPECTED_CODES[scenario] <= codes
    if scenario == "money_x100":                     # the regenerated matching contract does not bypass the check
        assert codes == {"UNIT_SCALE_ANOMALY"} and m["handoff"]["scenario"]["attacker_regenerated_flow_contract"]
    if scenario == "contract_column_deletion":       # the data are untouched; the contract itself lost a column
        assert m["handoff"]["as_gated_sha256"]["clean_data.csv"] == m["handoff"]["as_delivered_sha256"]["clean_data.csv"]
        assert any("budget_pre_vat" in v["detail"] for v in m["gate"]["violations"]
                   if v["code"] == "BASE_CONTRACT_COLUMNS")
    assert m["flow"]["routes"] == {"contract_gate": "STOPPED_CONTRACT"}
    assert m["flow"]["methods_executed"] == ["run_crew1", "prepare_handoff", "contract_gate", "stop_on_contract",
                                             "write_manifest"]
    assert calls == {k: 0 for k in calls}            # no Crew 2 kickoff, no CV, no fit, no TEST access
    assert m["crew2"] == {"kickoff_invoked": False, "kickoffs": 0} and not (out / "crew2").exists()
    assert sorted(m["required_outputs_sha256"]) == sorted(CREW1)
    assert m["crew1_outputs_unchanged_by_the_flow"]
    assert m["crew1"]["outputs_sha256"]["clean_data.csv"] == EXPECTED_CLEAN_SHA256
    assert m["handoff"]["as_gated_sha256"] != m["handoff"]["as_delivered_sha256"]
    gate = json.loads((out / "handoff" / "gate_report.json").read_text(encoding="utf-8"))
    assert gate["decision"] == "STOPPED_CONTRACT" and set(gate["violation_codes"]) == codes
    # P5-F: the digests the scanner exemption spares in the canonical evidence are reproduced by this fresh run
    canonical = CANONICAL / f"p5e-break-{scenario}"
    assert gate["validated_files_sha256"] == _json(canonical / "handoff" / "gate_report.json")["validated_files_sha256"]
    assert _request_digests(out / "crew1") == _request_digests(canonical / "crew1")


def test_the_cli_reports_a_stopped_contract_with_its_own_exit_code(tmp_path, capsys):
    assert F.main(["--out", str(tmp_path / "run"), "--scenario", "row_deletion"]) == 3
    assert '"status": "STOPPED_CONTRACT"' in capsys.readouterr().out


# ---------------------------------------------------------------- fail closed and runtime failures

def test_a_gate_exception_fails_closed(tmp_path, monkeypatch):
    def broken(*a, **k):
        raise OSError("unreadable")
    monkeypatch.setattr(F.FH, "validate_handoff", broken)
    calls = _instrument(monkeypatch)
    m = F.run_flow(tmp_path / "run")
    assert m["status"] == "STOPPED_CONTRACT" and m["gate"]["violation_codes"] == ["GATE_EXCEPTION"]
    assert calls["crew2_kickoff"] == calls["run_cv"] == 0


def test_a_crew2_runtime_error_is_failed_runtime_not_pass(tmp_path, monkeypatch):
    def crash(*a, **k):
        raise RuntimeError("simulated Crew 2 crash")
    monkeypatch.setattr(K2, "run_crew2_kickoff", crash)
    m = F.run_flow(tmp_path / "run")
    assert m["status"] == "FAILED_RUNTIME" and m["exit_code"] == 1 and "simulated Crew 2 crash" in m["error"]
    assert m["flow"]["routes"] == {"contract_gate": "PASS", "verify_outputs": "FAILED_RUNTIME"}
    assert "finish_pass" not in m["flow"]["methods_executed"] and sorted(m["required_outputs_sha256"]) == sorted(CREW1)


def test_a_crew1_runtime_error_never_reaches_the_gate_or_crew2(tmp_path, monkeypatch):
    import flow_crews.crew1_kickoff as K1

    def crash(*a, **k):
        raise RuntimeError("simulated Crew 1 crash")
    monkeypatch.setattr(K1, "run_crew1_kickoff", crash)
    calls = _instrument(monkeypatch)
    m = F.run_flow(tmp_path / "run")
    assert m["status"] == "FAILED_RUNTIME" and m["flow"]["routes"] == {"contract_gate": "FAILED_RUNTIME"}
    assert calls["crew2_kickoff"] == 0 and m["gate"] == {}


def test_a_run_folder_is_never_reused(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        F.run_flow(tmp_path)


# ---------------------------------------------------------------- the Crew 2 runtime boundary

def test_the_crew2_boundary_refuses_raw_crew1_and_crew1_outputs_only_while_active(tmp_path):
    crew1_out = tmp_path / "crew1"
    crew1_out.mkdir()
    (crew1_out / "clean_data.csv").write_text("x")
    raw = REPO / "data" / "synthetic" / "v2" / "raw" / "productions.csv"
    with G.crew2_boundary(REPO, crew1_out) as rec:
        for p in (raw, crew1_out / "clean_data.csv", REPO / "crew1" / "crew.py"):
            with pytest.raises(PermissionError):
                open(p, "rb").close()
        (tmp_path / "ok.txt").write_text("allowed")
    assert len(rec["violations"]) == 3
    open(raw, "rb").close()                                # inert outside the Crew 2 stage


def test_historical_evidence_is_unchanged():
    import hashlib
    d = REPO / "artifacts" / "crew1" / "p3-live-005"
    manifest = json.loads((d / "run_manifest.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((d / k).read_bytes()).hexdigest() == v for k, v in manifest["artifacts_sha256"].items())
