"""P5-D: real CrewAI ``crew.kickoff()`` for both crews, with scripted model responses. Offline: no key, no
network, no LLM inference. ALL DATA IS SYNTHETIC.

* Crew 1: the approved P3 crew kicked off in-process; three agents execute their own tools; the gate
  passes and the delivered ``clean_data.csv`` is the approved handoff, which the Flow's Crew 2 accepts.
* Crew 2: the Flow crew kicked off in a guarded child process (every interpreter refuses Crew 1 imports,
  raw reads and the network; every TEST entry point is a counting refusal); its four artifacts are byte-
  identical to the P5-C TRAIN-only run.
* Hard failures: a contract violation, a changed pre-registration or a TEST access inside a tool stops the
  kickoff and nothing is written.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

from budget_flow import crew2_kickoff as K2
from budget_flow import handoff as FH
from crew2.ml import protocol as P
from crew2.ml import selection as SEL
from crew_common import scripted as S
from tests.conftest import REPO
from tests.test_flow_contract import scale
from tests.test_flow_train import _SITECUSTOMIZE, guarded_run  # noqa: F401  (the P5-C guarded run, session-scoped)

APPROVED = REPO / "artifacts" / "crew1" / "p3-live-005"
FOUR = ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md")
CREW2_ROLES = ["Handoff & Feature Validator", "TRAIN-only Model Trainer", "Evaluation & Ethics Author"]
CREW1_ROLES = ["Data Ingestion & Cleaning Engineer", "Exploratory Data Analyst", "Data Contract & Insights Author"]


def _check_kickoff_evidence(k: dict, roles: list[str], tasks: list[str]):
    ev = k["crewai_events"]
    c = ev["event_counts"]
    assert c["crew_kickoff_started"] == c["crew_kickoff_completed"] == 1
    assert c["task_started"] == c["task_completed"] == c["agent_execution_started"] == 3
    assert ev["agents_executed"] == roles and ev["tasks_completed"] == tasks
    assert set(ev["tools_by_agent"]) == set(roles)                       # every agent ran tools of its own
    assert c["llm_call_started"] == k["llm_transport"]["requests"] >= 6
    assert all(e["source"] == "scripted" for e in k["llm_transport"]["exchanges"])
    assert k["llm_transport"]["network_attempts_blocked"] == []
    assert sum(k["llm_transport"]["forced_final_answers"].values()) == 0
    assert [o["task"] for o in k["task_outputs"]] == tasks
    assert all(o["source"] == "scripted" and o["text"].startswith(S.SCRIPTED_MARK) for o in k["task_outputs"])
    assert k["python_tool_calls"]["calls"] == c["tool_usage_finished"] and k["python_tool_calls"]["failed"] == 0


# ---------------------------------------------------------------- Crew 1

@pytest.fixture(scope="module")
def crew1_run(tmp_path_factory):
    from flow_crews.crew1_kickoff import run_crew1_kickoff
    out = tmp_path_factory.mktemp("crew1_kickoff") / "out"
    return run_crew1_kickoff(out), out


def test_crew1_real_kickoff_with_three_agents(crew1_run):
    k, out = crew1_run
    _check_kickoff_evidence(k, CREW1_ROLES, ["ingest_and_clean", "explore_and_report", "contract_and_insights"])
    tools = k["crewai_events"]["tools_by_agent"]
    assert set(tools[CREW1_ROLES[0]]) == {"load_raw_table", "detect_defects", "apply_policy_c", "eligibility_summary"}
    assert set(tools[CREW1_ROLES[2]]) == {"build_dataset_contract", "run_handoff_gate", "write_insights"}
    assert k["gate"]["status"] == "PASS"


def test_crew1_delivers_the_approved_handoff_which_the_flow_crew2_accepts(crew1_run):
    from crew2.ml.data import EXPECTED_CLEAN_SHA256
    k, out = crew1_run
    assert k["handoff_sha256"]["clean_data.csv"] == EXPECTED_CLEAN_SHA256
    assert (out / "clean_data.csv").read_bytes() == (APPROVED / "clean_data.csv").read_bytes()
    report = FH.validate_handoff(out / "clean_data.csv", out / "dataset_contract.json")
    assert report.ok, report.codes


def test_crew1_trace_holds_no_path_and_no_secret(crew1_run):
    _, out = crew1_run
    text = (out / "trace.jsonl").read_text(encoding="utf-8") + (out / "crew_kickoff.json").read_text(encoding="utf-8")
    assert str(REPO).lower() not in text.lower() and "sk-ant-" not in text and S.DUMMY_KEY not in text


# ---------------------------------------------------------------- Crew 2: hard failures stop the kickoff

def _count_fits(monkeypatch):
    calls = {"run_cv": 0}
    real = P.Session.run_cv

    def counted(self, run_id):
        calls["run_cv"] += 1
        return real(self, run_id)

    monkeypatch.setattr(P.Session, "run_cv", counted)
    return calls


def test_a_contract_violation_stops_the_kickoff(tmp_path, monkeypatch):
    bad = tmp_path / "clean_data.csv"
    bad.write_bytes(scale("budget_pre_vat", 100))
    fits = _count_fits(monkeypatch)
    with pytest.raises(K2.Crew2RunFailed, match="UNIT_SCALE_ANOMALY"):
        K2.run_crew2_kickoff(bad, APPROVED / "dataset_contract.json", tmp_path / "out")
    assert not (tmp_path / "out").exists() and fits["run_cv"] == 0


def test_a_changed_preregistration_stops_the_kickoff_before_any_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(SEL, "BRIER_TIE", 0.01)
    fits = _count_fits(monkeypatch)
    with pytest.raises(K2.Crew2RunFailed, match="PreregistrationChanged"):
        K2.run_crew2_kickoff(APPROVED / "clean_data.csv", APPROVED / "dataset_contract.json", tmp_path / "out")
    assert not (tmp_path / "out").exists() and fits["run_cv"] == 0


def test_a_test_access_inside_a_tool_helper_stops_the_kickoff(tmp_path, monkeypatch):
    """An artifact helper that reaches for the test labels hits the sealed vault; the run fails."""
    def leaky(parts):
        parts.vault.open({"primary": "0" * 64})

    monkeypatch.setattr(K2.O, "render_features", leaky)
    fits = _count_fits(monkeypatch)
    with pytest.raises(K2.Crew2RunFailed, match="TestAccessError"):
        K2.run_crew2_kickoff(APPROVED / "clean_data.csv", APPROVED / "dataset_contract.json", tmp_path / "out")
    assert not (tmp_path / "out").exists() and fits["run_cv"] == 0


def test_a_raw_export_input_is_refused_before_kickoff(tmp_path):
    raw = REPO / "data" / "synthetic" / "v2" / "raw" / "productions.csv"
    with pytest.raises(FH.PathPolicyError):
        K2.run_crew2_kickoff(raw, APPROVED / "dataset_contract.json", tmp_path / "out")


# ---------------------------------------------------------------- Crew 2: the guarded full kickoff

_DRIVER = r'''
import json, os, sys
from budget_flow import crew2_kickoff as K2, handoff as FH
from crew2.ml import data as D, metrics as MT, protocol as P
calls = {"TestVault.open": 0, "SealedVault.open": 0, "Session.evaluate_on_test_once": 0, "Session.run_all": 0,
         "Session.metrics": 0, "bootstrap_ci": 0}
def counting(name, real=None):
    def f(*a, **k):
        calls[name] += 1
        if real is not None:
            return real(*a, **k)
        raise RuntimeError(name + " is forbidden in the Flow")
    return f
D.TestVault.open = counting("TestVault.open")
FH.SealedVault.open = counting("SealedVault.open", FH.SealedVault.open)
P.Session.evaluate_on_test_once = counting("Session.evaluate_on_test_once")
P.Session.run_all = counting("Session.run_all")
P.Session.metrics = counting("Session.metrics")
P.bootstrap_ci = MT.bootstrap_ci = counting("bootstrap_ci")
m = K2.run_crew2_kickoff(CLEAN, CONTRACT, OUT, n_jobs=N_JOBS)
print(json.dumps({"manifest": m, "calls": calls,
                  "crew1_modules": sorted(x for x in sys.modules if x.split(".")[0] == "crew1"),
                  "crewai_loaded": "crewai" in sys.modules,
                  "p4_crew_modules": sorted(x for x in sys.modules if x in ("crew2.validators", "crew2.tools",
                                                                           "crew2.crew", "crew2.offline", "crew2.run"))}))
'''


@pytest.fixture(scope="session")
def guarded_kickoff(tmp_path_factory):
    base = tmp_path_factory.mktemp("p5d_guarded")
    for name in ("clean_data.csv", "dataset_contract.json"):
        shutil.copyfile(APPROVED / name, base / name)
    harness = base / "harness"
    harness.mkdir()
    (harness / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
    log, out = base / "violations.log", base / "flow_out"
    n_jobs = max(1, min(5, os.cpu_count() or 1))
    code = (f"CLEAN = {str(base / 'clean_data.csv')!r}\nCONTRACT = {str(base / 'dataset_contract.json')!r}\n"
            f"OUT = {str(out)!r}\nN_JOBS = {n_jobs}\n") + _DRIVER
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON") and k != "ANTHROPIC_API_KEY"}
    env.update(PYTHONPATH=os.pathsep.join([str(harness), str(REPO)]), PSBA_GUARD_REPO=str(REPO),
               PSBA_GUARD_LOG=str(log), PSBA_GUARD_BLOCK="crew1", PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(base), env=env,
                         timeout=5400)
    assert res.returncode == 0, res.stderr[-4000:]
    evidence = json.loads(res.stdout.strip().splitlines()[-1])
    evidence["violations"] = log.read_text(encoding="utf-8") if log.exists() else ""
    evidence["out"] = out
    evidence["kickoff"] = json.loads((out / "crew_kickoff.json").read_text(encoding="utf-8"))
    return evidence


def test_crew2_real_kickoff_with_three_agents_and_their_own_tools(guarded_kickoff):
    k = guarded_kickoff["kickoff"]
    _check_kickoff_evidence(k, CREW2_ROLES, list(K2.load_config()[1]))
    tools = k["crewai_events"]["tools_by_agent"]
    assert tools == {role: [t for t in plan_tools] for role, plan_tools in zip(CREW2_ROLES, (
        ["validate_handoff", "write_features"],
        ["check_preregistration", "list_runs", *["run_grouped_cv"] * len(P.RUNS), "apply_selection_rule",
         "fit_final_models", "verify_reproducibility"],
        ["write_model", "write_evaluation_report", "write_model_card", "verify_artifacts"]))}
    assert k["agents"] == [{"key": key, "role": role, "tools": list(K2.TOOLSETS[key])}
                           for key, role in zip(K2.TOOLSETS, CREW2_ROLES)]
    assert "real CrewAI crew.kickoff()" in k["orchestration"] and "not LLM inference" in k["disclosure"]


def test_crew2_kickoff_boundary_and_sealed_test(guarded_kickoff):
    g = guarded_kickoff
    assert g["violations"] == ""                               # no Crew 1 import, raw read or network, any process
    assert g["crew1_modules"] == [] and g["p4_crew_modules"] == [] and g["crewai_loaded"]
    assert g["calls"] == {k: 0 for k in g["calls"]}
    assert g["manifest"]["test_vault"] == {"class": "SealedVault", "opened": False, "opens": 0,
                                           "test_metrics_computed": False}


def test_crew2_kickoff_results_equal_the_p5c_run(guarded_kickoff, guarded_run):  # noqa: F811
    m = guarded_kickoff["manifest"]
    assert m["selected_model"] == guarded_run["manifest"]["selected_model"]
    assert m["reproducibility"] == guarded_run["manifest"]["reproducibility"]
    for name in FOUR + ("features.schema.json",):
        assert (guarded_kickoff["out"] / name).read_bytes() == (guarded_run["out"] / name).read_bytes(), name


def test_crew2_kickoff_evidence_holds_no_row_data_path_or_secret(guarded_kickoff):
    out = guarded_kickoff["out"]
    for name in ("trace.jsonl", "crew_kickoff.json", "run_manifest.json"):
        text = (out / name).read_text(encoding="utf-8")
        assert "SYN-" not in text and str(REPO).lower() not in text.lower() and "sk-ant-" not in text, name
    events = [json.loads(line) for line in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(e["computed_by"] == "python" and e["ok"] for e in events if e["event"] == "tool_call")
