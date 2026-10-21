"""The academic workflow as one CrewAI Flow (P5-E). ALL DATA IS SYNTHETIC.

    python -m orchestrator.flow --out out/flow_runs/<run_id> [--scenario NAME] [--n-jobs N]

    @start()                          run_crew1         Crew 1 kickoff -> <run>/crew1/ (its 4 required outputs)
    @listen(run_crew1)                prepare_handoff   isolated copy of THIS run's Crew 1 handoff -> <run>/handoff/,
                                                        its flow-1 contract, then the optional deliberate break
    @router(prepare_handoff)          contract_gate     "PASS" | "STOPPED_CONTRACT" | "FAILED_RUNTIME" (fail closed)
    @listen("PASS")                   run_crew2         Crew 2 kickoff, TRAIN-only -> <run>/crew2/
    @router(run_crew2)                verify_outputs    "COMPLETE" | "FAILED_RUNTIME": all 8 outputs present
    @listen("COMPLETE")               finish_pass       status PASS
    @listen("STOPPED_CONTRACT")       stop_on_contract  status STOPPED_CONTRACT, violation codes; Crew 2 never runs
    @listen("FAILED_RUNTIME")         record_failure    status FAILED_RUNTIME
    @listen(or_(finish_pass, stop_on_contract, record_failure))
                                      write_manifest    <run>/flow_manifest.json and <run>/flow_events.jsonl

The CrewAI orchestration (Flow, both Crews, their Agents and Tasks) is real; the LLM responses are scripted
(``crew_common.scripted``: no inference, no network); every number, check, fit and artifact is computed by
deterministic Python. Statuses: PASS (exit 0), STOPPED_CONTRACT (exit 3, the run stopped on purpose: no model
was trained), FAILED_RUNTIME (exit 1).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

import crew_common  # noqa: F401  (forces the telemetry settings before CrewAI is imported)
from budget_flow import contract as C
from budget_flow import crew2_kickoff as K2
from budget_flow import handoff as FH
from crew_common import scripted as S
from crew_common.workspace import REPO, sha256_bytes

from . import guard as G
from . import scenarios as SC

CREW1_REQUIRED = ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json")
CREW2_REQUIRED = ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md")
HANDOFF = ("clean_data.csv", "dataset_contract.json")
STATUSES = {"PASS": 0, "FAILED_RUNTIME": 1, "STOPPED_CONTRACT": 3}
DISCLOSURE = ("The CrewAI Flow, both Crews, their Agents and Tasks run for real (crew.kickoff() inside Flow "
              "methods). The LLM responses are scripted (" + S.SOURCE + "): no model inference takes place. Every "
              "number, check, fit and artifact is computed by deterministic Python tools.")


class FlowState(BaseModel):
    out: str = ""
    scenario: str = "none"
    n_jobs: int = 1
    status: str = "RUNNING"
    route: str = ""
    error: str | None = None
    timings: dict = Field(default_factory=dict)
    crew1: dict = Field(default_factory=dict)
    handoff: dict = Field(default_factory=dict)
    gate: dict = Field(default_factory=dict)
    crew2: dict = Field(default_factory=lambda: {"kickoff_invoked": False, "kickoffs": 0})
    outputs: dict = Field(default_factory=dict)


def _hashes(folder: Path, names) -> dict:
    return {n: sha256_bytes((folder / n).read_bytes()) for n in names if (folder / n).is_file()}


def _rel(p: Path, root: Path) -> str:
    return p.relative_to(root).as_posix()


def _flow_class():
    from crewai.flow.flow import Flow, listen, or_, router, start

    class BudgetFlow(Flow[FlowState]):
        """Crew 1 -> contract gate -> Crew 2 (TRAIN-only), with explicit routing."""

        def _timed(self, name, t0):
            self.state.timings[name] = round(time.monotonic() - t0, 1)

        def _fail(self, where: str, e: BaseException):
            self.state.error = f"{where}: {type(e).__name__}: {e}"[:800]

        @start()
        def run_crew1(self):
            from flow_crews.crew1_kickoff import run_crew1_kickoff
            t0, root = time.monotonic(), Path(self.state.out)
            try:
                k = run_crew1_kickoff(root / "crew1")
                self.state.crew1 = {
                    "kickoff": "crew1/crew_kickoff.json", "agents_executed": k["crewai_events"]["agents_executed"],
                    "crewai_events": k["crewai_events"]["event_counts"], "llm_requests": k["llm_transport"]["requests"],
                    "python_tool_calls": k["python_tool_calls"]["calls"], "gate": k["gate"],
                    "outputs_sha256": _hashes(root / "crew1", CREW1_REQUIRED)}
            except Exception as e:                                # noqa: BLE001  (routed to FAILED_RUNTIME)
                self._fail("crew1", e)
            self._timed("crew1_kickoff", t0)
            return "crew1 done"

        @listen(run_crew1)
        def prepare_handoff(self):
            """An isolated copy of the handoff Crew 1 wrote in THIS run, its flow-1 contract, then the break."""
            if self.state.error:
                return "skipped"
            t0, root = time.monotonic(), Path(self.state.out)
            try:
                src, dst = root / "crew1", root / "handoff"
                missing = [n for n in CREW1_REQUIRED if not (src / n).is_file()]
                self.state.handoff["crew1_outputs_missing"] = missing
                dst.mkdir()
                for n in HANDOFF:
                    if (src / n).is_file():
                        shutil.copyfile(src / n, dst / n)
                if not missing:
                    flow = C.build_flow_contract((dst / "clean_data.csv").read_bytes(),
                                                 (dst / "dataset_contract.json").read_bytes())
                    (dst / "flow_contract.json").write_text(json.dumps(flow, indent=2) + "\n", encoding="utf-8",
                                                            newline="\n")
                self.state.handoff["as_delivered_sha256"] = _hashes(dst, (*HANDOFF, "flow_contract.json"))
                if self.state.scenario != "none" and not missing:
                    rec = SC.apply(self.state.scenario, dst)
                    (dst / "scenario.json").write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8",
                                                       newline="\n")
                    self.state.handoff["scenario"] = rec
                self.state.handoff["as_gated_sha256"] = _hashes(dst, (*HANDOFF, "flow_contract.json"))
            except Exception as e:                                # noqa: BLE001
                self._fail("prepare_handoff", e)
            self._timed("prepare_handoff", t0)
            return "handoff ready"

        @router(prepare_handoff)
        def contract_gate(self):
            """Fail closed: only an explicit, complete, violation-free report routes to PASS."""
            if self.state.error:
                self.state.route = "FAILED_RUNTIME"
                return "FAILED_RUNTIME"
            t0, dst = time.monotonic(), Path(self.state.out) / "handoff"
            decision, codes, violations = "STOPPED_CONTRACT", [], []
            try:
                missing = self.state.handoff.get("crew1_outputs_missing", [])
                if missing:
                    codes = ["CREW1_OUTPUT_MISSING"]
                    violations = [{"code": "CREW1_OUTPUT_MISSING", "column": None, "detail": f"missing {missing}"}]
                else:
                    report = FH.validate_handoff(dst / "clean_data.csv", dst / "dataset_contract.json",
                                                 dst / "flow_contract.json")
                    codes = sorted(report.codes)
                    violations = [v.as_dict() for v in report.violations]
                    if report.ok and not codes:
                        decision = "PASS"
            except Exception as e:                                # noqa: BLE001  (fail closed)
                codes = ["GATE_EXCEPTION"]
                violations = [{"code": "GATE_EXCEPTION", "column": None, "detail": f"{type(e).__name__}: {e}"[:300]}]
            self.state.gate = {"contract": C.FLOW_CONTRACT_VERSION, "decision": decision, "violation_codes": codes,
                               "violations": violations[:50], "violation_count": len(violations),
                               "validated_files_sha256": _hashes(dst, (*HANDOFF, "flow_contract.json"))}
            (dst / "gate_report.json").write_text(json.dumps(self.state.gate, indent=2) + "\n", encoding="utf-8",
                                                  newline="\n")
            self._timed("contract_gate", t0)
            self.state.route = decision
            return decision

        @listen("PASS")
        def run_crew2(self):
            t0, root = time.monotonic(), Path(self.state.out)
            self.state.crew2.update(kickoff_invoked=True, kickoffs=self.state.crew2["kickoffs"] + 1)
            dst = root / "handoff"
            try:
                with G.crew2_boundary(REPO, root / "crew1") as boundary:
                    m = K2.run_crew2_kickoff(dst / "clean_data.csv", dst / "dataset_contract.json", root / "crew2",
                                             flow_contract_json=dst / "flow_contract.json", n_jobs=self.state.n_jobs)
                k = json.loads((root / "crew2" / "crew_kickoff.json").read_text(encoding="utf-8"))
                self.state.crew2.update(
                    kickoff="crew2/crew_kickoff.json", agents_executed=k["crewai_events"]["agents_executed"],
                    crewai_events=k["crewai_events"]["event_counts"], llm_requests=k["llm_transport"]["requests"],
                    python_tool_calls=k["python_tool_calls"]["calls"], selected_model=m["selected_model"],
                    reproducibility=m["reproducibility"], test_vault=m["test_vault"],
                    historical_test=m["historical_test"], input_sha256=m["input_sha256"])
            except Exception as e:                                # noqa: BLE001  (routed to FAILED_RUNTIME)
                self._fail("crew2", e)
                boundary = locals().get("boundary") or {"violations": []}
            self.state.crew2["boundary_violations"] = boundary["violations"]
            self._timed("crew2_kickoff", t0)
            return "crew2 done"

        @router(run_crew2)
        def verify_outputs(self):
            root = Path(self.state.out)
            paths = [root / "crew1" / n for n in CREW1_REQUIRED] + [root / "crew2" / n for n in CREW2_REQUIRED]
            self.state.outputs = {_rel(p, root): sha256_bytes(p.read_bytes()) for p in paths if p.is_file()}
            ok = (not self.state.error and len(self.state.outputs) == 8 and not self.state.crew2["boundary_violations"]
                  and self.state.crew2.get("test_vault", {}).get("opens") == 0)
            if not ok and not self.state.error:
                self.state.error = "verify_outputs: the 8 required outputs or the boundary checks are incomplete"
            self.state.route = "COMPLETE" if ok else "FAILED_RUNTIME"
            return self.state.route

        @listen("COMPLETE")
        def finish_pass(self):
            self.state.status = "PASS"
            return "PASS"

        @listen("STOPPED_CONTRACT")
        def stop_on_contract(self):
            self.state.status = "STOPPED_CONTRACT"
            return "STOPPED_CONTRACT"

        @listen("FAILED_RUNTIME")
        def record_failure(self):
            self.state.status = "FAILED_RUNTIME"
            return "FAILED_RUNTIME"

        @listen(or_(finish_pass, stop_on_contract, record_failure))
        def write_manifest(self):
            root = Path(self.state.out)
            if self.state.status != "PASS":                   # a stopped or failed run leaves no Crew 2 output
                self.state.outputs = {_rel(p, root): sha256_bytes(p.read_bytes())
                                      for p in (root / "crew1" / n for n in CREW1_REQUIRED) if p.is_file()}
            return self.state.status

    return BudgetFlow


class FlowEvidence:
    """The Flow's own lifecycle events from CrewAI's event bus: flow started / finished, every method's start and
    finish (with the router's returned route), and any method failure."""

    def __init__(self):
        self.events: list[dict] = []
        self.t0 = time.monotonic()

    def listening(self):
        from crewai.events import crewai_event_bus as bus
        from crewai.events.types import flow_events as FE

        scope = bus.scoped_handlers()
        scope.__enter__()
        kinds = {"FlowStartedEvent": "flow_started", "FlowFinishedEvent": "flow_finished",
                 "MethodExecutionStartedEvent": "method_started", "MethodExecutionFinishedEvent": "method_finished",
                 "MethodExecutionFailedEvent": "method_failed"}
        def make_handler(kind):
            # The bus may pass extra positional arguments (its runtime state); only (source, event) are used.
            def handler(source, event, *_):
                rec = {"t": round(time.monotonic() - self.t0, 3), "event": kind}
                if isinstance(getattr(event, "flow_name", None), str):
                    rec["flow"] = event.flow_name
                if isinstance(getattr(event, "method_name", None), str):
                    rec["method"] = event.method_name
                if kind == "method_finished" and isinstance(getattr(event, "result", None), str):
                    rec["result"] = event.result
                self.events.append(rec)
            return handler

        for cls_name, kind in kinds.items():
            bus.on(getattr(FE, cls_name))(make_handler(kind))
        return scope, bus


def run_flow(out: Path, scenario: str = "none", n_jobs: int = 1, log=None) -> dict:
    """Run the Flow once into ``out`` (new or empty). Returns the flow manifest."""
    out = Path(out).resolve()
    if scenario not in ("none", *SC.SCENARIOS):
        raise ValueError(f"unknown scenario {scenario!r}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("the run folder must be new or empty")
    out.mkdir(parents=True, exist_ok=True)
    started, t0 = datetime.now(timezone.utc), time.monotonic()
    flow = _flow_class()(suppress_flow_events=False)
    flow.state.out, flow.state.scenario, flow.state.n_jobs = str(out), scenario, n_jobs
    evidence = FlowEvidence()
    scope, bus = evidence.listening()
    try:
        flow.kickoff()
    except Exception as e:                                       # noqa: BLE001  (a Flow-level crash)
        flow.state.status, flow.state.error = "FAILED_RUNTIME", f"flow: {type(e).__name__}: {e}"[:800]
        flow.state.timings["traceback"] = traceback.format_exc()[-2000:]
    finally:
        bus.flush()
        scope.__exit__(None, None, None)
    st = flow.state
    if st.status == "RUNNING":
        st.status = "FAILED_RUNTIME"
    crew2_dir = out / "crew2"
    if st.status != "PASS" and crew2_dir.exists() and not st.crew2["kickoff_invoked"]:
        raise RuntimeError("a Crew 2 folder exists although Crew 2 was never invoked")
    methods = [e["method"] for e in evidence.events if e["event"] == "method_started"]
    manifest = {
        "synthetic": True, "status": st.status, "exit_code": STATUSES[st.status],
        "status_meaning": {"PASS": "both crews ran and all 8 required outputs were verified",
                           "STOPPED_CONTRACT": "the contract gate stopped the run on purpose; Crew 2 never ran and no "
                                               "model was trained",
                           "FAILED_RUNTIME": "the run failed at runtime; see error"}[st.status],
        "scenario": st.scenario, "disclosure": DISCLOSURE,
        "flow": {"class": "BudgetFlow (crewai.flow.flow.Flow)", "flow_id": getattr(st, "id", None),
                 "decorators": {"start": ["run_crew1"], "listen": ["prepare_handoff", "run_crew2 (route PASS)",
                                "finish_pass (route COMPLETE)", "stop_on_contract (route STOPPED_CONTRACT)",
                                "record_failure (route FAILED_RUNTIME)",
                                "write_manifest (or_ finish_pass, stop_on_contract, record_failure)"],
                                "router": ["contract_gate", "verify_outputs"]},
                 "methods_executed": methods,
                 "routes": {e["method"]: e.get("result") for e in evidence.events
                            if e["event"] == "method_finished" and e.get("method") in ("contract_gate", "verify_outputs")},
                 "events": "flow_events.jsonl"},
        "crew1": st.crew1, "handoff": st.handoff, "gate": st.gate, "crew2": st.crew2,
        "required_outputs_sha256": st.outputs,
        "crew1_outputs_unchanged_by_the_flow": st.crew1.get("outputs_sha256") == _hashes(out / "crew1", CREW1_REQUIRED),
        "error": st.error,
        "timing": {"started_at": started.isoformat(timespec="seconds"), "total_seconds": round(time.monotonic() - t0, 1),
                   **st.timings},
        "n_jobs": n_jobs,
    }
    (out / "flow_events.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in evidence.events),
                                           encoding="utf-8", newline="\n")
    (out / "flow_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8",
                                            newline="\n")
    if log:
        log(f"status {st.status} in {manifest['timing']['total_seconds']} s; methods {methods}; "
            f"gate {st.gate.get('decision')} {st.gate.get('violation_codes')}")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="a new run folder, e.g. out/flow_runs/<run_id>")
    ap.add_argument("--scenario", default="none", choices=("none", *SC.SCENARIOS),
                    help="a deliberate contract break applied to the Flow's handoff copy (default: none)")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args(argv)
    m = run_flow(args.out, args.scenario, args.n_jobs, log=lambda s: print(s, flush=True))
    if m["status"] == "PASS":
        print(f"selected {m['crew2']['selected_model']['candidate']} {m['crew2']['selected_model']['params']}")
    print(json.dumps({"status": m["status"], "gate": m["gate"].get("violation_codes"),
                      "outputs": sorted(m["required_outputs_sha256"])}, indent=2))
    return m["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
