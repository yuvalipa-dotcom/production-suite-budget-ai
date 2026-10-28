"""Crew 1 for the Flow: the approved P3 crew through a real CrewAI ``crew.kickoff()`` (P5-D). ALL DATA IS SYNTHETIC.

    python -m flow_crews.crew1_kickoff --out DIR

The crew is built by the unchanged P3 ``crew1.crew.build_crew`` (3 agents, 3 sequential tasks, each agent with
only its own deterministic tools) and kicked off in Crew 1's isolated process state. The model responses
come from ``crew_common.scripted.ScriptedTransport`` (no LLM inference, no network) by default. Only for a
scripted run, CrewAI's request-rate limit is switched off (``max_rpm``: it only delays requests to a real API);
the P3 files are not changed. With ``llm_mode="live"`` the same crew gets its responses from the Anthropic API
(``crew_common.live.RecordingTransport``: every provider call recorded, the P3 rate limits kept, a shared
request and token budget); there is no fallback to scripted responses.

Output: Crew 1's files (``clean_data.csv``, ``dataset_contract.json`` and the rest of the P3 handoff), the
redacted ``trace.jsonl`` and ``crew_kickoff.json`` (CrewAI events, scripted exchanges, Python tool calls).
The handoff gate runs inside the kickoff; the run fails unless it passes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import crew1.crew as CC
from crew1.tools import TOOLSETS
from crew1.tools.workspace import Sources, stage_workspace
from crew_common import kickoff as K
from crew_common import live as L
from crew_common import scripted as S
from crew_common.workspace import sha256_bytes

from . import crew1_script as SCRIPT

COPY = ("clean_data.csv", "cleaning_log.json", "stats.json", "eda_report.html", "dataset_contract.json",
        "gate_result.json", "insights.md")
HANDOFF = ("clean_data.csv", "dataset_contract.json")
AGENT_BY_TOOLS = {frozenset(names): key for key, names in TOOLSETS.items()}


class Crew1RunFailed(RuntimeError):
    pass


@contextlib.contextmanager
def _scripted_rate_limits():
    """No request-rate waiting while the transport is scripted (restored afterwards)."""
    saved_crew, saved_agents = CC.CREW_MAX_RPM, CC.AGENT_LIMITS
    CC.CREW_MAX_RPM = None
    CC.AGENT_LIMITS = {k: {**v, "max_rpm": None} for k, v in saved_agents.items()}
    try:
        yield
    finally:
        CC.CREW_MAX_RPM, CC.AGENT_LIMITS = saved_crew, saved_agents


def run_crew1_kickoff(out_dir: Path, log=None, llm_mode: str = "scripted", budget: L.LiveBudget | None = None) -> dict:
    if llm_mode not in ("scripted", "live"):
        raise ValueError(f"unknown llm_mode {llm_mode!r}")
    live = llm_mode == "live"
    out = Path(out_dir).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Crew1RunFailed("the output folder must be new or empty")
    started, t0 = datetime.now(timezone.utc), time.monotonic()
    ws = stage_workspace(Sources.default())
    try:
        recorder = CC.Recorder(ws.root)
        source = L.SOURCE if live else S.SOURCE
        recorder.add("run_started", orchestration="CrewAI crew.kickoff()", llm_responses=source)
        if live:
            transport = L.RecordingTransport(AGENT_BY_TOOLS, budget or L.LiveBudget.default(), "crew1", ws, log)
            crew, ctx = CC.build_crew(ws, Sources.default(), CC.DEFAULT_MODEL, recorder)
        else:
            transport = S.ScriptedTransport(SCRIPT.plans(), AGENT_BY_TOOLS, ws)
            with _scripted_rate_limits():
                crew, ctx = CC.build_crew(ws, Sources.default(), CC.DEFAULT_MODEL, recorder)
        evidence = K.KickoffEvidence()
        kickoff_started = L.utc_now()
        with transport.installed(), evidence.listening(), CC.isolated_process_state(ws.root):
            result = crew.kickoff()
        kickoff_seconds = round(time.monotonic() - t0, 1)
        recorder.add("run_finished")
        gate = json.loads(ws.read_output("out.gate_result")) if ws.has_output("out.gate_result") else {}
        missing = [n for n in COPY if not ws.has_output(f"out.{n.split('.')[0]}")]
        silent = [a for a in AGENT_BY_TOOLS.values()
                  if live and not transport.summary()["successful_responses_by_agent"].get(a)]
        if missing or gate.get("status") != "PASS" or transport.network_blocked or silent:
            raise Crew1RunFailed(f"Crew 1 kickoff incomplete: missing {missing}, gate {gate.get('status')}, "
                                 f"network {transport.network_blocked[:3]}, agents without a live response {silent}")
        out.mkdir(parents=True, exist_ok=True)
        for name in COPY:
            (out / name).write_bytes(ws.read_output(f"out.{name.split('.')[0]}"))
        tool_calls = [e for e in recorder.events if e["event"] == "tool_call"]
        kickoff = {
            "synthetic": True,
            "orchestration": "real CrewAI crew.kickoff() of the approved P3 Crew 1 (crew1.crew.build_crew, unchanged)",
            "llm_mode": llm_mode,
            "llm_responses": source,
            "disclosure": (("The CrewAI orchestration is real and every agent response came from live Anthropic "
                            f"inference ({CC.DEFAULT_MODEL}); each provider call is recorded in llm_transport.exchanges "
                            "(no header, no key). ") if live else
                           ("The CrewAI orchestration is real. The model responses are scripted (not LLM inference); "
                            f"every scripted text starts with '{S.SCRIPTED_MARK}'. "))
                          + "Every data value, statistic, chart, contract and gate result was computed by Crew 1's "
                            "deterministic Python tools.",
            "agents": [{"role": a.role, "tools": [t.name for t in a.tools]} for a in crew.agents],
            "tasks": [{"name": t.name, "agent": t.agent.role} for t in crew.tasks],
            "task_outputs": [{"task": o.name, "agent": o.agent, "source": "llm" if live else "scripted",
                              "text": recorder.redact(o.raw)}
                             for o in result.tasks_output],
            "crewai_events": evidence.summary() | {"events": evidence.events},
            "llm_transport": transport.summary() | {"exchanges": transport.exchanges},
            "python_tool_calls": {"computed_by": "python", "calls": len(tool_calls),
                                  "failed": sum(not e["ok"] for e in tool_calls)},
            "rate_limit_note": ("the P3 limits (max_rpm 20 per agent and crew)" if live
                                else "max_rpm switched off for the scripted transport only"),
            "gate": {"status": gate["status"], "checks": len(gate["checks"])},
            "handoff_sha256": {n: sha256_bytes((out / n).read_bytes()) for n in HANDOFF},
            "timing": {"started_at": started.isoformat(timespec="seconds"), "kickoff_seconds": kickoff_seconds,
                       "kickoff_started_utc": kickoff_started, "kickoff_finished_utc": L.utc_now()},
            "analyses_requested": ctx.analyses_requested, "charts_rendered": ctx.charts_rendered,
        }
        (out / "trace.jsonl").write_text(recorder.jsonl(), encoding="utf-8", newline="\n")
        (out / "crew_kickoff.json").write_text(json.dumps(kickoff, indent=2, default=str) + "\n", encoding="utf-8",
                                               newline="\n")
        if log:
            log(f"Crew 1 kickoff finished in {kickoff_seconds} s; agents executed: "
                f"{kickoff['crewai_events']['agents_executed']}; gate {gate['status']}")
        return kickoff
    finally:
        ws.cleanup()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    k = run_crew1_kickoff(args.out, log=lambda s: print(s, flush=True))   # scripted; live runs go through the Flow
    print(f"handoff written to {args.out}: {sorted(k['handoff_sha256'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
