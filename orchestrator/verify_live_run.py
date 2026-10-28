"""Verify the evidence of one live-LLM Flow run, from its files alone. ALL DATA IS SYNTHETIC.

    python -m orchestrator.verify_live_run <run folder> [--json]

Standard library only: it imports neither CrewAI nor the project, makes no network call and reads no key.
Exit 0 when every check passes, 1 otherwise. The checks:

L1  the Flow finished PASS (exit code 0, no error) in live LLM mode
L2  the Flow's own routing: Crew 1 -> handoff -> gate (PASS) -> Crew 2 -> output check (COMPLETE)
L3  the deterministic contract gate returned PASS with 0 violations, on exactly the files in handoff/
L4  all 8 required outputs exist and match the sha256 values recorded in the manifest
L5  every provider exchange is a live Anthropic call: HTTP status, provider request-id, message id, model and
    token usage recorded; no scripted, mock or offline-test identifiers
L6  each of the six agents (3 in Crew 1, 3 in Crew 2) received at least one successful live response
L7  no scripted response anywhere in the run (no scripted marker, no exchange marked scripted)
L8  ordering by UTC timestamps: every Crew 1 provider call finished before the gate decision; the gate decided
    PASS before Crew 2 started; every Crew 2 provider call started after Crew 2 started
L9  the manifest's provider-call and token totals equal the sum of the recorded exchanges
L10 no API key in the evidence (no "sk-ant-" string; the value of ANTHROPIC_API_KEY, if set, appears nowhere)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

CREW1_AGENTS = ("ingestion_engineer", "exploratory_analyst", "contract_author")
CREW2_AGENTS = ("handoff_validator", "train_only_modeler", "evaluation_author")
REQUIRED = tuple(f"crew1/{n}" for n in ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json")) \
    + tuple(f"crew2/{n}" for n in ("features.csv", "model.joblib", "evaluation_report.md", "model_card.md"))
HANDOFF = ("clean_data.csv", "dataset_contract.json", "flow_contract.json")
METHODS = ["run_crew1", "prepare_handoff", "contract_gate", "run_crew2", "verify_outputs", "finish_pass",
           "write_manifest"]
SCRIPTED_MARK = "[SCRIPTED RESPONSE - not LLM inference]"
NOT_GENUINE = ("msg_scripted", "msg_mock", "msg_offline", "req_offline")
USAGE_KEYS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def verify(run: Path, environ=os.environ) -> list[dict]:
    run = Path(run)
    results: list[dict] = []

    def check(cid: str, title: str, problems: list[str], detail: str = ""):
        results.append({"id": cid, "check": title, "status": "PASS" if not problems else "FAIL",
                        "detail": detail if not problems else "; ".join(problems[:8])})

    try:
        m = _json(run / "flow_manifest.json")
        kick = {c: _json(run / c / "crew_kickoff.json") for c in ("crew1", "crew2")}
    except (OSError, ValueError, KeyError) as e:
        return [{"id": "L0", "check": "evidence files readable", "status": "FAIL", "detail": f"{type(e).__name__}: {e}"}]
    llm = m.get("llm") or {}
    ex = {c: kick[c]["llm_transport"].get("exchanges", []) for c in kick}

    check("L1", "Flow status PASS in live LLM mode",
          [f"{k}={v!r}" for k, v in (("status", m.get("status")), ("exit_code", m.get("exit_code")),
                                    ("error", m.get("error")), ("llm.mode", llm.get("mode")))
           if v != {"status": "PASS", "exit_code": 0, "error": None, "llm.mode": "live"}[k]]
          + [f"{c} llm_mode={kick[c].get('llm_mode')!r}" for c in kick if kick[c].get("llm_mode") != "live"],
          f"status PASS, exit 0, model {llm.get('models_reported_by_provider')}")

    flow = m.get("flow", {})
    routes = flow.get("routes", {})
    check("L2", "Flow routing: Crew 1 -> gate PASS -> Crew 2 -> COMPLETE",
          ([f"methods {flow.get('methods_executed')}"] if flow.get("methods_executed") != METHODS else [])
          + ([f"routes {routes}"] if routes != {"contract_gate": "PASS", "verify_outputs": "COMPLETE"} else []),
          " -> ".join(METHODS))

    p3 = []
    try:
        gate = _json(run / "handoff" / "gate_report.json")
        if gate.get("decision") != "PASS" or gate.get("violation_count") != 0 or gate.get("violation_codes"):
            p3.append(f"gate {gate.get('decision')} {gate.get('violation_codes')}")
        if gate != m.get("gate"):
            p3.append("gate_report.json differs from the manifest's gate record")
        actual = {n: _sha(run / "handoff" / n) for n in HANDOFF if (run / "handoff" / n).is_file()}
        if gate.get("validated_files_sha256") != actual:
            p3.append("the gated files differ from the files in handoff/")
        if (run / "handoff" / "scenario.json").exists():
            p3.append("a deliberate contract break was applied (scenario.json)")
    except (OSError, ValueError) as e:
        p3.append(f"gate report unreadable: {e}")
    check("L3", "deterministic contract gate PASS, 0 violations", p3, "handoff/gate_report.json")

    rec = m.get("required_outputs_sha256", {})
    p4 = [f"missing {n}" for n in REQUIRED if not (run / n).is_file()]
    p4 += [f"sha256 mismatch {n}" for n in REQUIRED if (run / n).is_file() and rec.get(n) != _sha(run / n)]
    p4 += [f"unexpected recorded output set {sorted(rec)}"] if sorted(rec) != sorted(REQUIRED) else []
    check("L4", "all 8 required outputs present and hash-verified", p4, ", ".join(REQUIRED))

    p5 = []
    for c, rows in ex.items():
        if not rows:
            p5.append(f"{c}: no provider exchange recorded")
        for e in rows:
            tag = f"{c} #{e.get('n')}"
            if e.get("source") != "live":
                p5.append(f"{tag} source {e.get('source')!r}")
            if "refused" in e:
                continue                                  # refused before sending: listed, not a provider call
            ids = f"{e.get('message_id')} {e.get('provider_request_id')}"
            if any(s in ids for s in NOT_GENUINE):
                p5.append(f"{tag} identifiers are not from the provider ({ids})")
            if e.get("status") == 200:
                if not (str(e.get("message_id", "")).startswith("msg_") and e.get("provider_request_id")
                        and str(e.get("model", "")).startswith("claude-") and (e.get("usage") or {}).get("output_tokens")):
                    p5.append(f"{tag} incomplete provider record")
            elif e.get("status") is None and not e.get("error_type"):
                p5.append(f"{tag} no status")
    n_ok = sum(e.get("status") == 200 for rows in ex.values() for e in rows)
    check("L5", "every exchange is a recorded live Anthropic call", p5,
          f"{n_ok} successful provider responses "
          f"(crew1 {sum(e.get('status') == 200 for e in ex['crew1'])}, crew2 {sum(e.get('status') == 200 for e in ex['crew2'])})")

    counts = {}
    for c, agents in (("crew1", CREW1_AGENTS), ("crew2", CREW2_AGENTS)):
        for a in agents:
            counts[a] = sum(1 for e in ex[c] if e.get("agent") == a and e.get("status") == 200)
    executed = kick["crew1"]["crewai_events"].get("agents_executed", []) + kick["crew2"]["crewai_events"].get(
        "agents_executed", [])
    check("L6", "each of the six agents got live responses",
          [f"{a}: 0 live responses" for a, n in counts.items() if not n]
          + ([f"CrewAI reported {len(executed)} agents executed"] if len(executed) != 6 else []),
          ", ".join(f"{a} {n}" for a, n in counts.items()))

    p7 = []
    for p in sorted(run.rglob("*")):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".html", ".csv") \
                and SCRIPTED_MARK in p.read_text(encoding="utf-8", errors="replace"):
            p7.append(f"scripted marker in {p.relative_to(run).as_posix()}")
    p7 += [f"{c} task output marked {o.get('source')!r}" for c in kick for o in kick[c].get("task_outputs", [])
           if o.get("source") != "llm"]
    check("L7", "no scripted response anywhere in the run", p7, "0 scripted markers")

    utc = m.get("timing", {}).get("utc", {})
    p8 = [f"missing UTC timestamp {k}" for k in ("crew1_started", "gate_decided", "crew2_started") if not utc.get(k)]
    if not p8:
        c1_end = max((e.get("finished_utc", "") for e in ex["crew1"]), default="")
        c2_start = min((e.get("started_utc", "9") for e in ex["crew2"]), default="9")
        if not c1_end or c1_end > utc["gate_decided"]:
            p8.append(f"a Crew 1 provider call finished at {c1_end}, after the gate decision {utc['gate_decided']}")
        if not utc["gate_decided"] <= utc["crew2_started"] <= c2_start:
            p8.append(f"gate {utc['gate_decided']}, Crew 2 start {utc['crew2_started']}, first Crew 2 call {c2_start}")
        try:
            events = [json.loads(line) for line in (run / "flow_events.jsonl").read_text(encoding="utf-8").splitlines()]
            gate_i = next(i for i, e in enumerate(events) if e.get("event") == "method_finished"
                          and e.get("method") == "contract_gate" and e.get("result") == "PASS")
            crew2_i = next(i for i, e in enumerate(events) if e.get("event") == "method_started"
                           and e.get("method") == "run_crew2")
            if not gate_i < crew2_i:
                p8.append("flow_events.jsonl: run_crew2 started before contract_gate finished with PASS")
        except (OSError, ValueError, StopIteration):
            p8.append("flow_events.jsonl lacks the contract_gate PASS or run_crew2 event")
        detail = (f"last Crew 1 call {c1_end} < gate PASS {utc['gate_decided']} <= Crew 2 start "
                  f"{utc['crew2_started']} <= first Crew 2 call {c2_start}")
    check("L8", "Crew 2 began only after the gate returned PASS", p8, detail if not p8 else "")

    allx = ex["crew1"] + ex["crew2"]
    total = {k: sum(int((e.get("usage") or {}).get(k) or 0) for e in allx) for k in USAGE_KEYS}
    p9 = ([f"provider_calls {llm.get('provider_calls')} != {len(allx)}"] if llm.get("provider_calls") != len(allx) else [])
    p9 += [f"usage {llm.get('usage')} != {total}"] if llm.get("usage") != total else []
    check("L9", "manifest totals equal the recorded exchanges", p9,
          f"{len(allx)} calls, {total['input_tokens'] + total['cache_creation_input_tokens'] + total['cache_read_input_tokens']}"
          f" input / {total['output_tokens']} output tokens, estimated USD {(llm.get('cost') or {}).get('estimated_usd')}")

    key = environ.get("ANTHROPIC_API_KEY") or ""
    p10 = []
    for p in sorted(run.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            if b"sk-ant-" in data or (len(key) >= 8 and key.encode() in data):
                p10.append(f"key material in {p.relative_to(run).as_posix()}")
    check("L10", "no API key in the evidence", p10, "no 'sk-ant-' string" + (", key value absent" if key else ""))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="the run folder, e.g. out/flow_runs/live-001")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    results = verify(args.run)
    ok = all(r["status"] == "PASS" for r in results)
    if args.json:
        print(json.dumps({"run": args.run.name, "verdict": "PASS" if ok else "FAIL", "checks": results}, indent=2))
    else:
        for r in results:
            print(f"{r['id']:<4} {r['status']:<4}  {r['check']}" + (f"\n            {r['detail']}" if r["detail"] else ""))
        print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} ({sum(r['status'] == 'PASS' for r in results)}/{len(results)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
