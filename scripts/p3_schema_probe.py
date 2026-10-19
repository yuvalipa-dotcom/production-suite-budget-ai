"""P3 live schema probe (owner-run, two Anthropic requests). ALL DATA IS SYNTHETIC.

Shows that the two Crew 1 authoring tools whose arguments were redesigned after p3-live-003
(assemble_eda_report, build_dataset_contract) receive non-empty arguments through the REAL
CrewAI 1.15.28 path, without running an agent:

* the same LLM class and construction as the crew (``crew1.crew.make_llm``), with retries off,
  a 1,500-token output limit and an explicit timeout;
* the same tool objects (``crew1.tools.build_tools``), converted by CrewAI's own
  ``convert_tools_to_openai_schema`` and ``AnthropicCompletion._convert_tools_for_interference``;
* ``AnthropicCompletion.call`` with exactly one tool and its function, which makes CrewAI force
  that tool (``tool_choice``) and execute it through ``BaseTool.run`` (argument validation).

The tools run only on a tiny synthetic workspace in a fresh temporary folder; the frozen data,
the P3 run folders and every existing output are never read or written. A CrewAI interceptor
counts the requests and refuses a third before it is sent. Only tool definitions, criteria,
usage, filtered errors and a cost estimate are saved (``probe_output/``, git-ignored); never a
request body and never the key.

Run (owner only, after approval, in a shell where ANTHROPIC_API_KEY is already set):
    $env:P3_SCHEMA_PROBE_APPROVAL = "APPROVE P3 SCHEMA PROBE"
    .\\.venv\\Scripts\\python.exe -m scripts.p3_schema_probe
Exit codes: 0 PASS, 1 FAIL, 2 refused (no approval or no key; nothing is sent).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import crew1  # noqa: F401  (applies the safety environment before CrewAI is imported)
from crew1.crew import DEFAULT_MODEL, SECRET, Recorder, make_llm
from crew1.tools import ToolContext, build_tools
from crew1.tools import eda as E
from crew1.tools.workspace import REPO, Sources, Workspace
from crew1.validators import eda_narrative
from gate import handoff_spec as H

APPROVAL_ENV = "P3_SCHEMA_PROBE_APPROVAL"
APPROVAL = "APPROVE P3 SCHEMA PROBE"
KEY_ENV = "ANTHROPIC_API_KEY"
PROBED = ("assemble_eda_report", "build_dataset_contract")
MAX_REQUESTS = 2
SETTINGS = {"temperature": 0.0, "max_tokens": 1500, "timeout": 60.0, "max_retries": 0}
EXPECTED_TOOLS = Path(__file__).with_name("p3_schema_probe_expected_tools.json")
OUT_ROOT = REPO / "probe_output" / "p3-schema-probe"
DESCRIBED = ("production_id", "stage", "as_of_date")
PRICE = {"model": "Claude Haiku 4.5", "input_usd_per_mtok": 1.00, "output_usd_per_mtok": 5.00,
         "source": "https://docs.claude.com/en/docs/about-claude/pricing (Claude Haiku 4.5: $1 per million "
                   "input tokens, $5 per million output tokens)",
         "note": "an estimate from the reported usage, not a spending cap"}

SYSTEM = "You are testing a reporting tool on synthetic placeholder data. Use only the facts given."
REPORT_PROMPT = ("Facts: the dataset is a synthetic placeholder with 3 rows; no analysis was run; nothing is "
                 "known about real productions. Call assemble_eda_report with one short plain sentence (at most "
                 "20 words) in every one of its nine section fields, using only these facts.")
CONTRACT_PROMPT = ("Call build_dataset_contract with exactly three column descriptions, one short plain sentence "
                   "each, for the columns production_id, stage and as_of_date. Facts: production_id identifies a "
                   "synthetic production; stage is the snapshot stage, S0 to S3; as_of_date is the snapshot date.")


class ProbeAbort(RuntimeError):
    pass


def _guard_class():
    import httpx
    from crewai.llms.hooks.base import BaseInterceptor

    class ProbeGuard(BaseInterceptor[httpx.Request, httpx.Response]):
        """Counts requests, records the tool definitions sent, and refuses a request beyond the limit."""

        def __init__(self, limit: int = MAX_REQUESTS):
            self.limit = limit
            self.sent: list[dict] = []
            self.statuses: list[int] = []
            self.refused = 0

        def on_outbound(self, message: httpx.Request) -> httpx.Request:
            if len(self.sent) >= self.limit:
                self.refused += 1
                raise ProbeAbort(f"request {len(self.sent) + 1} refused: the probe sends at most {self.limit}")
            body = json.loads(message.read() or b"{}")
            self.sent.append({"tools": body.get("tools"), "tool_choice": body.get("tool_choice"),
                              "model": body.get("model"), "max_tokens": body.get("max_tokens"),
                              "temperature": body.get("temperature"), "stream": body.get("stream")})
            return message

        def on_inbound(self, message: httpx.Response) -> httpx.Response:
            self.statuses.append(message.status_code)
            return message

    return ProbeGuard


def _placeholder(dtype: str, i: int) -> str:
    return {"integer": str(i + 1), "number": f"{i + 1}.5", "boolean": "False",
            "date": f"2024-01-0{i + 1}"}.get(dtype, f"SYN-PROBE-{i + 1}")


def stage_probe_workspace(root: Path) -> tuple[Workspace, Sources]:
    """A tiny synthetic workspace: 3 placeholder rows with the approved header, no frozen data."""
    for d in ("raw", "out/charts", "contracts"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "raw" / "manifest.json").write_text(json.dumps({"synthetic": True, "purpose": "P3 schema probe"}),
                                                encoding="utf-8")
    (root / "dictionary.md").write_text("Synthetic placeholder for the P3 schema probe.\n", encoding="utf-8")
    for name in ("column_roles.json", "dataset_contract.schema.json"):          # specifications, read-only
        shutil.copyfile(REPO / "contracts" / name, root / "contracts" / name)
    (root / "contracts" / "frozen_split.json").write_text(json.dumps({"synthetic_stub": True, "counts": {}}),
                                                          encoding="utf-8")
    with open(root / "out" / "clean_data.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(H.COLUMNS))
        for i in range(3):
            w.writerow([_placeholder(dtype, i) for dtype, _, _ in H.COLUMNS.values()])
    (root / "out" / "cleaning_log.json").write_text(json.dumps(
        {"synthetic": True, "exclusion_counts": {}, "issues": [], "excluded_payments": []}), encoding="utf-8")
    (root / "out" / "stats.json").write_text(E.dumps({"synthetic": True, "definitions": E.DEFINITIONS,
                                                      "analyses": {}}), encoding="utf-8")
    return Workspace(root), Sources(root / "raw", root / "dictionary.md", root / "contracts")


def _usage(llm) -> dict:
    u = llm.get_token_usage_summary()
    return {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
            "successful_requests": u.successful_requests}


def _call(llm, tool, prompt: str, redact) -> dict:
    """One forced call of one tool through CrewAI; the tool runs through BaseTool.run."""
    from crewai.utilities.agent_utils import convert_tools_to_openai_schema
    openai_tools, functions, _ = convert_tools_to_openai_schema([tool])
    name = openai_tools[0]["function"]["name"]
    rec: dict = {"tool": name}

    def recording(**kwargs):
        rec["arguments"] = kwargs
        try:
            out = functions[name](**kwargs)
        except Exception as e:  # noqa: BLE001  (recorded, then re-raised to CrewAI)
            rec["tool_error"] = redact(f"{type(e).__name__}: {e}")
            raise
        rec["tool_output"] = out
        return out

    before = _usage(llm)
    try:
        rec["llm_result"] = llm.call([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                                     tools=openai_tools, available_functions={name: recording})
    except Exception as e:  # noqa: BLE001
        rec["error"] = redact(f"{type(e).__name__}: {e}")
    after = _usage(llm)
    rec["usage"] = {k: after[k] - before[k] for k in after}
    return rec


def _criterion(ok: bool, detail) -> dict:
    return {"status": "PASS" if ok else "FAIL", "detail": detail}


def _report_criteria(rec: dict, ws: Workspace) -> dict:
    args = rec.get("arguments")
    sections = {s: (args or {}).get(s) for s in E.SECTIONS}
    filled = {s: isinstance(v, str) and bool(v.strip()) for s, v in sections.items()}
    out = rec.get("tool_output")
    valid, detail = False, "the tool did not run"
    if isinstance(out, str) and not out.startswith("ERROR") and ws.has_output("out.eda_report"):
        narrative, _ = eda_narrative(ws.read_output("out.eda_report").decode("utf-8"))
        valid = all(narrative.get(s) for s in E.SECTIONS)
        detail = {"eda_report_narrative_chars": {s: len(narrative.get(s, "")) for s in E.SECTIONS}}
    elif isinstance(out, str):
        detail = out[:300]
    return {
        "report_response_is_tool_call": _criterion(args is not None, rec.get("error") or "tool called"),
        "report_nine_sections_nonempty": _criterion(args is not None and set(args) == set(E.SECTIONS)
                                                    and all(filled.values()), filled),
        "report_tool_accepts_and_writes": _criterion(valid, detail),
    }


def _contract_criteria(rec: dict, ws: Workspace) -> dict:
    entries = (rec.get("arguments") or {}).get("column_descriptions")
    good = (isinstance(entries, list) and len(entries) == 3
            and all(isinstance(e, dict) and str(e.get("column", "")).strip() and str(e.get("description", "")).strip()
                    for e in entries)
            and len({e["column"] for e in entries}) == 3)
    out = rec.get("tool_output")
    valid, detail = False, "the tool did not run"
    if isinstance(out, str) and not out.startswith("ERROR") and ws.has_output("out.dataset_contract"):
        contract = json.loads(ws.read_output("out.dataset_contract"))
        described = {c["name"] for c in contract["columns"] if c.get("description")}
        valid = json.loads(out).get("described_columns") == 3 and len(described) == 3
        detail = {"described_columns": sorted(described)}
    elif isinstance(out, str):
        detail = out[:300]
    return {
        "contract_response_is_tool_call": _criterion(rec.get("arguments") is not None,
                                                     rec.get("error") or "tool called"),
        "contract_three_descriptions_nonempty": _criterion(good, entries if isinstance(entries, list) else None),
        "contract_tool_accepts_and_writes": _criterion(valid, detail),
    }


def _scan_for_key(folder: Path) -> list[str]:
    key = os.environ.get(KEY_ENV) or ""
    hits = []
    for p in folder.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            if (key and key in text) or SECRET.search(text):
                hits.append(p.name)
    return hits


def run_probe(out_root: Path = OUT_ROOT) -> tuple[int, Path | None]:
    if os.environ.get(APPROVAL_ENV) != APPROVAL:
        print(f"refused: {APPROVAL_ENV} is not set to the approval phrase (nothing was sent)", file=sys.stderr)
        return 2, None
    if not os.environ.get(KEY_ENV):
        print(f"refused: {KEY_ENV} is not set in the environment (nothing was sent)", file=sys.stderr)
        return 2, None

    started = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(tempfile.mkdtemp(prefix="p3-schema-probe-"))
    redact = Recorder(root).redact
    guard = _guard_class()()
    calls: list[dict] = []
    settings: dict = {}
    sent: dict = {}
    try:
        ws, sources = stage_probe_workspace(root)
        tools = {t.name: t for ts in build_tools(ToolContext(ws, sources)).values() for t in ts}
        llm = make_llm(DEFAULT_MODEL, **SETTINGS, interceptor=guard)
        sdk = llm._get_sync_client()
        settings = {"model": DEFAULT_MODEL, **SETTINGS, "sdk_max_retries": sdk.max_retries,
                    "sdk_timeout": str(sdk.timeout), "llm_class": type(llm).__name__,
                    "litellm_installed": importlib.util.find_spec("litellm") is not None}
        calls.append(_call(llm, tools["assemble_eda_report"], REPORT_PROMPT, redact))
        calls.append(_call(llm, tools["build_dataset_contract"], CONTRACT_PROMPT, redact))
        expected = json.loads(EXPECTED_TOOLS.read_text(encoding="utf-8"))
        sent = {name: [t for s in guard.sent for t in (s["tools"] or []) if t.get("name") == name] for name in PROBED}
        criteria = {
            **_report_criteria(calls[0], ws),
            **_contract_criteria(calls[1], ws),
            "exactly_two_requests": _criterion(len(guard.sent) == MAX_REQUESTS and guard.refused == 0,
                                               {"sent": len(guard.sent), "refused": guard.refused,
                                                "http_status": guard.statuses}),
            "tool_definitions_match_crew1_agents": _criterion(
                all(len(sent[n]) == 1 and sent[n][0] == expected[n] for n in PROBED),
                {n: ("identical" if len(sent[n]) == 1 and sent[n][0] == expected[n] else "DIFFERENT")
                 for n in PROBED}),
            "tool_use_forced_and_strict": _criterion(
                [s["tool_choice"] for s in guard.sent] == [{"type": "tool", "name": n} for n in PROBED]
                and all(t.get("strict") is True for ts in sent.values() for t in ts),
                [s["tool_choice"] for s in guard.sent]),
            "retries_off_and_limits_set": _criterion(
                llm.max_retries == 0 and sdk.max_retries == 0 and not settings["litellm_installed"]
                and all(s["temperature"] == 0.0 and s["max_tokens"] == SETTINGS["max_tokens"] for s in guard.sent),
                settings),
        }
    except Exception as e:  # noqa: BLE001  (setup failure: reported, never re-raised with paths)
        criteria = {"probe_completed": _criterion(False, redact(f"{type(e).__name__}: {e}"))}
    finally:
        shutil.rmtree(root, ignore_errors=True)

    usage = {k: sum(c.get("usage", {}).get(k, 0) for c in calls)
             for k in ("input_tokens", "output_tokens", "successful_requests")}
    usd = (usage["input_tokens"] * PRICE["input_usd_per_mtok"] + usage["output_tokens"] * PRICE["output_usd_per_mtok"]) / 1e6
    passed = bool(criteria) and all(c["status"] == "PASS" for c in criteria.values())
    result = {
        "probe": "P3 schema probe", "synthetic": True, "started_utc": started, "result": "PASS" if passed else "FAIL",
        "criteria": criteria, "settings": settings,
        "sent_tool_definitions": sent, "tool_choice_sent": [s["tool_choice"] for s in guard.sent],
        "calls": [{"tool": c.get("tool"), "arguments": c.get("arguments"), "usage": c.get("usage"),
                   "error": c.get("error"), "tool_error": c.get("tool_error")} for c in calls],
        "usage": usage, "cost_estimate": {**PRICE, "estimated_usd": round(usd, 6)},
    }
    out = out_root / started
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True),
                                           encoding="utf-8")
    hits = _scan_for_key(out)
    if hits:
        (out / "probe_result.json").write_text(json.dumps({"result": "FAIL", "reason": "a key pattern was found; "
                                                           "the result was withheld"}), encoding="utf-8")
        passed = False
    print(json.dumps({"result": "PASS" if passed else "FAIL", "key_scan": "clean" if not hits else "HIT (withheld)",
                      "failed_criteria": [k for k, c in criteria.items() if c["status"] != "PASS"],
                      "usage": usage, "estimated_usd": round(usd, 6),
                      "output": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else out.name}, indent=2))
    return (0 if passed else 1), out


def main(argv=None) -> int:
    argparse.ArgumentParser(prog="python -m scripts.p3_schema_probe",
                            description=__doc__.splitlines()[0]).parse_args(argv)
    return run_probe()[0]


if __name__ == "__main__":
    sys.exit(main())
