"""An offline stand-in for the Anthropic Messages API at the httpx transport layer. ALL DATA IS SYNTHETIC.

``httpx.HTTPTransport.handle_request`` is replaced, so every request the Anthropic SDK sends, through
CrewAI's default client or through CrewAI's interceptor transport, is answered here and never leaves
the process. Each request body is kept for inspection.

``CrewScript`` answers a real Crew 1 kickoff: per agent (identified by its tool names) it replays a
list of turns: a list of tool calls, or a final text. Inputs may be callables of the workspace, so
they can quote values the tools computed earlier in the run.
"""

from __future__ import annotations

import itertools
import json
import socket
from typing import Callable

import httpx
import pytest

import crew1.crew as CC
from crew1.tools import TOOLSETS
from crew1.tools import eda as E
from crew1.tools.workspace import RAW_TABLES, Sources, stage_workspace
from tests.crew1_helpers import NetworkBlocked, eda_sections, insights_md

AGENT_BY_TOOLS = {frozenset(names): key for key, names in TOOLSETS.items()}
DUMMY_KEY = "offline-" + "dummy-value-not-a-key"


@pytest.fixture
def no_external_network(monkeypatch):
    """Like ``no_network``, but loopback is allowed: CrewAI's kickoff starts an asyncio loop whose
    self-pipe connects over 127.0.0.1 on Windows. Every other connection and every name lookup fails."""
    real_connect = socket.socket.connect

    def loopback_only(sock, address):
        if isinstance(address, tuple) and address[0] in ("127.0.0.1", "::1"):
            return real_connect(sock, address)
        raise NetworkBlocked("network access attempted in an offline test")

    def refuse(*args, **kwargs):
        raise NetworkBlocked("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield
MODEL = "claude-haiku-4-5-20251001"
_ids = itertools.count(1)


def message(content: list[dict], stop_reason: str, usage=(100, 20)) -> dict:
    return {"id": f"msg_mock_{next(_ids)}", "type": "message", "role": "assistant", "model": MODEL,
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}


def tool_use(calls: list[tuple[str, dict]], usage=(100, 20)) -> dict:
    return message([{"type": "tool_use", "id": f"toolu_mock_{next(_ids)}", "name": n, "input": i}
                    for n, i in calls], "tool_use", usage)


def text(t: str, usage=(100, 20)) -> dict:
    return message([{"type": "text", "text": t}], "end_turn", usage)


def api_error(status: int = 500) -> tuple[int, dict]:
    return status, {"type": "error", "error": {"type": "api_error", "message": "mock server error"}}


class MockAnthropic:
    """``respond(body, n)`` returns a message dict (HTTP 200) or a ``(status, payload)`` tuple."""

    def __init__(self, respond: Callable[[dict, int], dict | tuple[int, dict]]):
        self.respond = respond
        self.requests: list[dict] = []

    def install(self, monkeypatch) -> "MockAnthropic":
        mock = self

        def handle_request(transport, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read() or b"{}")
            mock.requests.append(body)
            out = mock.respond(body, len(mock.requests))
            status, payload = out if isinstance(out, tuple) else (200, out)
            return httpx.Response(status, json=payload, request=request)

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
        return self

    def tool_definitions(self, name: str) -> list[dict]:
        """Every definition of tool ``name`` that was sent, in request order."""
        return [t for b in self.requests for t in b.get("tools") or [] if t.get("name") == name]


class CrewScript:
    """Scripted turns per agent key. ``loop`` repeats an agent's last turn once its script is used up."""

    def __init__(self, ws, plans: dict[str, list], loop: frozenset[str] = frozenset()):
        self.ws, self.plans, self.loop = ws, plans, loop
        self.pos = {k: 0 for k in plans}
        self.requests = {k: 0 for k in plans}
        self.forced_final = {k: 0 for k in plans}
        self.current: str | None = None

    def __call__(self, body: dict, n: int) -> dict:
        tools = body.get("tools")
        if not tools:                                  # CrewAI's forced final answer after max_iter
            self.requests[self.current] += 1
            self.forced_final[self.current] += 1
            return text(f"Forced final answer ({self.current}).")
        agent = AGENT_BY_TOOLS[frozenset(t["name"] for t in tools)]
        self.current = agent
        self.requests[agent] += 1
        steps, i = self.plans[agent], self.pos[agent]
        if i >= len(steps):
            if agent not in self.loop:
                return text(f"Final answer ({agent}).")
            i = len(steps) - 1
        self.pos[agent] = i + 1
        step = steps[i]
        if isinstance(step, str):
            return text(step)
        return tool_use([(name, arg(self.ws) if callable(arg) else arg) for name, arg in step])


def correct_plans() -> dict[str, list]:
    """A correct Crew 1 run: every agent batches its tool calls the way the live runs did."""
    return {
        "ingestion_engineer": [[("load_raw_table", {"name": f"raw.{t}"}) for t in RAW_TABLES],
                               [("detect_defects", {})], [("apply_policy_c", {})], [("eligibility_summary", {})],
                               "Cleaning narrative (scripted)."],
        "exploratory_analyst": [[("list_analyses", {})],
                                [("run_analysis", {"analysis_id": a}) for a in E.ANALYSES],
                                [("render_chart", {"chart_id": c}) for c in E.CHARTS],
                                [("assemble_eda_report", lambda ws: eda_sections(E.load_stats(ws)))],
                                "EDA summary (scripted)."],
        "contract_author": [[("build_dataset_contract", {"column_descriptions": [
                                {"column": "stage", "description": "Snapshot stage, S0 to S3."}]})],
                            [("run_handoff_gate", {})],
                            [("write_insights", lambda ws: {"markdown": insights_md(
                                E.load_stats(ws), json.loads(ws.read_output("out.dataset_contract")))})],
                            "Contract summary (scripted)."],
    }


def looping_plans(*agents: str) -> dict[str, list]:
    """The correct plans, except that each named agent repeats its authoring call forever (p3-live-002/003)."""
    plans = correct_plans()
    for a in agents:
        plans[a] = plans[a][:-1]                       # drop the final answer: the last turn repeats
    return plans


def patch_for_mocked_kickoff(monkeypatch) -> None:
    """A dummy key for the SDK and no HTTP-rate waiting (max_rpm only delays requests; it does not count them)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.setattr(CC, "CREW_MAX_RPM", None)
    monkeypatch.setattr(CC, "AGENT_LIMITS", {k: {**v, "max_rpm": None} for k, v in CC.AGENT_LIMITS.items()})


def run_mocked_crew(monkeypatch, plans: dict[str, list], loop=frozenset(), respond=None):
    """A real Crew 1 kickoff (build_crew + crew.kickoff) against the mock. Returns (crew, script, mock, recorder, ws, error);
    the caller cleans up ``ws``."""
    patch_for_mocked_kickoff(monkeypatch)
    ws = stage_workspace()
    recorder = CC.Recorder(ws.root)
    script = CrewScript(ws, plans, loop=frozenset(loop))
    mock = MockAnthropic(respond or script).install(monkeypatch)
    crew, _ = CC.build_crew(ws, Sources.default(), CC.DEFAULT_MODEL, recorder)
    error = None
    try:
        crew.kickoff()
    except Exception as e:  # noqa: BLE001  (the caller inspects it)
        error = e
    return crew, script, mock, recorder, ws, error


def tool_calls_by_agent(recorder) -> dict[str, dict[str, int]]:
    """Tool executions per agent, from the run trace (task_completed events mark the agent boundaries)."""
    order = iter(TOOLSETS)
    agent = next(order)
    out: dict[str, dict[str, int]] = {k: {} for k in TOOLSETS}
    for e in recorder.events:
        if e["event"] == "tool_call" and agent:
            out[agent][e["tool"]] = out[agent].get(e["tool"], 0) + 1
        elif e["event"] == "task_completed":
            agent = next(order, None)
    return out
