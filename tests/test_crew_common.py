"""P5-D: the neutral shared crew package. Offline. ALL DATA IS SYNTHETIC.

* ``crew_common`` imports neither Crew 1 nor Crew 2;
* its copies of the Crew 1 workspace, recorder and LLM construction behave exactly like the frozen
  originals (which the historical P3/P4 entry points keep using);
* the scripted transport answers in-process, marks every text, blocks the network and restores
  everything it replaced.
"""

from __future__ import annotations

import ast
import os
import socket
import subprocess
import sys

import httpx
import pytest

from crew_common import scripted as S
from crew_common import workspace as W
from crew_common.llm import LLM_SETTINGS, make_llm
from crew_common.recorder import Recorder
from tests.conftest import REPO

# Absolute-path samples for the redaction parity test, built from parts so the source holds no literal path.
WIN_SAMPLE = "\\".join(["C:", "Users", "someone", "file.txt"])
HOME_SAMPLE = "/".join(["", "home", "u", "x"])
PRESET = "pre" + "set-value"
NAMES = ["raw.productions", "out.clean_data", "  out.stats ", "", None, 7, "C:\\x\\y.csv", "/etc/passwd",
         "\\\\server\\share", "\\\\?\\C:\\x", "//./pipe", "a/../b", "raw..productions", "PROGRA~1.x", "out.x:ads",
         "raw/productions", "raw\\productions", "Raw.Productions", "raw.unknown", "raw.a.b.c", "x.y", "out.chart.cv"]


def test_crew_common_imports_neither_crew_statically():
    for path in sorted((REPO / "crew_common").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] in ("crew1", "crew2", "budget_flow") for n in names), (path.name, names)


def test_importing_crew_common_and_the_flow_crew2_loads_no_crew1_module():
    code = ("import sys; sys.path.insert(0, %r)\nimport crew_common.workspace, crew_common.recorder, "
            "crew_common.scripted, crew_common.tools, crew_common.kickoff, budget_flow.crew2_kickoff\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'crew1' or m in "
            "('crew2.crew', 'crew2.tools', 'crew2.validators', 'crew2.offline', 'crew2.run')))" % str(REPO))
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip() == "[]"


# ---------------------------------------------------------------- parity with the frozen Crew 1 originals

def test_name_classification_equals_crew1():
    from crew1.tools import workspace as W1
    allowed = {"raw.productions", "out.clean_data", "out.stats", "out.chart.cv"}
    assert [W.classify_name(n, allowed) for n in NAMES] == [W1.classify_name(n, allowed) for n in NAMES]
    assert W.NAME_RE.pattern == W1.NAME_RE.pattern and W.REPARSE == W1.REPARSE


def test_workspace_resolution_and_access_equal_crew1(tmp_path):
    from crew1.tools import workspace as W1
    reads, outs = {"in.data": "in/data.csv"}, {"out.result": "out/result.txt"}
    for d in ("in", "out"):
        (tmp_path / d).mkdir()
    (tmp_path / "in" / "data.csv").write_text("a,b\n1,2\n")
    ours, theirs = W.Workspace(tmp_path, reads, outs), W1.Workspace(tmp_path, reads, outs)
    assert ours.root == theirs.root and ours.resolve("in.data") == theirs.resolve("in.data")
    assert ours.read_bytes("in.data") == theirs.read_bytes("in.data")
    def code(ws, name):
        try:
            ws.resolve(name)
        except (W.IsolationError, W1.IsolationError) as e:
            return e.code
        return None

    for bad in ("in/data.csv", "..", "out.result", "C:\\in\\data.csv"):
        assert code(ours, bad) == code(theirs, bad) is not None
    ours.write("out.result", "x")
    assert theirs.read_output("out.result") == ours.read_output("out.result") == b"x"


def test_staging_refuses_the_repository_and_links(tmp_path):
    with pytest.raises(W.IsolationError, match="workspace_inside_repository"):
        W.stage("t_", {}, {}, {}, parent=REPO / "out")
    ws = W.stage("t_", {"in/a.csv": REPO / "LICENSE"}, {"in.a": "in/a.csv"}, {})
    try:
        assert ws.read_bytes("in.a") == (REPO / "LICENSE").read_bytes()
        assert not str(ws.root).lower().startswith(str(REPO).lower())
    finally:
        ws.cleanup()


def test_recorder_redaction_equals_crew1(tmp_path, monkeypatch):
    from crew1.crew import Recorder as R1
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-" + "x" * 30)
    samples = [f"repo {REPO} and {str(REPO).replace(chr(92), '/')}", f"ws {tmp_path}\\out\\file.csv",
               "key sk-ant-api03-" + "x" * 30, WIN_SAMPLE, HOME_SAMPLE, "plain text",
               "x" * 7000, {"nested": [f"{tmp_path}/a", 1, None, True]}]
    a, b = Recorder(tmp_path), R1(tmp_path)
    for s in samples:
        a.add("e", value=s)
        b.add("e", value=s)
    strip = lambda evs: [{k: v for k, v in e.items() if k != "t"} for e in evs]   # noqa: E731
    assert strip(a.events) == strip(b.events)


def test_llm_construction_equals_crew1():
    import crew1.crew as CC
    assert LLM_SETTINGS == CC.LLM_SETTINGS
    ours, theirs = make_llm(CC.DEFAULT_MODEL), CC.make_llm(CC.DEFAULT_MODEL)
    assert type(ours) is type(theirs) and ours.model == theirs.model
    assert (ours.temperature, ours.max_tokens) == (theirs.temperature, theirs.max_tokens)


# ---------------------------------------------------------------- the scripted transport

def _transport():
    plans = {"a": [[("t1", {"x": 1})], lambda ctx: f"done {ctx}"]}
    return S.ScriptedTransport(plans, {frozenset({"t1"}): "a"}, context="ctx")


def test_scripted_responses_follow_the_plan_and_are_marked():
    t = _transport()
    r1 = t.respond({"tools": [{"name": "t1"}], "messages": [{}]})
    r2 = t.respond({"tools": [{"name": "t1"}], "messages": [{}, {}, {}]})
    r3 = t.respond({"messages": []})                         # a forced final answer (no tools offered)
    assert r1["content"][0] == {"type": "tool_use", "id": r1["content"][0]["id"], "name": "t1", "input": {"x": 1}}
    assert r2["content"][0]["text"] == f"{S.SCRIPTED_MARK} done ctx" and r3["content"][0]["text"].startswith(S.SCRIPTED_MARK)
    assert [e["source"] for e in t.exchanges] == ["scripted"] * 3 and t.forced_final == {"a": 1}


def test_the_transport_answers_in_process_blocks_the_network_and_restores_everything(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", PRESET)
    saved = (httpx.HTTPTransport.handle_request, socket.socket.connect, socket.getaddrinfo)
    t = _transport()
    with t.installed():
        assert os.environ["ANTHROPIC_API_KEY"] == S.DUMMY_KEY
        with httpx.Client() as c:
            r = c.post("https://api.anthropic.com/v1/messages", json={"tools": [{"name": "t1"}], "messages": []})
        assert r.json()["content"][0]["name"] == "t1"
        with pytest.raises(S.NetworkBlocked):
            socket.getaddrinfo("example.com", 443)
        s = socket.socket()
        try:
            with pytest.raises(S.NetworkBlocked):
                s.connect(("192.0.2.1", 443))
        finally:
            s.close()
    assert (httpx.HTTPTransport.handle_request, socket.socket.connect, socket.getaddrinfo) == saved
    assert os.environ["ANTHROPIC_API_KEY"] == PRESET
    assert len(t.network_blocked) == 2 and t.exchanges[0]["source"] == "scripted"
