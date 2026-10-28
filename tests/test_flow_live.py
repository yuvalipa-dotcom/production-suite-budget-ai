"""The Flow's live-LLM mode, tested offline: no key, no network, no inference. ALL DATA IS SYNTHETIC.

The live path (``crew_common.live.RecordingTransport``) is exercised end to end with an in-process stand-in for
the Anthropic API placed BEHIND the recording transport, exactly where the network call would be. The stand-in
answers with the P5 scripted turns, stripped of the scripted marker, under clearly offline identifiers
(``msg_offline_test_*``, ``req_offline_test_*``). So:

* a live Flow run records every provider exchange (UTC times, status, request-id, message id, model, usage) for
  all six agents, writes the 8 outputs, and the evidence verifier passes every check except provenance (L5),
  which correctly rejects the offline identifiers;
* a live run is refused without ANTHROPIC_API_KEY (exit 2, no folder, no call);
* an exhausted request budget fails the run closed (FAILED_RUNTIME; Crew 2 never starts);
* a contract break in a live run stops at the gate: Crew 2 never starts and makes no provider call;
* the key never appears in the evidence, and a scripted run never passes the verifier.
"""

from __future__ import annotations

import itertools
import json
import socket

import httpx
import pytest

from budget_flow import crew2_kickoff as K2
from crew_common import live as L
from crew_common import scripted as S
from flow_crews import crew1_kickoff as K1
from flow_crews import crew1_script as SCRIPT
from orchestrator import flow as F
from tests.conftest import REPO
from orchestrator import verify_live_run as V

TEST_KEY = "offline-" + "live-mode-test-value-0123456789"
CANONICAL = REPO / "flow_evidence" / "runs" / "p5e-run-001"


class _Blocked(RuntimeError):
    pass


def _offline(mp):
    """The test process may not reach the network (loopback only: CrewAI's asyncio self-pipe on Windows)."""
    real_connect = socket.socket.connect

    def loopback_only(sock, address):
        if isinstance(address, tuple) and address and address[0] in ("127.0.0.1", "::1"):
            return real_connect(sock, address)
        raise _Blocked("network access attempted in an offline test")

    def refuse(*a, **k):
        raise _Blocked("network access attempted in an offline test")

    mp.setattr(socket.socket, "connect", loopback_only)
    mp.setattr(socket, "create_connection", refuse)
    mp.setattr(socket, "getaddrinfo", refuse)


def _fake_provider(mp, seen: dict):
    """Stand in for api.anthropic.com behind the recording transport; remember each request's headers."""
    plans = {**SCRIPT.plans(), **K2.PLANS}
    by_tools = {**K1.AGENT_BY_TOOLS, **K2.AGENT_BY_TOOLS}
    scripts, ids = {}, itertools.count(1)

    def upstream(transport, request: httpx.Request) -> httpx.Response:
        seen.setdefault("headers", []).append(dict(request.headers))
        st = scripts.setdefault(id(transport), S.ScriptedTransport(plans, by_tools, transport.context))
        msg = st.respond(json.loads(request.content))
        for block in msg["content"]:
            if block["type"] == "text":
                block["text"] = block["text"].replace(S.SCRIPTED_MARK + " ", "")
        n = next(ids)
        msg |= {"id": f"msg_offline_test_{n}", "model": K2.DEFAULT_MODEL,
                "usage": {"input_tokens": 1000 + n, "output_tokens": 40}}
        return httpx.Response(200, json=msg, headers={"request-id": f"req_offline_test_{n}"})

    mp.setattr(L.RecordingTransport, "_test_upstream", upstream)
    mp.setattr(K2, "LIVE_MAX_RPM", None)         # the fake answers instantly; no need to wait for a minute window


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("flow_live") / "run"
    seen, lines = {}, []
    with pytest.MonkeyPatch.context() as mp:
        _offline(mp)
        mp.setenv("ANTHROPIC_API_KEY", TEST_KEY)
        _fake_provider(mp, seen)
        manifest = F.run_flow(out, n_jobs=5, llm_mode="live", log=lines.append)
        results = V.verify(out, environ={"ANTHROPIC_API_KEY": TEST_KEY})
    return manifest, out, seen, lines, results


# ---------------------------------------------------------------- the live run end to end (offline provider)

def test_live_flow_passes_with_every_step_and_all_eight_outputs(live_run):
    m, out, _, _, _ = live_run
    assert m["status"] == "PASS" and m["exit_code"] == 0 and m["error"] is None, m["error"]
    assert m["flow"]["routes"] == {"contract_gate": "PASS", "verify_outputs": "COMPLETE"}
    assert sorted(m["required_outputs_sha256"]) == sorted(V.REQUIRED)
    assert "live Anthropic inference" in m["disclosure"] and "scripted" not in m["disclosure"]


def test_live_flow_records_provider_calls_for_all_six_agents(live_run):
    m, out, _, _, _ = live_run
    llm = m["llm"]
    assert llm["mode"] == "live" and llm["scripted_responses"] == 0
    assert llm["models_reported_by_provider"] == [K2.DEFAULT_MODEL]
    agents = {**llm["by_crew"]["crew1"]["successful_responses_by_agent"],
              **llm["by_crew"]["crew2"]["successful_responses_by_agent"]}
    assert sorted(agents) == sorted(V.CREW1_AGENTS + V.CREW2_AGENTS) and all(agents.values())
    assert llm["budget"]["used"]["requests"] == llm["provider_calls"] > 0 and llm["budget"]["refused"] == []
    assert llm["cost"]["estimated_usd"] > 0
    for crew in ("crew1", "crew2"):
        k = json.loads((out / crew / "crew_kickoff.json").read_text(encoding="utf-8"))
        assert k["llm_mode"] == "live" and all(o["source"] == "llm" for o in k["task_outputs"])
        for e in k["llm_transport"]["exchanges"]:
            assert e["source"] == "live" and e["status"] == 200 and e["started_utc"] <= e["finished_utc"]
            assert e["provider_request_id"].startswith("req_") and e["message_id"].startswith("msg_")


def test_live_flow_progress_shows_gate_pass_before_crew2(live_run):
    _, _, _, lines, _ = live_run
    text = "\n".join(lines)
    gate = text.index("contract gate (deterministic Python) decision: PASS")
    assert text.index("Crew 1 completed tasks") < gate < text.index("Crew 2 kickoff starting (the gate returned PASS)")
    assert "[LLM   1] crew1/ingestion_engineer: HTTP 200" in text and "crew2/evaluation_author: HTTP 200" in text


def test_verifier_passes_everything_but_rejects_the_offline_identifiers(live_run):
    *_, results = live_run
    failed = {r["id"]: r["detail"] for r in results if r["status"] == "FAIL"}
    assert list(failed) == ["L5"] and "msg_offline_test_" in failed["L5"], results


def test_the_key_never_reaches_the_evidence_or_the_console(live_run):
    _, out, seen, lines, results = live_run
    assert any(h.get("x-api-key") == TEST_KEY for h in seen["headers"])     # the SDK did send it, upstream only
    for p in out.rglob("*"):
        if p.is_file():
            assert TEST_KEY.encode() not in p.read_bytes(), p
    assert TEST_KEY not in "\n".join(lines)
    assert next(r for r in results if r["id"] == "L10")["status"] == "PASS"


# ---------------------------------------------------------------- refusals and fail-closed behaviour

def test_a_live_run_is_refused_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(F.LiveConfigError):
        F.run_flow(tmp_path / "a", llm_mode="live")
    assert F.main(["--out", str(tmp_path / "b"), "--llm", "live"]) == 2
    assert "refused" in capsys.readouterr().err
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_an_exhausted_budget_fails_closed_before_crew2(tmp_path, monkeypatch):
    _offline(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_KEY)
    _fake_provider(monkeypatch, {})
    m = F.run_flow(tmp_path / "run", llm_mode="live", budget=L.LiveBudget(3, 10**9, 10**9))
    assert m["status"] == "FAILED_RUNTIME" and m["error"].startswith("crew1")
    assert m["crew2"]["kickoff_invoked"] is False and not (tmp_path / "run" / "crew2").exists()
    assert m["llm"]["budget"]["used"]["requests"] == 3 and m["llm"]["budget"]["refused"]


def test_a_live_contract_break_stops_at_the_gate_without_crew2_calls(tmp_path, monkeypatch):
    _offline(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_KEY)
    _fake_provider(monkeypatch, {})
    m = F.run_flow(tmp_path / "run", scenario="money_x100", llm_mode="live")
    assert m["status"] == "STOPPED_CONTRACT" and m["gate"]["violation_codes"] == ["UNIT_SCALE_ANOMALY"]
    assert m["crew2"]["kickoff_invoked"] is False and "crew2" not in m["llm"]["by_crew"]
    assert "crew2_started" not in m["timing"]["utc"]


def test_the_scripted_canonical_run_never_passes_the_verifier():
    failed = {r["id"] for r in V.verify(CANONICAL, environ={}) if r["status"] == "FAIL"}
    assert {"L1", "L5", "L7"} <= failed


def test_scripted_mode_is_still_the_default():
    import inspect
    for fn in (F.run_flow, K1.run_crew1_kickoff, K2.run_crew2_kickoff):
        assert inspect.signature(fn).parameters["llm_mode"].default == "scripted"


# ---------------------------------------------------------------- the recording transport on its own

def _request(host="api.anthropic.com", tools=("load_raw_table",)):
    body = {"model": K2.DEFAULT_MODEL, "max_tokens": 10, "messages": [{"role": "user", "content": "x"}],
            "tools": [{"name": t} for t in tools]}
    return httpx.Request("POST", f"https://{host}/v1/messages", json=body, headers={"x-api-key": TEST_KEY})


def test_recording_transport_records_metadata_only_and_returns_the_response(monkeypatch):
    def upstream(transport, request):
        return httpx.Response(200, headers={"request-id": "req_offline_test_x"},
                              json={"id": "msg_offline_test_x", "model": K2.DEFAULT_MODEL, "stop_reason": "tool_use",
                                    "usage": {"input_tokens": 7, "output_tokens": 3},
                                    "content": [{"type": "tool_use", "name": "detect_defects", "id": "t", "input": {}}]})

    monkeypatch.setattr(L.RecordingTransport, "_test_upstream", upstream)
    budget = L.LiveBudget(5, 100, 100)
    t = L.RecordingTransport(K1.AGENT_BY_TOOLS, budget, "crew1")
    with t.installed():
        r = httpx.Client().send(_request(tools=("load_raw_table", "detect_defects", "apply_policy_c",
                                                "eligibility_summary")))
    assert r.status_code == 200 and r.json()["id"] == "msg_offline_test_x"
    e = t.exchanges[0]
    assert (e["agent"], e["status"], e["provider_request_id"], e["tool_calls"]) == \
        ("ingestion_engineer", 200, "req_offline_test_x", ["detect_defects"])
    assert TEST_KEY not in json.dumps(t.exchanges) and "headers" not in json.dumps(t.exchanges)
    assert budget.used == {"requests": 1, "input_tokens": 7, "output_tokens": 3}


def test_recording_transport_refuses_other_hosts_and_an_exhausted_budget(monkeypatch):
    monkeypatch.setattr(L.RecordingTransport, "_test_upstream", lambda t, r: pytest.fail("must not be sent"))
    t = L.RecordingTransport(K1.AGENT_BY_TOOLS, L.LiveBudget(0, 1, 1), "crew1")
    with t.installed():
        with pytest.raises(L.LiveCallRefused):
            httpx.Client().send(_request(host="example.invalid"))
        with pytest.raises(L.LiveCallRefused):
            httpx.Client().send(_request())
    assert t.network_blocked == ["host example.invalid"] and "refused" in t.exchanges[-1]


def test_cost_estimate_equals_the_p3_estimate():
    from crew1.run import estimate_cost as p3_estimate
    usage = {"input_tokens": 125419, "output_tokens": 13444}
    p3 = p3_estimate(K2.DEFAULT_MODEL, {"prompt_tokens": 125419, "completion_tokens": 13444})
    assert L.estimate_cost(K2.DEFAULT_MODEL, usage)["estimated_usd"] == p3["estimated_usd"] == 0.1926


# ---------------------------------------------------------------- the committed live run (flow_evidence/live/)

LIVE = REPO / "flow_evidence" / "live" / "p5-live-001"


def test_the_committed_live_run_passes_every_check():
    if not LIVE.is_dir():
        pytest.skip("the live Flow evidence is not in this checkout")
    results = V.verify(LIVE, environ={})
    assert all(r["status"] == "PASS" for r in results), [r for r in results if r["status"] != "PASS"]


def test_the_committed_live_run_is_complete_unchanged_and_selects_the_approved_model():
    import hashlib
    if not LIVE.is_dir():
        pytest.skip("the live Flow evidence is not in this checkout")
    idx = json.loads((LIVE / "evidence.json").read_text(encoding="utf-8"))
    on_disk = {p.relative_to(LIVE).as_posix() for p in LIVE.rglob("*") if p.is_file()} - {"evidence.json"}
    assert set(idx["files_sha256"]) == on_disk and len(on_disk) == 23
    assert all(hashlib.sha256((LIVE / r).read_bytes()).hexdigest() == h for r, h in idx["files_sha256"].items())
    m = json.loads((LIVE / "flow_manifest.json").read_text(encoding="utf-8"))
    canonical = json.loads((CANONICAL / "flow_manifest.json").read_text(encoding="utf-8"))
    assert m["crew2"]["selected_model"] == canonical["crew2"]["selected_model"]
    assert m["crew2"]["reproducibility"]["ok"] and m["crew2"]["test_vault"]["opens"] == 0
    for rel in ("crew1/clean_data.csv", "crew2/features.csv", "crew2/model.joblib", "crew2/model_card.md"):
        assert m["required_outputs_sha256"][rel] == canonical["required_outputs_sha256"][rel], rel
