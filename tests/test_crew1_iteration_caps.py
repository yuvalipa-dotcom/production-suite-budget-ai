"""Crew 1 iteration caps, measured on a real crew kickoff against a mocked Anthropic API (offline).
ALL DATA IS SYNTHETIC.

In crewai 1.15.28 max_iter counts executor iterations, one model request each (a request may carry several
parallel tool calls). When it is reached CrewAI sends one more request without tools (forced final answer)
and the task completes with that text. Agent retries (max_retry_limit) would re-run a task after an
exception with a fresh counter; they are off.
"""

from __future__ import annotations

import json

import pytest

import crew1.crew as CC
from crew1.run import live_runner, main
from tests.crew1_mock_anthropic import (DUMMY_KEY, CrewScript, MockAnthropic, api_error, correct_plans,  # noqa: F401
                                        looping_plans, no_external_network, patch_for_mocked_kickoff,
                                        run_mocked_crew, tool_calls_by_agent)

pytestmark = pytest.mark.usefixtures("no_external_network")
OLD_MAX_ITER = {"ingestion_engineer": 12, "exploratory_analyst": 25, "contract_author": 15}


def test_the_caps_are_the_documented_values():
    assert {k: v["max_iter"] for k, v in CC.AGENT_LIMITS.items()} == {
        "ingestion_engineer": 12, "exploratory_analyst": 18, "contract_author": 10}
    assert CC.AGENT_MAX_RETRY_LIMIT == 0


def test_a_scripted_correct_run_passes_every_validator(monkeypatch, tmp_path):
    patch_for_mocked_kickoff(monkeypatch)
    holder = {}

    def runner(ws, sources, model, recorder):
        holder["script"] = script = CrewScript(ws, correct_plans())
        holder["mock"] = MockAnthropic(script).install(monkeypatch)
        return live_runner(ws, sources, model, recorder)        # the real build_crew + kickoff

    out = tmp_path / "run"
    assert main(["--out", str(out), "--live"], runner=runner, mode="offline-mocked-llm") == 0
    m = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "PASS" and all(v["status"] == "PASS" for v in m["validators"]), m["validators"]
    script = holder["script"]
    assert script.requests == {"ingestion_engineer": 5, "exploratory_analyst": 5, "contract_author": 4}
    assert script.forced_final == {k: 0 for k in script.forced_final}
    assert m["token_usage"]["successful_requests"] == len(holder["mock"].requests) == 14
    assert m["limits"]["agent_max_retry_limit"] == 0


def test_the_old_caps_reproduce_p3_live_003(monkeypatch):
    """Evidence: with max_iter 25 the EDA loop makes 22 report calls in 25 iterations plus 1 forced final
    request, as the p3-live-003 trace shows (39 EDA tool calls, 35 requests in the run)."""
    monkeypatch.setattr(CC, "AGENT_LIMITS", {k: {**v, "max_iter": OLD_MAX_ITER[k]} for k, v in CC.AGENT_LIMITS.items()})
    crew, script, mock, recorder, ws, error = run_mocked_crew(monkeypatch, looping_plans("exploratory_analyst"),
                                                             loop={"exploratory_analyst"})
    ws.cleanup()
    calls = tool_calls_by_agent(recorder)
    assert error is None and script.requests["exploratory_analyst"] == 25 + 1
    assert script.forced_final["exploratory_analyst"] == 1 and calls["exploratory_analyst"]["assemble_eda_report"] == 22
    assert sum(calls["exploratory_analyst"].values()) == 39 and len(mock.requests) == 35


def test_a_scripted_loop_is_stopped_by_the_new_caps(monkeypatch):
    crew, script, mock, recorder, ws, error = run_mocked_crew(
        monkeypatch, looping_plans("exploratory_analyst", "contract_author"),
        loop={"exploratory_analyst", "contract_author"})
    ws.cleanup()
    calls = tool_calls_by_agent(recorder)
    assert error is None                                                 # the crew finishes; no exception
    assert script.requests == {"ingestion_engineer": 5, "exploratory_analyst": 18 + 1, "contract_author": 10 + 1}
    assert script.forced_final == {"ingestion_engineer": 0, "exploratory_analyst": 1, "contract_author": 1}
    assert calls["exploratory_analyst"]["assemble_eda_report"] == 18 - 3     # list, analyses, charts came first
    assert calls["contract_author"]["write_insights"] == 10 - 2             # contract, gate came first
    assert [t.output.raw for t in crew.tasks][1:] == ["Forced final answer (exploratory_analyst).",
                                                       "Forced final answer (contract_author)."]
    assert len(mock.requests) == 5 + 19 + 11


def test_agent_retries_are_off(monkeypatch):
    """A server that always fails: the SDK retries the HTTP request (max_retries 2), the agent does not re-run
    the task; with the former max_retry_limit 1 this was 6 requests."""
    crew, script, mock, recorder, ws, error = run_mocked_crew(monkeypatch, correct_plans(),
                                                             respond=lambda body, n: api_error(500))
    ws.cleanup()
    assert error is not None and "500" in str(error)
    assert len(mock.requests) == 1 + CC.LLM_SETTINGS["max_retries"] == 3
