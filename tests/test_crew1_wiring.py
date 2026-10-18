"""Crew 1 wiring and runtime safety (P3 2.9.1). Offline: no key, no network. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import os
import socket

import pytest

import crew1
from crew1.crew import AGENT_LIMITS, CREW_MAX_RPM, DEFAULT_MODEL, LLM_SETTINGS, ConfigError, Recorder, build_crew, \
    make_llm, resolve_model
from crew1.tools import TOOLSETS
from crew1.tools.workspace import Sources, stage_workspace
from tests.crew1_helpers import NetworkBlocked, no_network  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")

EXPECTED_TOOLS = {
    "Data Ingestion & Cleaning Engineer": {"load_raw_table", "detect_defects", "apply_policy_c", "eligibility_summary"},
    "Exploratory Data Analyst": {"list_analyses", "run_analysis", "render_chart", "assemble_eda_report"},
    "Data Contract & Insights Author": {"build_dataset_contract", "run_handoff_gate", "write_insights"},
}


@pytest.fixture(scope="module")
def built():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    ws = stage_workspace()
    try:
        crew, ctx = build_crew(ws, Sources.default(), make_llm(resolve_model({})), Recorder(ws.root))
        yield crew, ws
    finally:
        ws.cleanup()
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_the_network_blocker_is_active():
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.invalid", 443))


def test_three_agents_with_distinct_roles_goals_and_exact_tools(built):
    crew, _ = built
    assert len(crew.agents) == 3
    assert len({a.role for a in crew.agents}) == 3 and len({a.goal for a in crew.agents}) == 3
    assert len({a.backstory for a in crew.agents}) == 3
    assert {a.role: {t.name for t in a.tools} for a in crew.agents} == EXPECTED_TOOLS
    names = [t.name for a in crew.agents for t in a.tools]
    # exactly these 11 deterministic tools: no shared tool, no generic file/shell/code/directory tool
    assert len(names) == len(set(names)) == 11 and set(names) == set().union(*EXPECTED_TOOLS.values())


def test_three_sequential_tasks_with_contexts(built):
    from crewai import Process
    crew, _ = built
    assert crew.process == Process.sequential
    t1, t2, t3 = crew.tasks
    assert [t.agent.role for t in crew.tasks] == list(EXPECTED_TOOLS)
    assert not t1.context
    assert t2.context == [t1] and t3.context == [t1, t2]


def test_memory_knowledge_embeddings_delegation_and_code_execution_are_off(built):
    crew, _ = built
    assert crew.memory is False and not crew.knowledge_sources and crew.embedder is None
    assert crew.cache is False and not crew.planning and not crew.share_crew
    for a in crew.agents:
        assert a.allow_delegation is False and not a.memory and not a.knowledge_sources and a.embedder is None
        assert not a.allow_code_execution and not a.reasoning and not a.multimodal


def test_limits_and_temperature(built):
    crew, _ = built
    assert crew.max_rpm == CREW_MAX_RPM
    by_role = {a.role: a for a in crew.agents}
    for key, role in zip(TOOLSETS, EXPECTED_TOOLS):
        a, lim = by_role[role], AGENT_LIMITS[key]
        assert (a.max_iter, a.max_execution_time, a.max_rpm) == (lim["max_iter"], lim["max_execution_time"],
                                                                   lim["max_rpm"])
        assert a.llm.temperature == 0.0 == LLM_SETTINGS["temperature"]
        assert a.llm.model == DEFAULT_MODEL


def test_telemetry_and_tracing_are_disabled():
    for k, v in crew1.SAFETY_ENV.items():
        assert os.environ.get(k) == v
    from crewai.telemetry.telemetry import Telemetry
    assert Telemetry._is_telemetry_disabled()
    env = {"CREWAI_DISABLE_TELEMETRY": "false", "OTEL_SDK_DISABLED": "false"}
    crew1.apply_safety_env(env)                                   # an inherited value cannot re-enable it
    assert env["CREWAI_DISABLE_TELEMETRY"] == "true" and env["OTEL_SDK_DISABLED"] == "true"


def test_model_comes_from_the_environment_only():
    assert resolve_model({}) == DEFAULT_MODEL == "claude-haiku-4-5-20251001"
    assert resolve_model({"CREW_MODEL": "anthropic/claude-sonnet-4-6"}) == "claude-sonnet-4-6"
    for bad in ("gpt-4o", "claude haiku", "../x", "anthropic/"):
        with pytest.raises(ConfigError):
            resolve_model({"CREW_MODEL": bad})


def test_the_default_model_is_listed_by_the_installed_crewai():
    from crewai.llms.constants import ANTHROPIC_MODELS
    assert DEFAULT_MODEL in ANTHROPIC_MODELS


def test_llm_is_built_without_a_key_and_makes_no_call():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        llm = make_llm(DEFAULT_MODEL)
        assert type(llm).__name__ == "AnthropicCompletion" and llm._client is None
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
