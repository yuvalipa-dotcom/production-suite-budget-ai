"""Builds Crew 2: three agents, three sequential tasks, deterministic tools only.

The same safety settings as Crew 1 (docs/crew1.md, "Runtime safety"): telemetry and tracing
off, no memory, knowledge, embeddings, delegation, code execution, reasoning or planning,
caching off, temperature 0, one LLM instance per agent, explicit per-agent limits. The model
comes from ``CREW_MODEL``; the API key is read only by the provider SDK. Offline runs replace
the LLM with a scripted stand-in (tests/crew2_helpers.py); a live run needs owner approval.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import yaml

from crew1.crew import LLM_SETTINGS, ConfigError, Recorder, make_llm, minimal_env, resolve_model  # noqa: F401

from . import SAFETY_ENV
from .tools import TOOLSETS, ToolContext, build_tools

CONFIG = Path(__file__).resolve().parent / "config"
AGENT_LIMITS = {
    "feature_engineer": {"max_iter": 10, "max_execution_time": 900, "max_rpm": 20},
    "model_trainer": {"max_iter": 20, "max_execution_time": 7200, "max_rpm": 20},
    "evaluator": {"max_iter": 15, "max_execution_time": 1800, "max_rpm": 20},
}
CREW_MAX_RPM = 20


def load_config() -> tuple[dict, dict]:
    agents = yaml.safe_load((CONFIG / "agents.yaml").read_text(encoding="utf-8"))
    tasks = yaml.safe_load((CONFIG / "tasks.yaml").read_text(encoding="utf-8"))
    return agents, tasks


@contextlib.contextmanager
def crew2_storage():
    """Crew 2's CrewAI app-data folder, set only while the crew is built (restored afterwards)."""
    saved = os.environ.get("CREWAI_STORAGE_DIR")
    os.environ["CREWAI_STORAGE_DIR"] = SAFETY_ENV["CREWAI_STORAGE_DIR"]
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("CREWAI_STORAGE_DIR", None)
        else:
            os.environ["CREWAI_STORAGE_DIR"] = saved


def build_crew(ctx: ToolContext, model: str, recorder: Recorder | None = None):
    with crew2_storage():
        return _build_crew(ctx, model, recorder)


def _build_crew(ctx: ToolContext, model: str, recorder: Recorder | None = None):
    import warnings

    from crewai import Agent, Crew, Process, Task

    warnings.filterwarnings("ignore", message="method callbacks cannot be serialized")
    agents_cfg, tasks_cfg = load_config()
    tools = build_tools(ctx, recorder)
    step_cb = recorder.step if recorder else None
    agents = {}
    for key in TOOLSETS:
        cfg = agents_cfg[key]
        agents[key] = Agent(
            role=cfg["role"].strip(), goal=cfg["goal"].strip(), backstory=cfg["backstory"].strip(),
            tools=tools[key], llm=make_llm(model), allow_delegation=False, memory=False, knowledge_sources=None,
            embedder=None, allow_code_execution=False, reasoning=False, planning=False, multimodal=False,
            inject_date=False, cache=False, max_retry_limit=1, respect_context_window=True,
            step_callback=step_cb, verbose=False, **AGENT_LIMITS[key])
    tasks = {}
    for name, cfg in tasks_cfg.items():
        tasks[name] = Task(name=name, description=cfg["description"].strip(),
                           expected_output=cfg["expected_output"].strip(), agent=agents[cfg["agent"]],
                           context=[tasks[c] for c in cfg.get("context", [])] or None)
    crew = Crew(agents=list(agents.values()), tasks=list(tasks.values()), process=Process.sequential,
                memory=False, cache=False, planning=False, tracing=False, share_crew=False,
                knowledge_sources=None, embedder=None, max_rpm=CREW_MAX_RPM, verbose=False,
                output_log_file=None, stream=False, task_callback=recorder.task if recorder else None)
    from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage
    crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(db_path=str(ctx.ws.root / "tmp" / "kickoff.db"))
    return crew


@contextlib.contextmanager
def isolated_process_state(ws_root: Path):
    """Crew 1's minimal environment with Crew 2's safety settings; the workspace as working directory."""
    saved_env, saved_cwd = dict(os.environ), os.getcwd()
    env = minimal_env(saved_env, ws_root) | SAFETY_ENV
    os.environ.clear()
    os.environ.update(env)
    os.chdir(ws_root)
    try:
        yield env
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
