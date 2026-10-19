"""Builds Crew 1: three agents, three sequential tasks, deterministic tools only.

Safety settings (P3 instruction 2.2): telemetry and tracing off (``crew1/__init__.py``), no
memory, knowledge or embeddings, no delegation, explicit iteration / time / rate limits,
temperature 0. The model comes from the environment only (``CREW_MODEL``; the API key is read
by the provider SDK and never by this code).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from pathlib import Path

import yaml

from . import SAFETY_ENV
from .tools import TOOLSETS, ToolContext, build_tools
from .tools.workspace import REPO, Sources, Workspace

CONFIG = Path(__file__).resolve().parent / "config"
# Verified in the installed crewai 1.15.28: crewai/llms/constants.py ANTHROPIC_MODELS lists the Haiku
# ids "claude-haiku-4-5" and the dated snapshot "claude-haiku-4-5-20251001"; the dated id is pinned.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MODEL_RE = re.compile(r"claude-[a-z0-9][a-z0-9.\-]*")
LLM_SETTINGS = {"temperature": 0.0, "max_tokens": 8192, "timeout": 180.0, "max_retries": 2}
AGENT_LIMITS = {
    "ingestion_engineer": {"max_iter": 12, "max_execution_time": 600, "max_rpm": 20},
    "exploratory_analyst": {"max_iter": 25, "max_execution_time": 900, "max_rpm": 20},
    "contract_author": {"max_iter": 15, "max_execution_time": 600, "max_rpm": 20},
}
CREW_MAX_RPM = 20


class ConfigError(ValueError):
    pass


def resolve_model(environ=os.environ) -> str:
    """The Anthropic model id from CREW_MODEL (optional ``anthropic/`` prefix), else the verified default."""
    raw = (environ.get("CREW_MODEL") or DEFAULT_MODEL).strip()
    model = raw.split("/", 1)[1] if raw.lower().startswith("anthropic/") else raw
    if not MODEL_RE.fullmatch(model):
        raise ConfigError("CREW_MODEL must be an Anthropic model id such as " + DEFAULT_MODEL)
    return model


def make_llm(model: str):
    from crewai import LLM
    return LLM(model=f"anthropic/{model}", **LLM_SETTINGS)


def load_config() -> tuple[dict, dict]:
    agents = yaml.safe_load((CONFIG / "agents.yaml").read_text(encoding="utf-8"))
    tasks = yaml.safe_load((CONFIG / "tasks.yaml").read_text(encoding="utf-8"))
    return agents, tasks


# ---------------------------------------------------------------- trace with redaction

ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s\"'<>|,;)\]]*|\\\\[^\s\"'<>|,;)\]]+|"
                      r"(?<![\w.])/(?:home|Users|tmp|var|etc|root|mnt|opt|private)/[^\s\"'<>|,;)\]]*")
SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]+|\bsk-[A-Za-z0-9_\-]{20,}")
FIELD_LIMIT = 6000


def _path_forms(p: Path) -> list[str]:
    s = str(p)
    forms = {s, s.replace("\\", "/"), s.replace("/", "\\"), s.replace("\\", "\\\\")}
    return sorted(forms, key=len, reverse=True)


class Recorder:
    """Collects trace events; every string is redacted when recorded."""

    def __init__(self, workspace_root: Path | None = None, secret_env: str = "ANTHROPIC_API_KEY"):
        self.events: list[dict] = []
        self.t0 = time.monotonic()
        self.workspace_forms = _path_forms(workspace_root) if workspace_root else []
        self.repo_forms = _path_forms(REPO)
        self.secret_env = secret_env

    def redact(self, text: str) -> str:
        secret = os.environ.get(self.secret_env)
        if secret:
            text = text.replace(secret, "<redacted-secret>")
        text = SECRET.sub("<redacted-secret>", text)
        for form in self.repo_forms:
            text = re.sub(re.escape(form), "<redacted-path>", text, flags=re.I)
        for form in self.workspace_forms:
            text = re.sub(re.escape(form), "<workspace>", text, flags=re.I)
        return ABS_PATH.sub("<redacted-path>", text)

    def _clean(self, value):
        if isinstance(value, str):
            v = self.redact(value)
            return v if len(v) <= FIELD_LIMIT else v[:FIELD_LIMIT] + " ...[truncated]"
        if isinstance(value, dict):
            return {str(k): self._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(v) for v in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self._clean(str(value))

    def add(self, kind: str, /, **fields) -> None:
        # Positional-only, so a recorded field may itself be named "kind" (as agent steps are).
        self.events.append({"seq": len(self.events) + 1, "t": round(time.monotonic() - self.t0, 3), "event": kind,
                            **self._clean(fields)})

    def step(self, step) -> None:
        """CrewAI step callback: an agent action (thought / tool call) or a final answer."""
        fields = {k: getattr(step, k) for k in ("thought", "tool", "tool_input", "result", "output", "text")
                  if getattr(step, k, None) not in (None, "")}
        self.add("agent_step", kind=type(step).__name__, **fields)

    def task(self, output) -> None:
        self.add("task_completed", task=getattr(output, "name", None) or "", agent=getattr(output, "agent", ""),
                 output=getattr(output, "raw", ""))

    def jsonl(self) -> str:
        return "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in self.events)


# ---------------------------------------------------------------- the crew

def build_crew(ws: Workspace, sources: Sources, model: str, recorder: Recorder | None = None):
    """Agents and tasks from the YAML configuration; each agent gets only its own tools and its own LLM.

    One LLM instance per agent: CrewAI sums each agent's ``llm.get_token_usage_summary()``, so a shared
    instance is counted once per agent (p3-live-002 reported 3x its real usage)."""
    import warnings

    from crewai import Agent, Crew, Process, Task

    # Bound-method callbacks only disable CrewAI checkpointing, which this crew does not use.
    warnings.filterwarnings("ignore", message="method callbacks cannot be serialized")
    agents_cfg, tasks_cfg = load_config()
    ctx = ToolContext(ws, sources)
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
    # CrewAI persists the latest kickoff's task outputs for replay; keep that store inside the run workspace.
    from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage
    crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(db_path=str(ws.root / "tmp" / "kickoff.db"))
    return crew, ctx


# ---------------------------------------------------------------- process isolation helpers

ENV_ALLOW = ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "OS", "NUMBER_OF_PROCESSORS",
             "PROCESSOR_ARCHITECTURE", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "LOCALAPPDATA", "APPDATA",
             "PROGRAMDATA", "ALLUSERSPROFILE", "LANG", "LC_ALL", "TZ", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
             "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "ANTHROPIC_API_KEY", "CREW_MODEL", *SAFETY_ENV)


def _contains_path(value: str, root: Path) -> bool:
    v = value.lower().replace("\\", "/")
    return str(root).lower().replace("\\", "/") in v


def minimal_env(environ, ws_root: Path, repo: Path = REPO) -> dict[str, str]:
    """An allow-listed environment for the crew run: no variable may reveal the repository."""
    env = {k: v for k, v in environ.items() if k.upper() in ENV_ALLOW and not _contains_path(v, repo)}
    path = [p for p in environ.get("PATH", "").split(os.pathsep) if p and not _contains_path(p, repo)]
    env["PATH"] = os.pathsep.join(path)
    env["TEMP"] = env["TMP"] = str(ws_root / "tmp")
    env.update(SAFETY_ENV)
    leaking = [k for k, v in env.items() if _contains_path(v, repo)]
    if leaking:
        raise ConfigError("environment would reveal the repository path")
    return env


@contextlib.contextmanager
def isolated_process_state(ws_root: Path):
    """Run with the workspace as working directory and the minimal environment; restore both after."""
    saved_env, saved_cwd = dict(os.environ), os.getcwd()
    env = minimal_env(saved_env, ws_root)
    os.environ.clear()
    os.environ.update(env)
    os.chdir(ws_root)
    try:
        yield env
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
