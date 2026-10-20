"""Start Flow runs from the app, only on an explicit user action (P5-F). Standard library only.

The app runs in its own environment (``.venv-app``: Streamlit only). A Flow run is a separate process of the ML
environment (``PSBA_ML_PYTHON``, default ``.venv\\Scripts\\python.exe``):

    <ml python> -W ignore -m orchestrator.flow --out out/flow_runs/<run_id> --scenario <name> --n-jobs <n>

Nothing here runs at import time or while a page renders: ``start_flow`` is called only from a button handler.
The child process gets the user's environment without any variable that looks like a credential (the Flow
itself never needs one: its model responses are scripted). Its output goes to ``out/flow_runs/_logs/``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import evidence as E

LOG_DIR = E.LOCAL_RUNS_DIR / "_logs"
SCENARIOS = ("column_rename", "money_x100", "row_deletion", "contract_column_deletion", "label_leakage")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH", re.I)
_PROCESSES: dict[str, subprocess.Popen] = {}          # survives Streamlit reruns (module state)


def ml_python() -> Path | None:
    configured = os.environ.get("PSBA_ML_PYTHON")
    candidates = [Path(configured)] if configured else [E.REPO / ".venv" / "Scripts" / "python.exe",
                                                        E.REPO / ".venv" / "bin" / "python"]
    return next((p for p in candidates if p.is_file()), None)


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def new_run_id(scenario: str) -> str:
    kind = "run" if scenario == "none" else f"break-{scenario}"
    return f"app-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def start_flow(scenario: str = "none", n_jobs: int = 1) -> str:
    """Start one Flow run in the background. Returns its run id."""
    if scenario not in ("none", *SCENARIOS):
        raise ValueError("unknown scenario")
    py = ml_python()
    if py is None:
        raise RuntimeError("the ML environment was not found; set PSBA_ML_PYTHON (see docs/app.md)")
    if any(p.poll() is None for p in _PROCESSES.values()):
        raise RuntimeError("a run is already in progress")
    run_id = new_run_id(scenario)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / f"{run_id}.log", "w", encoding="utf-8")
    cmd = [str(py), "-W", "ignore", "-m", "orchestrator.flow", "--out", str(E.LOCAL_RUNS_DIR / run_id),
           "--scenario", scenario, "--n-jobs", str(max(1, int(n_jobs)))]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    _PROCESSES[run_id] = subprocess.Popen(cmd, cwd=str(E.REPO), env=child_env(), stdout=log,
                                          stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    _PROCESSES[run_id].started = time.monotonic()  # type: ignore[attr-defined]
    return run_id


def progress(run_id: str) -> dict:
    """The stage of a run, from its folder and process (no file is parsed beyond small JSON evidence)."""
    root = E.LOCAL_RUNS_DIR / run_id
    proc = _PROCESSES.get(run_id)
    running = proc is not None and proc.poll() is None
    if proc is not None and not running and not hasattr(proc, "finished"):
        proc.finished = time.monotonic()  # type: ignore[attr-defined]   (elapsed stops at the end of the run)
    decision = _gate_decision(root)
    stages = {"crew1": (root / "crew1" / "crew_kickoff.json").is_file(),
              "gate": decision is not None,
              "crew2_started": (root / "crew2").is_dir() or (running and decision == "PASS"),
              "finished": (root / "flow_manifest.json").is_file()}
    done = sum((stages["crew1"], stages["gate"], stages["finished"]))
    return {"run_id": run_id, "running": running, "exit_code": None if proc is None or running else proc.returncode,
            "elapsed_s": round(getattr(proc, "finished", time.monotonic()) - proc.started, 0)  # type: ignore
            if proc is not None else None,
            "stages": stages, "gate_decision": decision, "fraction": done / 3}


def _gate_decision(root: Path) -> str | None:
    p = root / "handoff" / "gate_report.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("decision") if p.is_file() else None
    except (OSError, ValueError):
        return None


def active_runs() -> list[str]:
    return [r for r, p in _PROCESSES.items() if p.poll() is None]


def known_runs() -> list[str]:
    return list(_PROCESSES)


def log_tail(run_id: str, lines: int = 12) -> str:
    p = LOG_DIR / f"{run_id}.log"
    if not E.RUN_ID_RE.fullmatch(run_id) or not p.is_file():
        return ""
    return E.redact("\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]))
