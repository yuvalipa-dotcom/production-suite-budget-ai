"""Command line for one Crew-1 run. ALL DATA IS SYNTHETIC.

    python -m crew1.run --out artifacts/crew1/<run_id> [--live]

A live run is refused unless --live is given AND ANTHROPIC_API_KEY is set (its value is never
read here, printed or logged). Exit codes: 0 PASS, 1 FAIL, 2 usage or configuration error.

The run stages a fresh workspace outside the repository, runs the crew there, copies the
artifacts to --out, writes the redacted trace and the run manifest, and runs the post-run
validators. On any failure the manifest status is FAILED and the artifacts are kept.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

from . import SAFETY_ENV
from .crew import AGENT_LIMITS, CONFIG, CREW_MAX_RPM, LLM_SETTINGS, ConfigError, Recorder, resolve_model
from .tools.cleaning import dumps
from .tools.workspace import RAW_TABLES, REPO, Sources, Workspace, stage_workspace

COPY = ("clean_data.csv", "cleaning_log.json", "stats.json", "eda_report.html", "dataset_contract.json",
        "gate_result.json", "insights.md")
PACKAGES = ("crewai", "anthropic", "matplotlib", "pandas", "numpy", "pydantic", "PyYAML")
PRICE_CATALOGUE = ("crewai", "llms/litellm_catalog/model_prices_and_context_window.json")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(sources: Sources) -> dict:
    out = {f"raw/{t}.csv": _sha(sources.raw_dir / f"{t}.csv") for t in RAW_TABLES}
    out["raw/manifest.json"] = _sha(sources.raw_dir / "manifest.json")
    out["docs/raw_data_dictionary.md"] = _sha(sources.dictionary)
    for c in ("column_roles.json", "dataset_contract.schema.json", "frozen_split.json"):
        out[f"contracts/{c}"] = _sha(sources.contracts_dir / c)
    for c in ("agents.yaml", "tasks.yaml"):
        out[f"crew1/config/{c}"] = _sha(CONFIG / c)
    return out


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def estimate_cost(model: str, usage: dict) -> dict:
    """Estimated USD cost from the price catalogue bundled with the installed crewai (an estimate only)."""
    try:
        dist = metadata.distribution(PRICE_CATALOGUE[0])
        prices = json.loads(Path(dist.locate_file(f"{PRICE_CATALOGUE[0]}/{PRICE_CATALOGUE[1]}")).read_text("utf-8"))
        p = prices[model]
    except (OSError, KeyError, ValueError, metadata.PackageNotFoundError):
        return {"estimated_usd": None, "source": "no price entry for this model in the bundled catalogue"}
    cached = usage.get("cached_prompt_tokens") or 0
    fresh = max((usage.get("prompt_tokens") or 0) - cached, 0)
    usd = (fresh * p.get("input_cost_per_token", 0) + cached * p.get("cache_read_input_token_cost",
                                                                     p.get("input_cost_per_token", 0))
           + (usage.get("completion_tokens") or 0) * p.get("output_cost_per_token", 0))
    return {"estimated_usd": round(usd, 4), "source": "crewai bundled price catalogue (LiteLLM format)",
            "input_cost_per_token": p.get("input_cost_per_token"), "output_cost_per_token": p.get("output_cost_per_token")}


def live_runner(ws: Workspace, sources: Sources, model: str, recorder: Recorder) -> dict:
    """Build and kick off the crew in the isolated process state. Makes the LLM calls."""
    from .crew import build_crew, isolated_process_state, make_llm

    crew, ctx = build_crew(ws, sources, make_llm(model), recorder)
    with isolated_process_state(ws.root):
        crew.kickoff()
    m = crew.usage_metrics
    usage = m.model_dump() if hasattr(m, "model_dump") else dict(m or {})
    return {"usage": usage, "analyses_requested": ctx.analyses_requested, "charts_rendered": ctx.charts_rendered}


def _usage(raw: dict) -> dict:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "successful_requests", "cached_prompt_tokens")
    return {k: int(raw.get(k) or 0) for k in keys}


def main(argv=None, runner=None, mode: str = "live") -> int:
    ap = argparse.ArgumentParser(prog="python -m crew1.run", description="Run Crew 1 (synthetic data only).")
    ap.add_argument("--out", required=True, help="output folder for this run, e.g. artifacts/crew1/<run_id>")
    ap.add_argument("--live", action="store_true", help="required: makes real LLM calls")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if not args.live:
        print("refused: a run calls the LLM; pass --live to confirm (no call was made)", file=sys.stderr)
        return 2
    import os
    secret_present = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if not secret_present:
        print("refused: ANTHROPIC_API_KEY is not set in the environment (no call was made)", file=sys.stderr)
        return 2
    try:
        model = resolve_model()
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("usage error: --out must be a new or empty folder", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    sources = Sources.default()
    started, t0 = _now(), time.monotonic()
    ws = stage_workspace(sources)
    recorder = Recorder(ws.root)
    recorder.add("run_started", mode=mode, model=model)
    error, info = None, {"usage": {}, "analyses_requested": [], "charts_rendered": []}
    try:
        info = (runner or live_runner)(ws, sources, model, recorder)
    except Exception as e:                                   # the run fails; artifacts are still kept
        error = f"{type(e).__name__}: {recorder.redact(str(e))[:500]}"
        recorder.add("run_error", error=error)
    for name in COPY:
        p = ws.root / "out" / name
        if p.is_file() and not p.is_symlink():
            (out / name).write_bytes(p.read_bytes())
    recorder.add("run_finished", error=error)
    (out / "trace.jsonl").write_text(recorder.jsonl(), encoding="utf-8", newline="\n")
    usage = _usage(info.get("usage") or {})
    tasks = [{"task": e["task"], "agent": e["agent"], "completed_at_s": e["t"]}
             for e in recorder.events if e["event"] == "task_completed"]
    from .validators import fresh_handoff
    _, fresh_log, _ = fresh_handoff(sources)
    manifest = {
        "run_id": out.name, "status": "RUNNING", "mode": mode, "synthetic": True,
        "model": {"id": model, "provider": "anthropic (native SDK through crewai)"},
        "llm_settings": LLM_SETTINGS, "limits": {"agents": AGENT_LIMITS, "crew_max_rpm": CREW_MAX_RPM},
        "safety_env": SAFETY_ENV, "package_versions": package_versions(), "input_sha256": input_hashes(sources),
        "policy_c_counts": {"scheduled": fresh_log["snapshots"]["scheduled"],
                            "eligible": fresh_log["snapshots"]["eligible"],
                            "exclusion_counts": fresh_log["exclusion_counts"]},
        "token_usage": usage, "cost": estimate_cost(model, usage),
        "timings": {"started_at": started, "finished_at": _now(),
                    "duration_seconds": round(time.monotonic() - t0, 1), "tasks": tasks or [{"task": "none"}]},
        "analyses_requested": info.get("analyses_requested", []),
        "charts_rendered": info.get("charts_rendered", []),
        "tool_calls": sum(1 for e in recorder.events if e["event"] == "tool_call"),
        "error": error,
        "artifacts_sha256": {p.name: _sha(p) for p in sorted(out.iterdir()) if p.is_file()},
    }
    from .validators import run_validators
    results = run_validators(out, sources, manifest, os.environ.get("ANTHROPIC_API_KEY"))
    passed = error is None and all(r["status"] == "PASS" for r in results)
    manifest |= {"status": "PASS" if passed else "FAILED", "validators": results}
    (out / "run_manifest.json").write_text(dumps(manifest), encoding="utf-8", newline="\n")
    ws.cleanup()
    print(json.dumps({"status": manifest["status"], "validators": {r["id"]: r["status"] for r in results},
                      "error": error, "token_usage": usage, "out": out.name}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
