"""Command line for one Crew-2 run. ALL DATA IS SYNTHETIC.

    python -m crew2.run --out out/crew2/<run_id> --offline     # scripted stand-in for the LLM, no call
    python -m crew2.run --out out/crew2/<run_id> --live        # real LLM calls: needs owner approval

A live run is refused unless --live is given AND ANTHROPIC_API_KEY is set (its value is never
read here, printed or logged). Exit codes: 0 PASS, 1 FAIL, 2 usage or configuration error.

The run stages a fresh workspace outside the repository, runs the crew there, copies the
artifacts to --out, writes the redacted trace and the run manifest, and runs the post-run
validators (crew2/validators.py). On any failure the manifest status is FAILED and the
artifacts are kept.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

from . import SAFETY_ENV
from .crew import AGENT_LIMITS, CONFIG, CREW_MAX_RPM, LLM_SETTINGS, ConfigError, Recorder, resolve_model
from .tools import ToolContext, dumps
from .tools.workspace import CHART_IDS, CONTRACTS_DIR, RAW_DIR, RAW_TABLES, stage_workspace

COPY = ("metrics.json", "model.joblib", "model_card.md", "demo_predictions.csv", "cv_predictions.csv",
        "leakage_report.json", "feature_availability.json", *(f"charts/{c}.png" for c in CHART_IDS))
PACKAGES = ("crewai", "anthropic", "scikit-learn", "scipy", "joblib", "threadpoolctl", "matplotlib", "pandas",
            "numpy", "pydantic", "PyYAML")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> dict:
    out = {f"raw/{t}.csv": _sha(RAW_DIR / f"{t}.csv") for t in RAW_TABLES}
    out["raw/manifest.json"] = _sha(RAW_DIR / "manifest.json")
    for c in ("column_roles.json", "dataset_contract.schema.json", "frozen_split.json"):
        out[f"contracts/{c}"] = _sha(CONTRACTS_DIR / c)
    for c in ("agents.yaml", "tasks.yaml"):
        out[f"crew2/config/{c}"] = _sha(CONFIG / c)
    out["docs/model_selection_rule.md"] = _sha(CONTRACTS_DIR.parent / "docs" / "model_selection_rule.md")
    return out


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def live_runner(ctx: ToolContext, model: str, recorder: Recorder) -> dict:
    from .crew import build_crew, isolated_process_state

    crew = build_crew(ctx, model, recorder)
    with isolated_process_state(ctx.ws.root):
        crew.kickoff()
    m = crew.usage_metrics
    return {"usage": m.model_dump() if hasattr(m, "model_dump") else dict(m or {})}


def _usage(raw: dict) -> dict:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens", "successful_requests", "cached_prompt_tokens")
    return {k: int(raw.get(k) or 0) for k in keys}


def main(argv=None, runner=None, n_jobs: int = 1, recompute=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m crew2.run", description="Run Crew 2 (synthetic data only).")
    ap.add_argument("--out", required=True, help="output folder for this run, e.g. out/crew2/<run_id>")
    mode_group = ap.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true", help="makes real LLM calls (owner approval required)")
    mode_group.add_argument("--offline", action="store_true", help="scripted stand-in for the LLM; no LLM call")
    ap.add_argument("--jobs", type=int, default=n_jobs, help="parallel outer folds (results are identical)")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print("refused: ANTHROPIC_API_KEY is not set in the environment (no call was made)", file=sys.stderr)
        return 2
    mode = "live" if args.live else "offline-scripted"
    try:
        model = resolve_model() if args.live else "none (offline scripted stand-in)"
    except ConfigError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print("usage error: --out must be a new or empty folder", file=sys.stderr)
        return 2
    (out / "charts").mkdir(parents=True, exist_ok=True)
    if runner is None:
        if args.live:
            runner = live_runner
        else:
            from .offline import scripted_runner
            runner = scripted_runner()

    started, t0 = _now(), time.monotonic()
    ws = stage_workspace()
    recorder = Recorder(ws.root)
    recorder.add("run_started", mode=mode, model=model)
    ctx = ToolContext(ws, n_jobs=args.jobs)
    error, info = None, {"usage": {}}
    try:
        info = runner(ctx, model, recorder)
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
    manifest = {
        "run_id": out.name, "status": "RUNNING", "mode": mode, "synthetic": True,
        "model": {"id": model, "provider": "anthropic (native SDK through crewai)" if args.live else "none"},
        "llm_settings": LLM_SETTINGS, "limits": {"agents": AGENT_LIMITS, "crew_max_rpm": CREW_MAX_RPM},
        "safety_env": SAFETY_ENV, "package_versions": package_versions(), "input_sha256": input_hashes(),
        "token_usage": usage,
        "timings": {"started_at": started, "finished_at": _now(), "duration_seconds": round(time.monotonic() - t0, 1),
                    "tasks": tasks or [{"task": "none"}]},
        "tool_calls": sum(1 for e in recorder.events if e["event"] == "tool_call"),
        "error": error,
        "artifacts_sha256": {p.relative_to(out).as_posix(): _sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    from .validators import fresh_metrics, run_validators
    results = run_validators(out, manifest, os.environ.get("ANTHROPIC_API_KEY"),
                             recompute=recompute or (lambda: fresh_metrics(args.jobs)))
    passed = error is None and all(r["status"] == "PASS" for r in results)
    manifest |= {"status": "PASS" if passed else "FAILED", "validators": results}
    (out / "run_manifest.json").write_text(dumps(manifest), encoding="utf-8", newline="\n")
    ws.cleanup()
    print(json.dumps({"status": manifest["status"], "validators": {r["id"]: r["status"] for r in results},
                      "error": error, "out": out.name}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
