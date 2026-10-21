"""P5-F: the demo app's evidence layer and its separation from the ML code. ALL DATA IS SYNTHETIC.

The app's modules use the standard library (and, for the page itself, Streamlit) only: no ML code is imported
and no pickle is ever loaded. The committed canonical runs verify; any tampered or missing file makes a run
NOT VERIFIED; file access is limited to allow-listed names inside a run folder; starting a run needs an
explicit call and never passes a credential to the child process. The rendering tests run in the app's own
environment (app/tests/test_app_ui.py).
"""

from __future__ import annotations

import ast
import json
import shutil
import sys

import pytest

from app import evidence as E
from app import runner as RUN
from tests.conftest import REPO

STDLIB = set(sys.stdlib_module_names)


def _imports(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield (node.module or "").split(".")[0]


def test_the_app_imports_only_the_standard_library_streamlit_and_itself():
    for path in sorted((REPO / "app").glob("*.py")):
        extra = set(_imports(path)) - STDLIB - {"app", "streamlit", "__future__"}
        assert not extra, (path.name, extra)
    text = "".join(p.read_text(encoding="utf-8") for p in (REPO / "app").glob("*.py"))
    for forbidden in ("pickle", "joblib", "sklearn", "crew1", "crew2", "budget_flow", "orchestrator."):
        assert f"import {forbidden}" not in text and f"from {forbidden}" not in text


def test_every_committed_canonical_run_verifies():
    runs = E.list_runs("canonical")
    assert "p5e-run-001" in runs and sum("break" in r for r in runs) == 5
    for rid in runs:
        run = E.load_run("canonical", rid)
        assert run.verified, (rid, run.verification)
        assert run.status == ("PASS" if rid == "p5e-run-001" else "STOPPED_CONTRACT")


def test_the_canonical_pass_run_shows_the_approved_results():
    run = E.load_run("canonical", "p5e-run-001")
    assert sorted(run.manifest["required_outputs_sha256"]) == sorted(E.REQUIRED)
    cv = {r["run_id"]: r for r in run.cv_results()}
    assert len(cv) == 9 and cv["F27.is_overrun.LR"]["params_per_fold"] == [{"C": 10.0}] * 5
    assert run.manifest["crew2"]["selected_model"]["candidate"] == "LR"
    ht = E.historical_test(run)
    assert ht["source_run"] == "p4-offline-002" and ht["opened"] == 1
    assert round(ht["primary"]["reliability"]["ece"], 4) == 0.1107 and round(ht["secondary"]["reliability"]["ece"], 4) == 0.0681
    s = run.csv_summary("crew2/features.csv")
    assert s["rows"] == 1538 and s["outcome_columns_present"] == [] and s["rows_by_split"]["train"] == 1084


def test_every_stopped_canonical_run_proves_crew2_never_ran():
    for rid in (r for r in E.list_runs("canonical") if "break" in r):
        run = E.load_run("canonical", rid)
        m = run.manifest
        assert m["crew2"] == {"kickoff_invoked": False, "kickoffs": 0} and m["gate"]["violation_codes"]
        assert not any(r.startswith("crew2/") for r in m["required_outputs_sha256"])
        assert not (run.root / "crew2").exists()


def _copy_run(tmp_path, monkeypatch, rid="p5e-run-001"):
    root = tmp_path / "flow_evidence"
    shutil.copytree(E.EVIDENCE_DIR / "runs" / rid, root / "runs" / rid)
    monkeypatch.setattr(E, "EVIDENCE_DIR", root)
    return root / "runs" / rid


@pytest.mark.parametrize("rel", ["crew2/model.joblib", "crew2/evaluation_report.md", "crew1/clean_data.csv",
                                 "flow_events.jsonl"])
def test_a_tampered_file_makes_the_run_not_verified(tmp_path, monkeypatch, rel):
    d = _copy_run(tmp_path, monkeypatch)
    (d / rel).write_bytes((d / rel).read_bytes() + b"\n")
    run = E.load_run("canonical", "p5e-run-001")
    assert not run.verified and any(rel in p for p in run.verification["problems"])


def test_a_missing_file_or_index_makes_the_run_not_verified(tmp_path, monkeypatch):
    d = _copy_run(tmp_path, monkeypatch)
    (d / "crew2" / "model_card.md").unlink()
    assert not E.load_run("canonical", "p5e-run-001").verified
    (d / "evidence.json").unlink()
    assert E.load_run("canonical", "p5e-run-001").verification["problems"] == ["no evidence.json"]


def test_file_access_is_limited_to_allow_listed_names_inside_the_run():
    run = E.load_run("canonical", "p5e-run-001")
    for bad in ("../flow_manifest.json", "/etc/passwd", "C:\\x", "crew2\\model.joblib", "crew2/../../README.md",
                "crew2/model.joblib:stream", ""):
        with pytest.raises(E.EvidenceError):
            run.path(bad)
    for bad_id in ("../x", "a/b", "", ".hidden", "x" * 100):
        with pytest.raises(E.EvidenceError):
            E.load_run("canonical", bad_id)


def test_redaction_removes_repository_home_and_absolute_paths():
    text = f"{REPO}\\out\\x.log and {str(REPO).replace(chr(92), '/')}/x " + "\\".join(["D:", "data", "x"])
    out = E.redact(text)
    assert str(REPO).lower() not in out.lower() and "data\\x" not in out and out.count("<path>") >= 3


def test_html_reports_render_in_an_empty_sandbox():
    hostile = '<p>x</p><script>parent.alert(1)</script><img src=x onerror="alert(2)">"\'</iframe>'
    out = E.sandboxed_html(hostile)
    assert out.startswith('<iframe sandbox="" ') and out.count("<iframe") == 1 and out.endswith("</iframe>")
    assert "<script" not in out and 'onerror="' not in out and "</iframe>" not in out[:-len("</iframe>")]
    eda = E.load_run("canonical", "p5e-run-001").read_text("crew1/eda_report.html")
    assert "<script" not in eda.lower() and E.sandboxed_html(eda).count('sandbox=""') == 1


def test_runner_starts_nothing_on_import_and_strips_credentials(monkeypatch):
    assert RUN.known_runs() == [] or all(r.startswith("app-") for r in RUN.known_runs())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("SOME_PASSWORD", "x")
    env = RUN.child_env()
    assert not {"ANTHROPIC_API_KEY", "GITHUB_TOKEN", "SOME_PASSWORD"} & set(env) and "PATH" in {k.upper() for k in env}
    with pytest.raises(ValueError):
        RUN.start_flow("drop_tables")


def test_runner_command_is_the_documented_flow_cli(monkeypatch, tmp_path):
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen["cmd"], seen["kw"] = cmd, kw

        def poll(self):
            return 0

    monkeypatch.setattr(RUN.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(RUN, "ml_python", lambda: tmp_path / "python.exe")
    monkeypatch.setattr(RUN, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(RUN, "_PROCESSES", {})
    rid = RUN.start_flow("money_x100", 2)
    assert seen["cmd"][1:5] == ["-W", "ignore", "-m", "orchestrator.flow"]
    assert seen["cmd"][-4:] == ["--scenario", "money_x100", "--n-jobs", "2"] and rid.startswith("app-break-money_x100-")
    assert seen["kw"]["stdin"] is RUN.subprocess.DEVNULL and "ANTHROPIC_API_KEY" not in seen["kw"]["env"]


def test_runner_lists_runs_in_start_order_and_freezes_elapsed_at_exit(monkeypatch, tmp_path):
    class Proc:
        def __init__(self, started):
            self.started, self.returncode, self.code = started, None, None

        def poll(self):
            self.returncode = self.code
            return self.code

    clock = [1000.0]
    monkeypatch.setattr(RUN.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(E, "LOCAL_RUNS_DIR", tmp_path)
    old, new = Proc(900.0), Proc(990.0)
    monkeypatch.setattr(RUN, "_PROCESSES", {"app-run-a": old, "app-run-b": new})
    assert RUN.known_runs() == ["app-run-a", "app-run-b"] and RUN.active_runs() == ["app-run-a", "app-run-b"]
    assert RUN.progress("app-run-b")["elapsed_s"] == 10 and RUN.progress("app-run-b")["running"]
    new.code = 0
    first = RUN.progress("app-run-b")
    clock[0] = 5000.0                                   # long after the exit: elapsed must not keep growing
    later = RUN.progress("app-run-b")
    assert first["elapsed_s"] == later["elapsed_s"] == 10 and not later["running"] and later["exit_code"] == 0
    assert RUN.progress("app-run-a")["elapsed_s"] == 4100 and RUN.active_runs() == ["app-run-a"]
    assert RUN.known_runs() == ["app-run-a", "app-run-b"]


def test_export_never_overwrites_canonical_evidence():
    from app import export_evidence as X
    with pytest.raises(E.EvidenceError):
        X.export("p5e-run-001")


def test_streamlit_config_keeps_the_app_local_and_without_usage_statistics():
    text = (REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in text and 'address = "localhost"' in text


def test_the_canonical_evidence_holds_no_absolute_path_or_secret():
    for p in (E.EVIDENCE_DIR / "runs").rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".html", ".csv"):
            t = p.read_text(encoding="utf-8")
            assert str(REPO).lower() not in t.lower() and "sk-ant-" not in t and "AppData" not in t, p.name
    idx = json.loads((E.EVIDENCE_DIR / "runs" / "p5e-run-001" / "evidence.json").read_text(encoding="utf-8"))
    assert idx["status"] == "PASS" and "crew2/model.joblib" in idx["files_sha256"]
