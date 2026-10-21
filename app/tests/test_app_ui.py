"""P5-F: the Streamlit app, rendered headlessly with Streamlit's AppTest. ALL DATA IS SYNTHETIC.

Run in the app's own environment (not the ML one):

    .venv-app\\Scripts\\python -m pytest app/tests -q

* the app opens on the verified canonical run and renders every tab and every report viewer without error;
* rendering never starts a run; a run starts only from a button, and the full run also needs the
  confirmation box; the ML packages are not even installed in this environment.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app import evidence as E  # noqa: E402
from app import runner as RUN  # noqa: E402

APP = str(REPO / "app" / "streamlit_app.py")


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Record every start request instead of starting a process; make the ML environment look present."""
    calls = []
    fake_python = tmp_path / "python.exe"
    fake_python.write_text("")
    monkeypatch.setenv("PSBA_ML_PYTHON", str(fake_python))
    monkeypatch.setattr(RUN, "start_flow", lambda scenario="none", n_jobs=1: calls.append((scenario, n_jobs))
                        or f"app-test-{len(calls)}")
    monkeypatch.setattr(RUN, "_PROCESSES", {})
    return calls


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_the_app_environment_holds_no_ml_package():
    for pkg in ("sklearn", "crewai", "joblib", "anthropic"):
        assert importlib.util.find_spec(pkg) is None, pkg


def test_opens_on_the_verified_canonical_run_without_starting_anything(started):
    at = _app()
    assert at.title[0].value.startswith("Budget Overrun AI")
    assert any(s.value.startswith("VERIFIED") for s in at.success)
    assert {m.label: m.value for m in at.metric}["Selected model"] == "LR"
    assert len(at.tabs) == 6 and started == []
    for name in ("crew2", "budget_flow", "orchestrator", "sklearn", "joblib", "crewai"):
        assert name not in sys.modules


def test_results_labels_separate_current_cv_from_historical_test(started):
    at = _app()
    subheaders = [s.value for s in at.subheader]
    assert "A · CURRENT RUN: TRAIN-only grouped cross-validation" in subheaders
    assert "B · HISTORICAL P4 TEST: cited, not recomputed" in subheaders
    assert any("did NOT reopen TEST" in c.value for c in at.caption)
    badges = " ".join(m.value for m in at.markdown)            # st.badge renders as a markdown badge
    assert "LLM responses: SCRIPTED" in badges and "CrewAI orchestration: REAL" in badges
    assert "ALL DATA IS SYNTHETIC" in badges


def test_every_report_viewer_renders(started):
    at = _app()
    view = next(s for s in at.selectbox if s.label == "View")
    for option in view.options:
        at = view.select(option).run()
        assert not at.exception, (option, [e.value for e in at.exception])
        view = next(s for s in at.selectbox if s.label == "View")
    assert any("never loaded" in c.value or "Not loaded" in c.value for c in at.caption)


def test_a_stopped_run_shows_that_crew2_never_ran(started):
    at = _app()
    run_box = at.sidebar.selectbox[0]
    at = run_box.select("p5e-break-money_x100").run()
    assert not at.exception
    assert any("Crew 2 never ran" in m.value for m in at.markdown)
    assert any("Model results exist only for a PASS run" in i.value for i in at.info)
    assert started == []


def test_the_full_run_needs_the_confirmation_box_and_a_click(started):
    at = _app()
    full = next(b for b in at.button if b.label == "Start the full Flow run")
    assert full.disabled
    at = next(c for c in at.checkbox if c.label.startswith("I understand")).check().run()
    full = next(b for b in at.button if b.label == "Start the full Flow run")
    assert not full.disabled and started == []
    at = full.click().run()
    assert started == [("none", min(5, __import__("os").cpu_count() or 1))]


def test_a_contract_break_starts_only_on_click(started):
    at = _app()
    at = next(s for s in at.selectbox if s.label == "Scenario").select("contract_column_deletion").run()
    assert started == []
    at = next(b for b in at.button if b.label == "Run this contract break").click().run()
    assert started == [("contract_column_deletion", 1)]


def test_runtime_notes_give_the_observed_approximate_ranges(started):
    at = _app()
    assert any(c.value.startswith("Roughly 1 minute (observed: about 30–80 s)") for c in at.caption)
    assert any("roughly 4–8 minutes with 5 parallel jobs" in i.value for i in at.info)
    text = " ".join([c.value for c in at.caption] + [i.value for i in at.info])
    assert "About 20 seconds" not in text and "4–5 minutes" not in text


def test_local_source_without_runs_renders(started, monkeypatch, tmp_path):
    monkeypatch.setattr(E, "LOCAL_RUNS_DIR", tmp_path / "none")
    at = _app()
    at = at.sidebar.radio[0].set_value("Local runs (out/flow_runs)").run()
    assert not at.exception and started == []


class _Proc:
    """A stand-in for a started Flow process: running while ``code`` is None."""

    def __init__(self, code, age_s):
        self.returncode, self.started = code, time.monotonic() - age_s

    def poll(self):
        return self.returncode


@pytest.fixture
def local_runs(started, monkeypatch, tmp_path):
    """Three app runs in start order: two finished (with a flow manifest), the newest still running."""
    monkeypatch.setattr(E, "LOCAL_RUNS_DIR", tmp_path / "flow_runs")
    monkeypatch.setattr(RUN, "LOG_DIR", tmp_path / "flow_runs" / "_logs")
    ids = ["app-run-20261020-100000", "app-break-row_deletion-20261020-110000", "app-run-20261020-120000"]
    for rid, total in zip(ids[:2], (271.4, 18.9)):
        (tmp_path / "flow_runs" / rid).mkdir(parents=True)
        (tmp_path / "flow_runs" / rid / "flow_manifest.json").write_text(
            json.dumps({"status": "PASS", "timing": {"total_seconds": total}}), encoding="utf-8")
    procs = {ids[0]: _Proc(0, 900), ids[1]: _Proc(0, 600), ids[2]: _Proc(None, 42)}
    monkeypatch.setattr(RUN, "_PROCESSES", procs)
    monkeypatch.setattr(RUN, "active_runs", lambda: [])      # keep AppTest from entering the live-refresh loop
    return ids


def test_run_boxes_follow_start_order_without_a_ghost_entry(local_runs):
    at = _app()
    heads = [m.value for m in at.markdown if m.value.startswith("**app-")]
    assert heads == [f"**{rid}**: " + s for rid, s in
                     zip(local_runs, ("finished (exit code 0)", "finished (exit code 0)", "running"))]


def test_a_finished_run_shows_its_flow_total_and_only_a_running_run_shows_elapsed(local_runs):
    at = _app()
    captions = [c.value for c in at.caption]
    assert "Flow total 271.4 s (from its flow_manifest.json)" in captions
    assert "Flow total 18.9 s (from its flow_manifest.json)" in captions
    elapsed = [c for c in captions if c.startswith("elapsed")]           # only the running run; render takes ~1 s
    assert len(elapsed) == 1 and 42 <= int(elapsed[0].split()[1]) < 100 and elapsed[0].endswith(" s")
