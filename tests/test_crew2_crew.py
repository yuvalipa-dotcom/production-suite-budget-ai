"""Crew 2 wiring, tool schemas, the offline end-to-end run and its validators. Offline: no key, no
network, no LLM call (a scripted stand-in drives the real tools). ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

import crew2
from crew2 import run as R
from crew2.crew import AGENT_LIMITS, Recorder, build_crew, resolve_model
from crew2.offline import card_sections, scripted_runner
from crew2.tools import TOOLSETS, ToolContext
from crew2.tools.workspace import stage_workspace
from crew2.validators import run_validators
from tests.crew1_helpers import NetworkBlocked  # noqa: F401
from tests.crew2_helpers import no_network, small_grids, small_session  # noqa: F401
from tests.test_crew1_authoring import _objects, anthropic_schemas

pytestmark = pytest.mark.usefixtures("no_network")

EXPECTED_TOOLS = {
    "Feature & Leakage Engineer": {"build_handoff", "feature_availability", "run_leakage_checks"},
    "Model Trainer": {"list_runs", "run_grouped_cv", "apply_selection_rule", "fit_final_models"},
    "Evaluator & Model Card Author": {"evaluate_on_test_once", "predict_demo", "render_chart", "write_model_card"},
}


@pytest.fixture(scope="module")
def built():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    ws = stage_workspace()
    try:
        ctx = ToolContext(ws)
        yield build_crew(ctx, resolve_model({}), Recorder(ws.root)), ws
    finally:
        ws.cleanup()
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


# ---------------------------------------------------------------- wiring

def test_three_agents_with_their_own_tools_and_limits(built):
    crew, _ = built
    assert {a.role: {t.name for t in a.tools} for a in crew.agents} == EXPECTED_TOOLS
    names = [t.name for a in crew.agents for t in a.tools]
    assert len(names) == len(set(names))                        # no tool shared between agents
    for a, key in zip(crew.agents, TOOLSETS):
        assert a.max_iter == AGENT_LIMITS[key]["max_iter"] and a.max_execution_time == AGENT_LIMITS[key][
            "max_execution_time"]
        assert not a.allow_delegation and not a.allow_code_execution and not a.memory
        assert a.llm is not crew.agents[0].llm or a is crew.agents[0]                  # one LLM per agent
    assert [t.name for t in crew.tasks] == ["prepare_and_prove_features", "train_and_select", "evaluate_and_document"]


def test_safety_environment_is_forced():
    assert os.environ["CREWAI_DISABLE_TELEMETRY"] == "true" and os.environ["CREWAI_TRACING_ENABLED"] == "false"
    assert os.environ["CREWAI_STORAGE_DIR"] == crew2.SAFETY_ENV["CREWAI_STORAGE_DIR"] == "psba-crew2"


def test_tool_arguments_are_typed_and_strict_schema_safe(built):
    """No free-form mapping argument (p3-live-003): every object the strict schema closes has named fields."""
    crew, _ = built
    schemas = anthropic_schemas([t for a in crew.agents for t in a.tools])
    assert set(schemas) == set().union(*EXPECTED_TOOLS.values())
    for name, tool in schemas.items():
        for path, obj in _objects(tool["input_schema"]):
            if path == "$" and not obj.get("properties"):
                continue                                         # a no-argument tool
            assert obj.get("properties"), (name, path)
            assert all(p.get("type") != "object" or p.get("properties") for p in obj["properties"].values()), name
    assert set(schemas["write_model_card"]["input_schema"]["properties"]) == {
        "summary", "intended_use", "data_and_target", "features", "protocol", "cv_results", "selection",
        "test_results", "secondary_results", "limitations"}
    assert not schemas["apply_selection_rule"]["input_schema"].get("properties")   # C3: nothing to choose


def test_live_run_is_refused_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert R.main(["--out", str(tmp_path / "x"), "--live"]) == 2
    assert R.main(["--out", str(tmp_path / "x")]) == 2                          # a mode is required


# ---------------------------------------------------------------- offline end-to-end

@pytest.fixture(scope="module")
def offline_run(tmp_path_factory, small_session):
    out = tmp_path_factory.mktemp("crew2") / "run"
    code = R.main(["--out", str(out), "--offline"], recompute=lambda: small_session.metrics())
    return code, out


def test_the_offline_run_passes_every_validator(offline_run):
    code, out = offline_run
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert code == 0, [v for v in manifest["validators"] if v["status"] != "PASS"]
    assert manifest["status"] == "PASS" and [v["id"] for v in manifest["validators"]] == [f"W{i}" for i in range(1, 11)]
    assert manifest["token_usage"]["total_tokens"] == 0 and manifest["mode"] == "offline-scripted"
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert m["complete"] and m["test"]["opened"] == 1 and m["demo"]["metrics"] is None
    card = (out / "model_card.md").read_text(encoding="utf-8")
    assert "not a commercially material overrun" in card and "All rights reserved" in card


def _copy(out: Path, tmp: Path) -> Path:
    dst = tmp / "copy"
    shutil.copytree(out, dst)
    return dst


def _validate(out: Path, small_session) -> dict:
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    return {v["id"]: v for v in run_validators(out, manifest, None, recompute=lambda: small_session.metrics())}


def test_an_invented_number_in_the_card_fails_w7(offline_run, tmp_path, small_session):
    _, out = offline_run
    dst = _copy(out, tmp_path)
    card = (dst / "model_card.md").read_text(encoding="utf-8")
    card = card.replace("<!-- /narrative -->", "The model is 97.3% accurate.\n<!-- /narrative -->", 1)
    (dst / "model_card.md").write_text(card, encoding="utf-8")
    v = _validate(dst, small_session)
    assert v["W7"]["status"] == "FAIL" and v["W3"]["status"] == "PASS"


def test_a_forbidden_claim_or_missing_disclosure_fails_w8(offline_run, tmp_path, small_session):
    _, out = offline_run
    dst = _copy(out, tmp_path)
    card = (dst / "model_card.md").read_text(encoding="utf-8")
    card = card.replace("not a commercially material overrun", "an overrun").replace(
        "<!-- /narrative -->", "It has proven real-world accuracy.\n<!-- /narrative -->", 1)
    (dst / "model_card.md").write_text(card, encoding="utf-8")
    assert _validate(dst, small_session)["W8"]["status"] == "FAIL"


def test_an_overridden_selection_fails_w4_and_w3(offline_run, tmp_path, small_session):
    _, out = offline_run
    dst = _copy(out, tmp_path)
    m = json.loads((dst / "metrics.json").read_text(encoding="utf-8"))
    m["selection"]["selected"] = next(c for c in ("B0", "LR", "GB") if c != m["selection"]["selected"])
    (dst / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    v = _validate(dst, small_session)
    assert v["W4"]["status"] == "FAIL" and v["W3"]["status"] == "FAIL"


def test_tampered_demo_predictions_fail_w6(offline_run, tmp_path, small_session):
    _, out = offline_run
    dst = _copy(out, tmp_path)
    lines = (dst / "demo_predictions.csv").read_text(encoding="utf-8").splitlines()
    parts = lines[1].split(",")
    parts[2] = "0.999999"
    lines[1] = ",".join(parts)
    (dst / "demo_predictions.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _validate(dst, small_session)["W6"]["status"] == "FAIL"


def test_a_second_test_evaluation_is_refused_by_the_tool(tmp_path, small_grids, small_session):
    out = tmp_path / "second"
    code = R.main(["--out", str(out), "--offline"], runner=scripted_runner(extra_test_call=True),
                  recompute=lambda: small_session.metrics())
    trace = [json.loads(x) for x in (out / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [e for e in trace if e.get("event") == "tool_call" and e["tool"] == "evaluate_on_test_once"]
    assert [c["ok"] for c in calls] == [True, False] and "TestAlreadyOpened" in calls[1]["output"]
    assert code == 0                                             # the refusal is the guard working


def test_scripted_card_narrative_has_no_unverified_number(small_session):
    from crew2.claims import unverified
    m = json.loads(json.dumps(small_session.metrics()))
    text = "\n\n".join(card_sections(m).values())
    assert unverified(text, m) == []
