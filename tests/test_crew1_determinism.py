"""Crew 1 determinism (P3 2.9.3) and P2 compatibility (2.9.4). Offline. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from crew1.tools import ToolContext
from crew1.tools import eda as E
from crew1.tools.workspace import Sources, stage_workspace
from crew1.validators import fresh_handoff
from gate.run import main as gate_main
from tests.crew1_helpers import eda_sections, no_network  # noqa: F401
from tests.handoff_reference import reference_frames

pytestmark = pytest.mark.usefixtures("no_network")

ORDER_A = list(E.ANALYSES)
ORDER_B = ORDER_A[::-1]


def _build(order):
    ws = stage_workspace()
    ctx = ToolContext(ws, Sources.default())
    ctx.apply_policy_c()
    ctx.build_dataset_contract({"stage": "Snapshot stage, S0 to S3."})
    for a in order:
        ctx.run_analysis(a)
    for c in E.CHARTS:
        ctx.render_chart(c)
    ctx.assemble_eda_report(eda_sections(E.load_stats(ws)))
    names = ["out.clean_data", "out.cleaning_log", "out.dataset_contract", "out.stats", "out.eda_report",
             *(f"out.chart.{c}" for c in E.CHARTS)]
    files = {n: ws.read_output(n) for n in names}
    return ws, files


@pytest.fixture(scope="module")
def two_runs():
    ws1, a = _build(ORDER_A)
    ws2, b = _build(ORDER_B)
    yield (ws1, a), (ws2, b)
    ws1.cleanup()
    ws2.cleanup()


def test_policy_c_contract_stats_charts_and_report_are_byte_identical_across_runs(two_runs):
    (_, a), (_, b) = two_runs
    for name in a:
        assert a[name] == b[name], name                    # stats.json is identical whatever the analysis order


def test_repeated_tool_calls_in_one_workspace_are_byte_identical(two_runs):
    (ws, first), _ = two_runs
    ctx = ToolContext(ws, Sources.default())
    ctx.apply_policy_c()
    ctx.build_dataset_contract({"stage": "Snapshot stage, S0 to S3."})
    for n in ("out.clean_data", "out.cleaning_log", "out.dataset_contract"):
        assert ws.read_output(n) == first[n], n


def test_tool_output_equals_the_validators_fresh_recomputation(two_runs):
    (_, files), _ = two_runs
    data, log, contract = fresh_handoff(Sources.default())
    assert files["out.clean_data"] == data
    assert json.loads(files["out.cleaning_log"]) == log


def test_the_tool_built_handoff_passes_the_unmodified_p2_gate(two_runs, tmp_path):
    (ws, _), _ = two_runs
    code = gate_main(["--stage", "handoff", "--raw", str(Sources.default().raw_dir), "--out", str(tmp_path),
                      "--data", str(ws.root / "out" / "clean_data.csv"),
                      "--contract", str(ws.root / "out" / "dataset_contract.json")])
    result = json.loads((tmp_path / "gate_result.json").read_text(encoding="utf-8"))
    assert code == 0 and result["status"] == "PASS"
    assert all(c["status"] == "PASS" for c in result["checks"] if c["severity"] == "HARD")


def test_the_tool_built_handoff_matches_the_independent_reference(two_runs, expected_ineligible):
    """Independent of gate/policy_c: values from the committed clean v2 snapshots, eligibility from the
    ground truth (tests/handoff_reference.py)."""
    (_, files), _ = two_runs
    ref, ref_contract = reference_frames(expected_ineligible)
    got = pd.read_csv(io.BytesIO(files["out.clean_data"]), dtype=str, keep_default_na=False)
    contract = json.loads(files["out.dataset_contract"])
    assert set(got.columns) == set(ref.columns) and len(got) == len(ref) == contract["row_count"]
    assert contract["exclusion_counts"] == ref_contract["exclusion_counts"]
    key = ["production_id", "stage"]
    got, ref = got.sort_values(key).reset_index(drop=True), ref.sort_values(key).reset_index(drop=True)
    from gate.handoff_spec import COLUMNS
    for c in got.columns:
        if COLUMNS[c][0] == "number":
            diff = (pd.to_numeric(got[c]) - pd.to_numeric(ref[c])).abs().max()
            assert diff <= 0.01 + 1e-9, c
        else:
            assert (got[c] == ref[c]).all(), c
