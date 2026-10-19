"""Crew 1 post-run validators (P3 2.9.5) and the CLI (2.9.6, 2.9.7). Offline: the runs use a scripted
stand-in for the LLM that drives the real tools; no LLM call and no network. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from crew1 import validators as V
from crew1.run import main
from crew1.tools import eda as E
from crew1.tools.workspace import Sources
from tests.crew1_helpers import no_network, scripted_runner  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")
DUMMY_KEY = "offline-" + "test-value-not-a-key"
SOURCES = Sources.default()


def _run(out: Path, monkeypatch, runner, argv_extra=("--live",)) -> int:
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.delenv("CREW_MODEL", raising=False)
    return main(["--out", str(out), *argv_extra], runner=runner, mode="offline-simulated")


@pytest.fixture(scope="module")
def good_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    out = tmp_path_factory.mktemp("crew1") / "good"
    try:
        code = _run(out, mp, scripted_runner())
    finally:
        mp.undo()
    return code, out


@pytest.fixture
def run_copy(good_run, tmp_path):
    dst = tmp_path / "run"
    shutil.copytree(good_run[1], dst)
    return dst


def _manifest(out: Path) -> dict:
    return json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))


def _stats(out: Path) -> dict:
    return json.loads((out / "stats.json").read_text(encoding="utf-8"))


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


def _sources(out: Path) -> dict:
    frozen = json.loads((SOURCES.contracts_dir / "frozen_split.json").read_text(encoding="utf-8"))
    contract = json.loads((out / "dataset_contract.json").read_text(encoding="utf-8"))
    return V.claim_sources(_stats(out), contract, frozen["counts"])


# ---------------------------------------------------------------- a complete offline run

def test_a_complete_offline_run_passes_every_validator(good_run):
    code, out = good_run
    m = _manifest(out)
    assert code == 0 and m["status"] == "PASS"
    assert [v["id"] for v in m["validators"]] == [f"V{i}" for i in range(1, 10)]
    assert all(v["status"] == "PASS" for v in m["validators"]), m["validators"]
    for name in ("clean_data.csv", "eda_report.html", "insights.md", "dataset_contract.json", "stats.json",
                 "cleaning_log.json", "gate_result.json", "trace.jsonl", "run_manifest.json"):
        assert (out / name).is_file(), name
    assert m["model"]["id"] and m["token_usage"]["total_tokens"] == 0 and m["mode"] == "offline-simulated"
    assert DUMMY_KEY not in (out / "trace.jsonl").read_text(encoding="utf-8")


def test_every_agent_used_its_tools_in_the_trace(good_run):
    events = [json.loads(x) for x in (good_run[1] / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    used = {e["tool"] for e in events if e["event"] == "tool_call"}
    assert used == {"load_raw_table", "detect_defects", "apply_policy_c", "eligibility_summary", "list_analyses",
                    "run_analysis", "render_chart", "assemble_eda_report", "build_dataset_contract",
                    "run_handoff_gate", "write_insights"}


# ---------------------------------------------------------------- 5.1 / 5.2 / 5.3 quantitative claims

def test_a_fabricated_quantitative_claim_fails(run_copy):
    claim = V.extract_claims("93.1%")[0]
    assert not V.claim_matches(claim, _sources(run_copy))                 # precondition: truly fabricated
    _append(run_copy / "insights.md", "\nAlso, 93.1% of eligible snapshots overrun.\n")
    res = V.v5_claims(run_copy, SOURCES)
    assert res["status"] == "FAIL" and any("93.1%" in d for d in res["details"])


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_insights_prompt_limits_numbers_to_the_v5_sources():
    """Prompt constraint (p3-live-002): the insights author was allowed 'numbers from the context', which V5 rejects."""
    from crew1.crew import load_config
    _, tasks = load_config()
    desc = _flat(tasks["contract_and_insights"]["description"])
    assert "numbers from the context or from your tools" not in desc
    assert ("every number in insights.md must come from stats.json" in desc and "dataset_contract.json" in desc
            and "approved frozen-split counts" in desc)
    assert "Do not quote any other number" in desc and "cleaning narrative" in desc
    assert "Before you write any split count, verify it against the deterministic tool output" in desc
    assert "Never estimate, round or derive a split count" in desc
    assert "remove or correct each number it lists as unverified" in desc
    handover = _flat(tasks["explore_and_report"]["expected_output"])
    assert "number of eligible snapshots in each frozen split group" in handover


def test_split_and_cleaning_counts_are_checked_against_the_v5_sources(run_copy):
    """The verification the prompt relies on: verified split counts pass; the p3-live-002 split counts
    (train 1,100 / test 308 / demo 130) and a cleaning-narrative count are flagged by write_insights' checks
    and fail V5."""
    groups = _stats(run_copy)["analyses"]["frozen_split_prevalence_difference"]["groups"]
    n = {g: groups[g]["n"] for g in ("train", "test", "demo")}
    good = f"\nThe split holds {n['train']:,} train, {n['test']:,} test and {n['demo']:,} demo snapshots.\n"
    _append(run_copy / "insights.md", good)
    assert V.v5_claims(run_copy, SOURCES)["status"] == "PASS"
    wrong = "\nTrain has 1,100 snapshots, test 308 and demo 130; 22 duplicate commitments were dropped.\n"
    src = _sources(run_copy)
    assert not any(V.claim_matches(c, src) for c in V.extract_claims(wrong))     # precondition: all unverifiable
    _append(run_copy / "insights.md", wrong)
    res = V.v5_claims(run_copy, SOURCES)
    assert res["status"] == "FAIL" and len(res["details"]) == 4
    frozen = json.loads((SOURCES.contracts_dir / "frozen_split.json").read_text(encoding="utf-8"))
    contract = json.loads((run_copy / "dataset_contract.json").read_text(encoding="utf-8"))
    lint = V.lint_narrative(wrong, _stats(run_copy), contract, frozen["counts"])
    assert len(lint["unverified_quantitative_claims"]) == 4


def test_a_fabricated_count_and_amount_fail(run_copy):
    _append(run_copy / "insights.md", "\nThe handoff has 1,537 rows and a median gap of 98,765 ILS.\n")
    res = V.v5_claims(run_copy, SOURCES)
    assert res["status"] == "FAIL" and len(res["details"]) == 2


def test_a_claim_with_the_wrong_unit_fails(run_copy):
    diff = _stats(run_copy)["analyses"]["frozen_split_prevalence_difference"]["test_minus_train_pp"]
    right, wrong = f"{diff:.1f} pp", f"{diff:.1f}%"
    src = _sources(run_copy)
    assert V.claim_matches(V.extract_claims(right)[0], src)
    assert not V.claim_matches(V.extract_claims(wrong)[0], src)            # pp written as a percentage
    _append(run_copy / "insights.md", f"\nTest exceeds train by {wrong}.\n")
    res = V.v5_claims(run_copy, SOURCES)
    assert res["status"] == "FAIL" and any(wrong in d for d in res["details"])


def test_a_rate_written_in_pp_and_a_count_written_in_ils_fail(run_copy):
    rate = _stats(run_copy)["analyses"]["overrun_prevalence"]["overall"]["rate"]
    _append(run_copy / "insights.md", f"\nPrevalence is {100 * rate:.1f} pp and the handoff has 1,538 ILS.\n")
    res = V.v5_claims(run_copy, SOURCES)
    assert res["status"] == "FAIL" and len(res["details"]) == 2


def test_identifiers_stage_names_chart_numbers_and_dates_do_not_fail(run_copy):
    _append(run_copy / "insights.md",
            "\n1. SYN-0001 and SYN-0042-C03 at stages S0, S1, S2 and S3 (S0-S3), see Chart 2, Table 4, "
            "Figure 1 and Section 3.\n2. Snapshots dated 2024-09-04, 2025/01/14 and 14-Jan-2025, in 2025; "
            "seed 42, version 1.15.28, dataset v2, Crew 2 and H1; the 95th percentile and a 95% Wilson interval; "
            "the 2+ revisions bucket.\n")
    res = V.v5_claims(run_copy, SOURCES)
    assert res["status"] == "PASS", res["details"]


def test_rounding_signs_ranges_and_intervals():
    src = {"fraction": [0.473342, 0.444649, 0.591503], "pp": [14.6854, 3.270041, 26.220708],
           "count": [1538.0, 728.0], "ILS": [-13581.87, 440677.97], "ratio": [0.266408]}
    ok = ("47.3%", "47%", "(728/1538)", "1,538", "+14.7 pp", "14.7 percentage points", "[+3.3, +26.2] pp",
          "44.5%-59.2%", "-13,581.87 ILS", "ILS 440,678", "about 440,000 ILS", "0.27", "0.266")
    bad = ("47.4%", "-14.7 pp", "+3.2 pp", "440,000 ILS", "0.3 ILS", "14.7%", "1539", "0.28")
    for text in ok:
        assert all(V.claim_matches(c, src) for c in V.extract_claims(text)), text
    for text in bad:
        assert not all(V.claim_matches(c, src) for c in V.extract_claims(text)), text


def test_an_unclassifiable_number_is_treated_as_quantitative():
    claims = V.extract_claims("We saw 7 odd things and 3.14159 of something.")
    assert [c.unit for c in claims] == ["integer", "decimal"]
    assert not any(V.claim_matches(c, {"count": [1.0]}) for c in claims)


def test_every_stats_value_has_a_unit(good_run):
    leaves = list(E.numeric_leaves(_stats(good_run[1])))
    assert len(leaves) > 300 and {u for _, u in leaves} <= {"count", "fraction", "pp", "ILS", "ratio", "number",
                                                            "days"}
    with pytest.raises(ValueError):
        list(E.numeric_leaves({"analyses": {"x": {"mystery": 1.0}}}))


# ---------------------------------------------------------------- 5.4 / 5.5 sections and forbidden claims

def test_a_missing_insights_section_fails(run_copy):
    p = run_copy / "insights.md"
    p.write_text(p.read_text(encoding="utf-8").replace("## Limitations", "## Other notes"), encoding="utf-8")
    res = V.v7_insights(run_copy)
    assert res["status"] == "FAIL" and any("Limitations" in d for d in res["details"])


def test_a_missing_eda_section_or_too_few_charts_fails(run_copy):
    p = run_copy / "eda_report.html"
    html = p.read_text(encoding="utf-8")
    emptied = re.sub(r'(<div class="narrative" data-section="limitations">).*?(</div>)', r"\1\2", html, flags=re.S)
    p.write_text(emptied, encoding="utf-8")
    assert V.v6_eda(run_copy)["status"] == "FAIL"
    p.write_text(re.sub(r"<img [^>]*>", "", html), encoding="utf-8")
    res = V.v6_eda(run_copy)
    assert res["status"] == "FAIL" and any("charts" in d for d in res["details"])


def test_the_frozen_split_note_must_carry_the_numbers_and_statements(run_copy):
    p = run_copy / "insights.md"
    md = p.read_text(encoding="utf-8")
    note = V.markdown_sections(md)["frozen split note"]
    p.write_text(md.replace(note, "The split is frozen."), encoding="utf-8")
    res = V.v7_insights(run_copy)
    assert res["status"] == "FAIL" and sum("frozen split note" in d for d in res["details"]) >= 5


@pytest.mark.parametrize("sentence", ["The model will predict overruns for new productions.",
                                      "We recommend rebalancing the frozen split.",
                                      "The test group should be re-split to match train.",
                                      "These results demonstrate real-world accuracy."])
def test_a_forbidden_claim_fails(run_copy, sentence):
    _append(run_copy / "insights.md", f"\n{sentence}\n")
    res = V.v7_insights(run_copy)
    assert res["status"] == "FAIL" and any("forbidden" in d for d in res["details"])


def test_negated_statements_are_not_forbidden():
    for s in ("The split was not rebalanced or re-split.", "Nothing here is evidence of real-world accuracy.",
              "No model will predict anything in this phase."):
        assert V.forbidden_claims(s) == [], s


# ---------------------------------------------------------------- 5.6 / 5.7 contract and data tampering

def test_an_edited_structural_contract_field_fails(run_copy):
    p = run_copy / "dataset_contract.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    c["columns"][1]["unit"] = "days"
    p.write_text(json.dumps(c, indent=2), encoding="utf-8")
    _, _, fresh = V.fresh_handoff(SOURCES)
    res = V.v3_contract(run_copy, fresh)
    assert res["status"] == "FAIL" and any("columns" in d for d in res["details"])


def test_only_free_text_descriptions_may_differ(run_copy):
    p = run_copy / "dataset_contract.json"
    c = json.loads(p.read_text(encoding="utf-8"))
    c["columns"][0]["description"] = "Production identifier (synthetic)."
    p.write_text(json.dumps(c, indent=2), encoding="utf-8")
    _, _, fresh = V.fresh_handoff(SOURCES)
    assert V.v3_contract(run_copy, fresh)["status"] == "PASS"
    c["columns"][0]["description"] = "<script>x</script>"
    p.write_text(json.dumps(c, indent=2), encoding="utf-8")
    assert V.v3_contract(run_copy, fresh)["status"] == "FAIL"


def test_one_changed_clean_data_value_fails_the_byte_check_and_the_gate(run_copy):
    p = run_copy / "clean_data.csv"
    lines = p.read_text(encoding="utf-8").split("\n")
    header = lines[0].split(",")
    row = lines[1].split(",")
    i = header.index("budget_pre_vat")
    row[i] = f"{float(row[i]) + 100:.2f}"
    lines[1] = ",".join(row)
    p.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    expected, _, _ = V.fresh_handoff(SOURCES)
    assert V.v2_clean_data(run_copy, expected)["status"] == "FAIL"
    gate, result = V.v4_gate(run_copy, SOURCES)
    failing = {c["id"] for c in result["checks"] if c["status"] == "FAIL"}
    assert gate["status"] == "FAIL" and {"G-FIN-RECON", "G-FIN-ASOF-BUDGET"} <= failing


# ---------------------------------------------------------------- manifest and trace

def test_an_incomplete_manifest_fails(good_run):
    m = _manifest(good_run[1])
    assert V.v8_manifest(m)["status"] == "PASS"
    for field in ("token_usage", "package_versions", "input_sha256", "policy_c_counts", "model"):
        broken = dict(m)
        broken.pop(field)
        assert V.v8_manifest(broken)["status"] == "FAIL", field


def test_an_unredacted_trace_fails(run_copy):
    p = run_copy / "trace.jsonl"
    original = p.read_text(encoding="utf-8")
    for leak in ("C:" + "\\\\Users\\\\someone\\\\x", "/home/" + "someone/x", "sk-" + "ant-" + "A1b2C3d4E5f6G7h8",
                 DUMMY_KEY):
        p.write_text(original + json.dumps({"event": "x", "output": leak}) + "\n", encoding="utf-8")
        assert V.v9_trace(run_copy, DUMMY_KEY)["status"] == "FAIL", leak


# ---------------------------------------------------------------- 2.9.6 / 2.9.7 the CLI

def test_a_gate_failure_exits_1_with_a_failed_manifest_and_keeps_the_artifacts(tmp_path, monkeypatch):
    def corrupt(ws):
        text = ws.read_output("out.clean_data").decode("utf-8").split("\n")
        header, row = text[0].split(","), text[1].split(",")
        i = header.index("committed_pre_vat")
        row[i] = f"{float(row[i]) + 1000:.2f}"
        text[1] = ",".join(row)
        ws.write("out.clean_data", "\n".join(text))

    out = tmp_path / "corrupt"
    assert _run(out, monkeypatch, scripted_runner(corrupt=corrupt)) == 1
    m = _manifest(out)
    status = {v["id"]: v["status"] for v in m["validators"]}
    assert m["status"] == "FAILED" and status["V2"] == "FAIL" and status["V4"] == "FAIL"
    assert json.loads((out / "gate_result.json").read_text(encoding="utf-8"))["status"] == "FAIL"
    assert all((out / n).is_file() for n in V.REQUIRED_ARTIFACTS)


def test_a_crashing_run_exits_1_with_a_failed_manifest(tmp_path, monkeypatch):
    def boom(ws, sources, model, recorder):
        raise RuntimeError("simulated failure in " + str(ws.root))

    out = tmp_path / "boom"
    assert _run(out, monkeypatch, boom) == 1
    m = _manifest(out)
    assert m["status"] == "FAILED" and m["error"].startswith("RuntimeError") and "<workspace>" in m["error"]


def _never(*args, **kwargs):
    raise AssertionError("the runner must not be called")


def test_without_live_the_cli_exits_2_and_makes_no_call(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    out = tmp_path / "x"
    assert main(["--out", str(out)], runner=_never) == 2 and not out.exists()


def test_without_the_key_the_cli_exits_2_and_makes_no_call(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "x"
    assert main(["--out", str(out), "--live"], runner=_never) == 2 and not out.exists()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert main(["--out", str(out), "--live"], runner=_never) == 2


def test_usage_and_configuration_errors_exit_2(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    assert main(["--live"], runner=_never) == 2                                  # --out missing
    assert main(["--out", str(tmp_path / "a"), "--live", "--bogus"], runner=_never) == 2
    monkeypatch.setenv("CREW_MODEL", "gpt-4o")
    assert main(["--out", str(tmp_path / "b"), "--live"], runner=_never) == 2
    monkeypatch.delenv("CREW_MODEL")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "old.txt").write_text("x", encoding="utf-8")
    assert main(["--out", str(tmp_path / "c"), "--live"], runner=_never) == 2   # never overwrites a run
