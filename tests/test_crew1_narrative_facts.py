"""Narrative facts and the narrative rules of V6/V7 (after p3-live-004). Offline. ALL DATA IS SYNTHETIC.

The fixtures are the statements of p3-live-004 (and p3-live-003) that V1-V9 passed although they were
wrong; each must now fail. Correctly phrased counterparts must pass.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pytest

from crew1 import validators as V
from crew1.crew import load_config
from crew1.tools import ToolContext
from crew1.tools import narrative_checks as N
from crew1.tools.workspace import REPO, Sources, stage_workspace
from gate.reconstruct import OVERRUN_TOLERANCE, WARNING_THRESHOLD
from tests.crew1_helpers import no_network  # noqa: F401
from tests.test_crew1_validators import SOURCES, good_run, run_copy  # noqa: F401  (shared offline run)

pytestmark = pytest.mark.usefixtures("no_network")

EDA_ERRORS = [   # (section, verbatim p3-live-004 statement, rule)
    ("overview", "Overrun rates decline across stages from 51.6% at S0 to 42.9% at S3, reflecting increasing budget "
                 "commitment and cost certainty as events approach.", "N-CAUSAL"),
    ("overview", "The analysis describes budget overrun prevalence, signed budget gaps, feature associations, and the "
                 "impact of budget revisions and fee terms on cost outcomes.", "N-CAUSAL"),
    ("signed_gap", "The signed budget gap (pre-VAT) has a mean of −1,863.6 ILS and a median of −988.9 ILS "
                   "across all snapshots, indicating a tendency toward underbudgeting.", "N-SIGN"),
    ("signed_gap", "Gap distributions vary by stage: at S0, the mean gap is 1,894.2 ILS (median 282.4 ILS), reflecting "
                   "early-stage overbudgeting; by S3, the mean gap is −6,026.0 ILS (median −3,441.9 ILS), "
                   "reflecting final-stage underbudgeting.", "N-SIGN"),
    ("budget_revisions", "This pattern suggests that budget revisions are associated with downward adjustments that "
                         "reduce overrun prevalence.", "N-CAUSAL"),
    ("correlations", "Most feature correlations are weak to moderate, suggesting that budget gap and overrun status "
                     "are influenced by multiple factors rather than dominated by any single feature.", "N-CAUSAL"),
]
INSIGHT_ERRORS = [   # (verbatim statement, rule)
    ("The dataset comprises 1,538 eligible snapshots from 375 fully eligible productions, spanning four stages "
     "(S0–S3) of budget tracking.", "N-POPULATION"),
    ("Overall overrun prevalence—where commitment amounts exceed payment amounts—is 47.3% (95% CI: "
     "44.8%–49.8%).", "N-DEFINITION"),
    ("This downward trend suggests that as productions progress, the gap between commitments and payments tends to "
     "narrow.", "N-DEFINITION"),
    ("Signed gaps (commitment minus payment) show substantial variation, with both positive overruns and negative "
     "underruns present across the dataset.", "N-DEFINITION"),
    ("The signed gap metric depends on the completeness and accuracy of commitment and payment recording; gaps may "
     "reflect data quality issues as well as genuine budget dynamics.", "N-DEFINITION"),
    ("The test minus train overrun-prevalence difference is +14.7 pp, 95% bootstrap interval [+3.3, +26.2] pp, "
     "assessed as a sampling outcome independent of the outcome mechanism via the frozen_split_prevalence_difference "
     "analysis in stats.json.", "N-ATTRIBUTION"),
    # p3-live-003
    ("The clean dataset contains 1,538 eligible snapshots across four stages (S0-S3) from 400 productions.",
     "N-POPULATION"),
    ("The test minus train overrun-prevalence difference and its 95% bootstrap interval are reported in the "
     "frozen_split_prevalence_difference metric from the handoff gate assessment.", "N-ATTRIBUTION"),
]
CORRECT = [
    "The 1,538 eligible snapshots come from 397 productions: 375 fully eligible and 22 partially eligible.",
    "Among the 400 scheduled productions, 375 are fully eligible.",
    "The mean signed gap is −1,863.6 ILS, so on average the final cost came in under budget.",
    "At S0 the mean gap is 1,894.2 ILS (over budget); at S3 it is −6,026.0 ILS (under budget).",
    "Overrun prevalence is lower at S3 (42.9%) than at S0 (51.6%); this is an association, not an effect of the stage.",
    "Snapshots with two or more budget revisions have the lowest overrun prevalence (9.8%), an association only.",
    "The signed gap is the verified final cost plus the included fee minus the approved budget; payments are not "
    "part of it.",
    "The difference comes from the frozen_split_prevalence_difference analysis (production-level bootstrap). The split "
    "is frozen and was not tuned; the difference was assessed as a sampling outcome, independent of the outcome "
    "mechanism.",
    "Does the number of budget revisions to date help explain the signed gap?",
    "Additionally, 15 payment rows were excluded as invalid due to out-of-range amount_gross values.",
    # the Policy C exclusion reasons and hypotheses a correct run writes (false positives found before p3-live-005)
    "62 snapshots at S0 to S3 were excluded under Policy C because their commitment category was missing.",
    "Snapshots at S2 were excluded due to an undated manual cost.",
    "Invalid payment rows were excluded before the gap was computed.",
    "Snapshots with more budget revisions show lower overrun prevalence; whether revisions reduce overruns is a "
    "hypothesis for Crew 2.",
]


@pytest.fixture(scope="module")
def facts():
    ws = stage_workspace()
    try:
        ctx = ToolContext(ws, Sources.default())
        ctx.apply_policy_c()
        ctx.build_dataset_contract(None)
        yield N.narrative_facts(ctx.log(), ctx.contract())
    finally:
        ws.cleanup()


# ---------------------------------------------------------------- the facts

def test_the_facts_hold_the_contract_definitions_and_the_eligible_population(facts):
    assert facts["population"] == {
        "eligible_snapshots": 1538, "scheduled_snapshots": 1600, "productions_scheduled": 400,
        "productions_with_eligible_snapshots": 397, "productions_fully_eligible": 375,
        "productions_partially_eligible": 22, "productions_fully_excluded": 3,
        "sentence": "The 1,538 eligible snapshots come from 397 productions: 375 fully eligible and 22 partially "
                    "eligible (3 of the 400 scheduled productions have no eligible snapshot)."}
    d = facts["definitions"]
    assert (OVERRUN_TOLERANCE, WARNING_THRESHOLD) == (0.5, 0.02)
    assert "exceeds 0.5 ILS" in d["is_overrun"] and "exceeds 2% of that budget" in d["is_material_overrun"]
    assert d["sign"] == "a positive gap means over budget; a negative gap means under budget"
    doc = (REPO / "docs" / "data_generation.md").read_text(encoding="utf-8")
    assert "budget_gap_ils     = final_cost_pre_vat + fee_in_budget_t − approved budget at the snapshot" in doc
    assert "is_overrun         = budget_gap_ils > 0.5" in doc and "gap > 2% of the budget" in doc
    assert set(facts["statement_kinds"]) == {"fact", "association", "hypothesis"}
    assert "not the handoff gate" in facts["attribution"]["frozen split difference and its interval"]


def test_the_production_counts_are_a_v5_source(run_copy):
    a = json.loads((run_copy / "stats.json").read_text(encoding="utf-8"))["analyses"]["policy_c_exclusions"]
    assert (a["n_productions_scheduled"], a["n_productions_with_eligible_snapshots"], a["n_productions_fully_eligible"],
            a["n_productions_partially_eligible"], a["n_productions_fully_excluded"]) == (400, 397, 375, 22, 3)
    with open(run_copy / "insights.md", "a", encoding="utf-8") as f:
        f.write("\nThe 1,538 eligible snapshots come from 397 productions: 375 fully eligible and 22 partially "
                "eligible.\n")
    assert V.v5_claims(run_copy, SOURCES)["status"] == "PASS"


def test_both_writing_tasks_carry_the_definitions_verbatim():
    _, tasks = load_config()
    flat = lambda t: " ".join(t.split())
    for name, tool in (("explore_and_report", "list_analyses"), ("contract_and_insights", "build_dataset_contract")):
        desc = flat(tasks[name]["description"])
        assert N.DEFINITIONS_TEXT in desc, name
        assert f"narrative_facts that {tool} returns" in desc, name
        assert "never present them as causes, effects, impacts or explanations" in desc, name
        assert "Hypotheses: questions for Crew 2." in desc, name


def test_the_first_tool_of_each_writing_agent_returns_the_facts():
    ws = stage_workspace()
    try:
        ctx = ToolContext(ws, Sources.default())
        ctx.apply_policy_c()
        for out in (ctx.list_analyses(), ctx.build_dataset_contract(None)):
            f = out["narrative_facts"]
            assert f["definitions_text"] == N.DEFINITIONS_TEXT
            assert f["population"]["productions_with_eligible_snapshots"] == 397
    finally:
        ws.cleanup()


# ---------------------------------------------------------------- the rules

@pytest.mark.parametrize("section,statement,rule", EDA_ERRORS)
def test_each_p3_live_004_eda_statement_fails_v6(facts, run_copy, section, statement, rule):
    assert rule in {r for r, _ in N.findings(statement, facts)}
    p = run_copy / "eda_report.html"
    marker = f'<div class="narrative" data-section="{section}">'
    page = p.read_text(encoding="utf-8")
    assert page.count(marker) == 1
    p.write_text(page.replace(marker, f"{marker}<p>{html.escape(statement)}</p>"), encoding="utf-8")
    res = V.v6_eda(run_copy)
    assert res["status"] == "FAIL" and any(f"[{section}]: {rule}:" in d for d in res["details"]), res["details"]


@pytest.mark.parametrize("statement,rule", INSIGHT_ERRORS)
def test_each_known_insights_statement_fails_v7(facts, run_copy, statement, rule):
    assert rule in {r for r, _ in N.findings(statement, facts)}
    with open(run_copy / "insights.md", "a", encoding="utf-8") as f:
        f.write(f"\n{statement}\n")
    res = V.v7_insights(run_copy)
    assert res["status"] == "FAIL" and any(f"insights.md: {rule}:" in d for d in res["details"]), res["details"]


@pytest.mark.parametrize("statement", CORRECT)
def test_a_correctly_phrased_statement_passes(facts, run_copy, statement):
    assert N.findings(statement, facts) == []
    with open(run_copy / "insights.md", "a", encoding="utf-8") as f:
        f.write(f"\n{statement}\n")
    assert V.v7_insights(run_copy)["status"] == "PASS"


def test_the_scripted_correct_run_still_passes_v6_and_v7(good_run):
    _, out = good_run
    assert V.v6_eda(out)["status"] == "PASS" and V.v7_insights(out)["status"] == "PASS"


def test_the_pipeline_uses_facts_recomputed_independently(run_copy):
    """run_validators derives the facts from a fresh Policy C recomputation, not from the run's own log."""
    (run_copy / "cleaning_log.json").unlink()
    with open(run_copy / "insights.md", "a", encoding="utf-8") as f:
        f.write(f"\n{INSIGHT_ERRORS[0][0]}\n")
    _, log, contract = V.fresh_handoff(SOURCES)
    res = V.v7_insights(run_copy, N.narrative_facts(log, contract))
    assert any("N-POPULATION" in d for d in res["details"])


@pytest.mark.parametrize("statement,rule", [
    ("Ineligible snapshots at S0 reflect early-stage budget uncertainty.", "N-CAUSAL"),
    ("Excluded snapshots drive the lower overrun prevalence at S3.", "N-CAUSAL"),
    ("After invalid payments were excluded, the signed gap (commitment minus payment) was recomputed.",
     "N-DEFINITION"),
])
def test_an_exclusion_sentence_does_not_excuse_a_cause_or_a_wrong_definition(facts, statement, rule):
    assert rule in {r for r, _ in N.findings(statement, facts)}


def test_an_exclusion_reason_finding_is_feedback_not_a_failure(facts):
    s = "Invalid payments were excluded because they had no matching commitment."
    assert [r for r, _ in N.findings(s, facts)] == ["N-EXCLUSION"] and N.blocking_problems(s, facts) == []


def test_the_unchanged_p3_live_004_fails_the_strengthened_validators_when_present():
    out = REPO / "artifacts" / "crew1" / "p3-live-004"
    if not (out / "insights.md").is_file():
        pytest.skip("the p3-live-004 audit record is local only")
    v6, v7 = V.v6_eda(out), V.v7_insights(out)
    assert v6["status"] == "FAIL" and v7["status"] == "FAIL"
    rules6 = {r for r in ("N-SIGN", "N-CAUSAL") if any(r in d for d in v6["details"])}
    rules7 = {r for r in ("N-DEFINITION", "N-POPULATION", "N-ATTRIBUTION") if any(r in d for d in v7["details"])}
    assert rules6 == {"N-SIGN", "N-CAUSAL"} and rules7 == {"N-DEFINITION", "N-POPULATION", "N-ATTRIBUTION"}
