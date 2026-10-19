"""Crew 1 authoring tools after p3-live-003. Offline: no key, no network. ALL DATA IS SYNTHETIC.

* The tool schemas CrewAI 1.15.28 actually sends to Anthropic (strict mode) admit non-empty report
  sections and column descriptions (p3-live-003: 22 assemble_eda_report calls arrived as {}).
* write_insights returns the frozen split values from stats.json.
* Deterministic feedback flags the misleading statements of p3-live-003 and invented exclusion reasons.
* The agent instructions state the authoring rules.
"""

from __future__ import annotations

import json
import os

import jsonschema
import pytest
from pydantic import BaseModel

from crew1.crew import DEFAULT_MODEL, load_config
from crew1.tools import ToolContext, build_tools
from crew1.tools import contract as K
from crew1.tools import eda as E
from crew1.tools import narrative_checks as N
from crew1.tools.workspace import Sources, stage_workspace
from crew1.validators import INSIGHT_SECTIONS, eda_narrative, frozen_note_problems
from tests.crew1_helpers import eda_sections, no_network  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")


def anthropic_schemas(tools) -> dict[str, dict]:
    """The tool definitions as crewai 1.15.28 sends them to Anthropic (native tool calling)."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion
    from crewai.utilities.agent_utils import convert_tools_to_openai_schema
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        openai_tools, _, _ = convert_tools_to_openai_schema(tools)
        converted = AnthropicCompletion(model=DEFAULT_MODEL)._convert_tools_for_interference(openai_tools)
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved
    return {t["name"]: t for t in converted}


def _objects(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield path, node
        for k, v in node.items():
            yield from _objects(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _objects(v, f"{path}[{i}]")


@pytest.fixture(scope="module")
def run_ws():
    ws = stage_workspace()
    try:
        ctx = ToolContext(ws, Sources.default())
        ctx.apply_policy_c()
        for a in E.ANALYSES:
            ctx.run_analysis(a)
        for c in E.CHARTS:
            ctx.render_chart(c)
        ctx.build_dataset_contract(None)
        tools = {t.name: t for ts in build_tools(ctx).values() for t in ts}
        yield ws, ctx, tools
    finally:
        ws.cleanup()


# ---------------------------------------------------------------- strict tool schemas (R1)

def test_the_installed_crewai_is_the_pinned_version():
    import crewai
    assert crewai.__version__ == "1.15.28"


def test_no_strict_tool_schema_has_an_object_that_admits_only_empty(run_ws):
    _, _, tools = run_ws
    schemas = anthropic_schemas(list(tools.values()))
    assert set(schemas) == set(tools)
    for name, t in schemas.items():
        assert t.get("strict") is True, name                   # what crewai 1.15.28 actually sends
        for path, obj in _objects(t["input_schema"]):
            if path == "$" and not tools[name].args_schema.model_fields:
                continue                                       # a tool without arguments
            assert obj.get("properties"), f"{name} {path}: an object with no properties admits only {{}}"


def test_strict_schemas_admit_nonempty_sections_and_column_descriptions(run_ws):
    _, _, tools = run_ws
    schemas = anthropic_schemas(list(tools.values()))
    report = schemas["assemble_eda_report"]["input_schema"]
    assert list(report["properties"]) == list(E.SECTIONS) and sorted(report["required"]) == sorted(E.SECTIONS)
    assert all(p["type"] == "string" for p in report["properties"].values())
    jsonschema.validate({s: f"Interpretation text for {s}." for s in E.SECTIONS}, report)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"sections": {}}, report)                       # the p3-live-003 call is now invalid
    contract = schemas["build_dataset_contract"]["input_schema"]
    jsonschema.validate({"column_descriptions": [{"column": "stage", "description": "Snapshot stage."}]}, contract)
    jsonschema.validate({"column_descriptions": None}, contract)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"column_descriptions": [{"column": "stage"}]}, contract)


def test_the_former_dict_argument_admits_only_an_empty_object_under_strict_mode():
    """Documents the p3-live-003 defect with the same conversion: a dict argument is closed to {}."""
    from crewai.tools import BaseTool

    class OldArgs(BaseModel):
        sections: dict[str, str]

    class Old(BaseTool):
        name: str = "old_assemble"
        description: str = "old"
        args_schema: type[BaseModel] = OldArgs

        def _run(self, **kwargs):
            return ""

    schema = anthropic_schemas([Old()])["old_assemble"]["input_schema"]
    jsonschema.validate({"sections": {}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"sections": {"overview": "text"}}, schema)


def test_report_text_and_descriptions_reach_the_artifacts_through_crewai_validation(run_ws):
    ws, ctx, tools = run_ws
    texts = eda_sections(E.load_stats(ws))
    out = json.loads(tools["assemble_eda_report"].run(**texts))   # BaseTool.run validates against args_schema
    assert out["sections_missing_or_short"] == [] and out["context_problems"] == []
    narrative, _ = eda_narrative(ws.read_output("out.eda_report").decode("utf-8"))
    assert {s: len(narrative[s]) >= 40 for s in E.SECTIONS} == {s: True for s in E.SECTIONS}
    entries = [{"column": "stage", "description": "Snapshot stage, S0 to S3."},
               {"column": "production_id", "description": "Synthetic production identifier."}]
    out = json.loads(tools["build_dataset_contract"].run(column_descriptions=entries))
    assert out["described_columns"] == 2
    built = json.loads(ws.read_output("out.dataset_contract"))
    assert {c["name"]: c["description"] for c in built["columns"] if "description" in c} == {
        e["column"]: e["description"] for e in entries}
    log = ctx.log()
    plain = K.build_contract(ws.read_output("out.clean_data"), Sources.default().contracts_dir,
                             built["raw_manifest_sha256"], log["exclusion_counts"], None)
    strip = lambda c: {**c, "columns": [{k: v for k, v in x.items() if k != "description"} for x in c["columns"]]}
    assert strip(built) == strip(plain)                          # only the free-text descriptions differ
    dup = tools["build_dataset_contract"].run(column_descriptions=entries + entries[:1])
    assert dup.startswith("ERROR") and "duplicate column" in dup
    ctx.build_dataset_contract(None)                             # restore the plain contract for other tests


# ---------------------------------------------------------------- frozen split values (R2)

def test_write_insights_returns_the_frozen_split_values_from_stats_json(run_ws):
    ws, _, tools = run_ws
    a = E.load_stats(ws)["analyses"]["frozen_split_prevalence_difference"]
    out = json.loads(tools["write_insights"].run(markdown="## Frozen split note\n\nPending.\n"))
    v = out["frozen_split_values"]
    assert (v["test_minus_train_pp"], v["test_minus_train_lo_pp"], v["test_minus_train_hi_pp"]) == (
        a["test_minus_train_pp"], a["test_minus_train_lo_pp"], a["test_minus_train_hi_pp"])
    assert v["unit"].startswith("pp") and "frozen_split_prevalence_difference" in v["source"]
    assert v["eligible_snapshots_per_group"] == {g: x["n"] for g, x in a["groups"].items()}
    assert out["required_headings"] == [f"## {t}" for t in INSIGHT_SECTIONS]
    note = (f"The test minus train overrun-prevalence difference is {v['write_as']}, from the "
            "frozen_split_prevalence_difference analysis. The split is frozen and was not tuned. The difference "
            "was assessed as a sampling outcome, independent of the outcome mechanism.")
    assert frozen_note_problems(note, E.load_stats(ws)) == []
    out = json.loads(tools["write_insights"].run(markdown=f"## Frozen split note\n\n{note}\n"))
    assert out["frozen_split_note_problems"] == [] and out["unverified_quantitative_claims"] == []
    assert out["context_problems"] == []
    assert N.frozen_split_values({"analyses": {}}) is None


# ---------------------------------------------------------------- p3-live-003 statements (item 4)

P3_LIVE_003_PRODUCTIONS = ("The clean dataset contains 1,538 eligible snapshots across four stages (S0-S3) "
                           "from 400 productions.")
P3_LIVE_003_ATTRIBUTION = ("The test minus train overrun-prevalence difference and its 95% bootstrap interval are "
                           "reported in the frozen_split_prevalence_difference metric from the handoff gate "
                           "assessment.")


def _problems(ctx, ws, text):
    return N.context_problems(text, E.load_stats(ws), ctx.contract(), ctx.frozen_counts(), ctx.log())


def test_the_eligible_snapshots_come_from_397_productions_not_400(run_ws):
    ws, ctx, tools = run_ws
    assert E.load_stats(ws)["analyses"]["descriptive_statistics"]["productions"] == 397
    found = _problems(ctx, ws, P3_LIVE_003_PRODUCTIONS)
    assert len(found) == 1 and "'400 productions'" in found[0] and "397 productions" in found[0]
    assert _problems(ctx, ws, "The clean dataset contains 1,538 eligible snapshots from 397 productions.") == []
    assert _problems(ctx, ws, "The 1,538 eligible snapshots come from 397 of the 400 productions.") == []
    assert _problems(ctx, ws, "Among the 400 productions, 375 were fully eligible.") == []
    out = json.loads(tools["write_insights"].run(markdown=f"## Observations\n\n{P3_LIVE_003_PRODUCTIONS}\n"))
    assert len(out["context_problems"]) == 1                    # the feedback reaches the agent


def test_frozen_split_figures_must_be_attributed_to_the_bootstrap_analysis(run_ws):
    ws, ctx, _ = run_ws
    found = _problems(ctx, ws, P3_LIVE_003_ATTRIBUTION)
    assert len(found) == 1 and "not from the handoff gate" in found[0]
    assert _problems(ctx, ws, "The difference and its 95% bootstrap interval come from the "
                              "frozen_split_prevalence_difference analysis in stats.json.") == []
    assert _problems(ctx, ws, "The handoff gate passed all hard checks.") == []


@pytest.mark.parametrize("sentence", [
    "Invalid payments were excluded because they had no matching commitment.",
    "Fifteen payments were excluded due to currency errors.",
    "Payments were excluded because they were future-dated.",
    "Invalid payments with missing supplier codes were excluded.",
    "Snapshots were excluded because of out-of-range amounts.",
])
def test_an_invented_exclusion_reason_is_flagged(run_ws, sentence):
    ws, ctx, _ = run_ws
    found = _problems(ctx, ws, sentence)
    assert len(found) == 1 and found[0].startswith("exclusion reason"), found


@pytest.mark.parametrize("sentence", [
    "Invalid payments with out-of-range amounts were excluded.",            # p3-live-003: recorded, correct
    "Exact duplicates were removed from commitments, manual_costs, and payments.",
    "Sixty-two snapshots were excluded under Policy C: 50 due to missing commitment category values and 12 due "
    "to undated manual costs.",
])
def test_a_recorded_exclusion_reason_is_not_flagged(run_ws, sentence):
    ws, ctx, _ = run_ws
    assert _problems(ctx, ws, sentence) == []


def test_the_recorded_payment_reason_comes_from_the_cleaning_log(run_ws):
    _, ctx, _ = run_ws
    log = ctx.log()
    reasons = {p for i in log["issues"] if i["table"] == "payments" for p in i["problems"]}
    assert reasons == {"out_of_range:amount_gross"} and len(log["excluded_payments"]) == 15


# ---------------------------------------------------------------- agent instructions (R3)

def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_authoring_instructions_state_the_rules():
    _, tasks = load_config()
    for name in ("explore_and_report", "contract_and_insights"):
        desc = _flat(tasks[name]["description"])
        assert "There is no human in the loop: nobody will read or answer a question." in desc, name
        assert "Never end your task by asking for information or waiting for input" in desc, name
        assert "Do not invent explanations, causes, reasons or numerical relationships" in desc, name
        assert "until every list is empty" in desc, name
    desc = _flat(tasks["contract_and_insights"]["description"])
    assert "copied character for character" in desc
    for t in INSIGHT_SECTIONS:
        assert f"## {t}" in tasks["contract_and_insights"]["description"], t
    assert "frozen_split_values (copy its write_as text)" in desc and "never to the handoff gate" in desc
    assert "context_problems" in desc and "Do not finish while any problem remains" in desc
    assert "nine section fields" in _flat(tasks["explore_and_report"]["description"])
    assert "list of {column, description} entries" in desc
