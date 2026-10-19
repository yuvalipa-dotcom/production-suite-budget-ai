"""Offline helpers for the Crew-1 tests. ALL DATA IS SYNTHETIC.

* ``no_network``: a fixture that blocks every socket connection and name lookup.
* ``scripted_runner``: an offline stand-in for the LLM that drives the REAL agent tools
  (through their CrewAI wrappers) in task order, with narrative text built from the computed
  statistics. It makes no LLM call; tests use it to exercise the CLI end to end.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

import crew1  # noqa: F401  (applies the safety environment first)
from crew1.tools import ToolContext, build_tools
from crew1.tools import eda as E
from crew1.tools.workspace import RAW_TABLES


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise NetworkBlocked("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    yield


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def pp(x: float) -> str:
    return f"{x:+.1f} pp"


def eda_sections(stats: dict) -> dict[str, str]:
    a = stats["analyses"]
    d, p, g = a["descriptive_statistics"], a["overrun_prevalence"], a["signed_gap_distribution"]
    s, x = a["frozen_split_prevalence_difference"], a["policy_c_exclusions"]
    r, f, c = a["budget_revision_effects"], a["fee_breakdown"], a["feature_correlations"]
    stage = p["by_stage"]
    top = c["features"][0]
    return {
        "overview": f"The handoff holds {d['snapshots']:,} eligible snapshots from {d['productions']} productions "
                    f"across the 4 stages S0 to S3 (Table 1).",
        "data_quality": f"Policy C excluded {x['ineligible']} of {x['scheduled']:,} scheduled snapshots; "
                        f"{x['eligible']:,} remain eligible. No value was imputed.",
        "prevalence": f"Overall {pct(p['overall']['rate'])} of eligible snapshots overrun "
                      f"({p['overall']['k']}/{p['overall']['n']}), from {pct(stage['S0']['rate'])} at S0 to "
                      f"{pct(stage['S3']['rate'])} at S3, with 95% Wilson intervals shown in Chart 1.",
        "signed_gap": f"The median signed gap is {g['gap']['p50_ils']:,.0f} ILS, and the middle half of "
                      f"snapshots lies between {g['gap']['p25_ils']:,.0f} ILS and {g['gap']['p75_ils']:,.0f} ILS.",
        "budget_revisions": f"Snapshots with no budget revision overrun at "
                            f"{pct(r['by_revisions_to_date']['0']['rate'])}; in the 2+ revisions bucket the rate is "
                            f"{pct(r['by_revisions_to_date']['2+']['rate'])}. This is an association only.",
        "fee_terms": f"Overrun prevalence is {pct(f['by_fee_mode']['percent']['rate'])} under a percentage fee "
                     f"and {pct(f['by_fee_mode']['amount']['rate'])} under a fixed-amount fee.",
        "correlations": f"On the train group ({c['n']:,} snapshots) the strongest Spearman correlation with the "
                        f"signed gap is {top['feature']} at {top['spearman']:.2f}; all correlations are weak.",
        "frozen_split": f"Test minus train prevalence is {pp(s['test_minus_train_pp'])}, with a 95% bootstrap "
                        f"interval [{s['test_minus_train_lo_pp']:+.1f}, {s['test_minus_train_hi_pp']:+.1f}] pp "
                        f"({s['bootstrap_reps']:,} resamples, seed 42). The split is frozen.",
        "limitations": "All data are synthetic. These descriptive results are not evidence of real-world accuracy, "
                       "and no model was trained or evaluated here.",
    }


def insights_md(stats: dict, contract: dict) -> str:
    a = stats["analyses"]
    p, s, x = a["overrun_prevalence"], a["frozen_split_prevalence_difference"], a["policy_c_exclusions"]
    stage = p["by_stage"]
    excl = ", ".join(f"{k}: {v}" for k, v in x["exclusion_counts"].items())
    return f"""# Crew 1 insights (synthetic data)

## Observations (descriptive, from eligible synthetic snapshots)

1. The contract declares {contract['row_count']:,} eligible snapshots (SYN-0001 to SYN-0400, 2024-09-04 onwards).
2. Overall overrun prevalence is {pct(p['overall']['rate'])}; it is {pct(stage['S0']['rate'])} at S0 and \
{pct(stage['S3']['rate'])} at S3 (Chart 1, Table 2).

## Hypotheses for Crew 2 (not predictions)

- H1: stage S3 snapshots may carry more information about the final gap than S0 snapshots.
- H2: budget revisions to date may be associated with the signed gap; Crew 2 can test this on the train group.

## Frozen split note

The test minus train row-level overrun prevalence difference is {pp(s['test_minus_train_pp'])}, with a 95% \
production-level bootstrap interval of [{s['test_minus_train_lo_pp']:+.1f}, {s['test_minus_train_hi_pp']:+.1f}] pp. \
The split is frozen and was not tuned. The difference was assessed as a sampling outcome, independent of the \
outcome mechanism, so the split was not re-drawn or rebalanced.

## Data quality and Policy C exclusions

Policy C excluded {x['ineligible']} of {x['scheduled']:,} scheduled snapshots ({excl}). No value was imputed.

## Limitations

The data are synthetic. There is no real-world validation, and nothing here is evidence of real-world accuracy.
"""


def scripted_runner(corrupt=None, eda_text=None, insights_text=None):
    """An offline runner: real tools, scripted narrative, no LLM. ``corrupt(ws)`` runs before the gate."""

    def run(ws, sources, model, recorder):
        ctx = ToolContext(ws, sources)
        tools = {t.name: t for ts in build_tools(ctx, recorder).values() for t in ts}
        for t in RAW_TABLES:
            tools["load_raw_table"]._run(name=f"raw.{t}")
        tools["detect_defects"]._run()
        tools["apply_policy_c"]._run()
        tools["eligibility_summary"]._run()
        recorder.task(SimpleNamespace(name="ingest_and_clean", agent="Data Ingestion & Cleaning Engineer",
                                      raw="scripted"))
        tools["list_analyses"]._run()
        for aid in E.ANALYSES:
            tools["run_analysis"]._run(analysis_id=aid)
        for cid in E.CHARTS:
            tools["render_chart"]._run(chart_id=cid)
        stats = E.load_stats(ws)
        tools["assemble_eda_report"].run(**(eda_text or eda_sections)(stats))    # validated, as CrewAI calls it
        recorder.task(SimpleNamespace(name="explore_and_report", agent="Exploratory Data Analyst", raw="scripted"))
        tools["build_dataset_contract"].run(column_descriptions=[{"column": "stage",
                                                             "description": "Snapshot stage, S0 to S3."}])
        if corrupt:
            corrupt(ws)
        tools["run_handoff_gate"]._run()
        contract = ctx.contract()
        tools["write_insights"]._run(markdown=(insights_text or insights_md)(stats, contract))
        recorder.task(SimpleNamespace(name="contract_and_insights", agent="Data Contract & Insights Author",
                                      raw="scripted"))
        return {"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "successful_requests": 0},
                "analyses_requested": ctx.analyses_requested, "charts_rendered": ctx.charts_rendered}

    return run
