"""The scripted model responses for a Crew 1 kickoff (no LLM inference). ALL DATA IS SYNTHETIC.

The turns are those of the approved P3 offline tests (``tests/crew1_mock_anthropic.correct_plans`` with
``tests/crew1_helpers.eda_sections`` / ``insights_md``), copied here because the P3 test files are frozen
and a Flow module must not import from ``tests``. Tool arguments that quote statistics are callables of
the workspace: they read what Crew 1's own tools computed earlier in the same kickoff.
"""

from __future__ import annotations

import json

from crew1.tools import eda as E
from crew1.tools.workspace import RAW_TABLES


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


def plans() -> dict[str, list]:
    """One correct Crew 1 kickoff: every agent batches its tool calls, then gives a (scripted) final answer."""
    return {
        "ingestion_engineer": [[("load_raw_table", {"name": f"raw.{t}"}) for t in RAW_TABLES],
                               [("detect_defects", {})], [("apply_policy_c", {})], [("eligibility_summary", {})],
                               "Ingestion and Policy C cleaning complete; counts are in the tool results."],
        "exploratory_analyst": [[("list_analyses", {})],
                                [("run_analysis", {"analysis_id": a}) for a in E.ANALYSES],
                                [("render_chart", {"chart_id": c}) for c in E.CHARTS],
                                [("assemble_eda_report", lambda ws: eda_sections(E.load_stats(ws)))],
                                "EDA report assembled from the computed statistics."],
        "contract_author": [[("build_dataset_contract", {"column_descriptions": [
                                {"column": "stage", "description": "Snapshot stage, S0 to S3."}]})],
                            [("run_handoff_gate", {})],
                            [("write_insights", lambda ws: {"markdown": insights_md(
                                E.load_stats(ws), json.loads(ws.read_output("out.dataset_contract")))})],
                            "Dataset contract written and the handoff gate run; insights written."],
    }
