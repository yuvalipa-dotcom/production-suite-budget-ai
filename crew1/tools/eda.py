"""A2 tools: a FIXED menu of descriptive analyses over the tool-built clean_data.csv,
charts rendered from stats.json only, and the EDA report assembly. Deterministic Python.

Every numeric value in ``stats.json`` carries a unit through its key name (see ``unit_of``),
so narrative claims can be validated against it with their units. Rates are fractions;
differences between rates are percentage points (``_pp``); amounts are ILS pre-VAT (``_ils``).
Nothing here is a model: no fitting, no prediction.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections import Counter
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd

from gate import handoff_spec as H
from gate.reconstruct import OVERRUN_TOLERANCE, VAT_RATE, WARNING_THRESHOLD

from .cleaning import dumps
from .workspace import CHART_IDS, Workspace

GROUPS = ("train", "test", "demo")
STAGES = ("S0", "S1", "S2", "S3")
ARCHETYPES = ("wedding", "bar_bat_mitzvah", "corporate", "conference")
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 2000
Z95 = 1.959963984540054
DIGITS = 6
TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "eda_report.html"

ANALYSES = {
    "descriptive_statistics": "Counts of eligible snapshots and productions by split group, stage and event type; "
                              "mean, median, spread of key amounts and ratios, overall and by stage.",
    "overrun_prevalence": "Row-level is_overrun and is_material_overrun prevalence with Wilson 95% intervals, "
                          "overall and by split group, stage, event type and group x stage.",
    "signed_gap_distribution": "Distribution of the signed budget gap (ILS) and of the gap as a share of the "
                               "as-of budget: quantiles overall and by stage, plus a histogram.",
    "budget_revision_effects": "Overrun prevalence (Wilson 95%) and gap by number of budget revisions to date "
                               "(0, 1, 2+).",
    "feature_correlations": "Spearman correlation of each numeric feature with budget_gap_ils and Pearson "
                            "(point-biserial) correlation with is_overrun, on the train group only.",
    "fee_breakdown": "Overrun prevalence (Wilson 95%) and gap by fee_mode and by fee_included.",
    "frozen_split_prevalence_difference": "Row-level overrun prevalence per frozen group with production-level "
                                          "bootstrap intervals, and the test minus train difference in percentage "
                                          "points with its bootstrap interval.",
    "policy_c_exclusions": "Policy C eligibility: scheduled, eligible and ineligible snapshots, exclusions by "
                           "reason and by stage.",
}

CHARTS = {
    "prevalence_by_stage": ("overrun_prevalence", "Overrun prevalence by stage (Wilson 95% interval)"),
    "prevalence_by_archetype": ("overrun_prevalence", "Overrun prevalence by event type (Wilson 95% interval)"),
    "split_prevalence": ("frozen_split_prevalence_difference",
                         "Overrun prevalence by frozen group (production-level bootstrap 95% interval)"),
    "gap_histogram": ("signed_gap_distribution", "Signed gap as a share of the as-of budget"),
    "gap_by_stage": ("signed_gap_distribution", "Signed gap (ILS) by stage: 5th-95th and 25th-75th percentiles"),
    "revision_effects": ("budget_revision_effects", "Overrun prevalence by budget revisions to date"),
    "fee_breakdown": ("fee_breakdown", "Overrun prevalence by fee terms"),
    "feature_correlations": ("feature_correlations", "Spearman correlation with the signed gap (train group)"),
}
assert tuple(CHARTS) == CHART_IDS

SECTIONS = {
    "overview": ("Overview", ("descriptive_statistics",), ()),
    "data_quality": ("Data quality and Policy C exclusions", ("policy_c_exclusions",), ()),
    "prevalence": ("Overrun prevalence", ("overrun_prevalence",), ("prevalence_by_stage", "prevalence_by_archetype")),
    "signed_gap": ("Signed budget gap", ("signed_gap_distribution",), ("gap_histogram", "gap_by_stage")),
    "budget_revisions": ("Budget revisions", ("budget_revision_effects",), ("revision_effects",)),
    "fee_terms": ("Fee mode and fee inclusion", ("fee_breakdown",), ("fee_breakdown",)),
    "correlations": ("Feature correlations (train group)", ("feature_correlations",), ("feature_correlations",)),
    "frozen_split": ("Frozen split prevalence difference", ("frozen_split_prevalence_difference",),
                     ("split_prevalence",)),
    "limitations": ("Limitations", (), ()),
}
MIN_CHARTS = 4

DEFINITIONS = {                       # documented constants (docs/raw_data_dictionary.md, gate/reconstruct.py)
    "vat_rate": VAT_RATE,
    "overrun_tolerance_ils": OVERRUN_TOLERANCE,
    "material_threshold_share": WARNING_THRESHOLD,
    "s1_lead_time_share": 0.35,
    "s2_lead_time_share": 0.65,
    "s3_before_event_days": 14,
    "n_stages": len(STAGES),
    "n_groups": len(GROUPS),
    "n_archetypes": len(ARCHETYPES),
    "bootstrap_reps": BOOTSTRAP_REPS,
}

# ---------------------------------------------------------------- units

COUNT_KEYS = {"n", "k", "snapshots", "productions", "scheduled", "eligible", "ineligible", "bootstrap_reps",
              "rows"}
NON_CLAIM_KEYS = {"seed"}             # metadata: never a source for a quantitative claim
SUFFIX_UNITS = (("_ils", "ILS"), ("_pp", "pp"), ("_days", "days"), ("_share", "fraction"), ("_rate", "fraction"),
                ("_ratio", "ratio"), ("_value", "number"), ("_counts", "count"), ("_count", "count"))


def unit_of(key: str, parent: str | None) -> str | None:
    """Unit of a numeric leaf from its key (or its parent key for maps and lists).
    Returns None for metadata that is never a claim source; raises for an unclassified key."""
    for k in (key, parent):
        if k is None:
            continue
        if k in NON_CLAIM_KEYS:
            return None
        if k in COUNT_KEYS or k.startswith("n_"):
            return "count"
        if k in ("rate", "lo", "hi"):
            return "fraction"
        if k in ("spearman", "pearson"):
            return "ratio"
        for suffix, unit in SUFFIX_UNITS:
            if k.endswith(suffix):
                return unit
    raise ValueError(f"stats key without a unit: {parent}.{key}")


def numeric_leaves(obj, key: str | None = None, parent: str | None = None):
    """Yield (value, unit) for every numeric leaf of a stats or contract structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, str(k), key)
    elif isinstance(obj, list):
        for v in obj:
            yield from numeric_leaves(v, key, parent)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if isinstance(obj, float) and not math.isfinite(obj):
            return
        unit = unit_of(key, parent)
        if unit:
            yield float(obj), unit


# ---------------------------------------------------------------- helpers

def _r(x) -> float | None:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), DIGITS)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def _prev(d: pd.DataFrame, col: str = "is_overrun") -> dict:
    k, n = int(d[col].sum()), int(len(d))
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": _r(k / n) if n else None, "lo": _r(lo), "hi": _r(hi)}


def _q(s: pd.Series, suffix: str) -> dict:
    qs = np.quantile(s.to_numpy(dtype=float), [0.05, 0.25, 0.5, 0.75, 0.95])
    out = {f"p{int(p * 100):02d}{suffix}": _r(v) for p, v in zip((0.05, 0.25, 0.5, 0.75, 0.95), qs)}
    out |= {f"mean{suffix}": _r(s.mean()), f"std{suffix}": _r(s.std(ddof=1))}
    return out


UNIT_SUFFIX = {"ILS pre-VAT": "_ils", "ratio": "_ratio", "days": "_days", "people": "_value", "": "_value"}


def load_clean(ws: Workspace) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(ws.read_output("out.clean_data")), dtype=str, keep_default_na=False)
    for c, (dtype, _, _) in H.COLUMNS.items():
        if dtype in ("integer", "number"):
            df[c] = pd.to_numeric(df[c])
        elif dtype == "boolean":
            df[c] = df[c] == "True"
    df["gap_share"] = df["budget_gap_ils"] / df["budget_pre_vat"]
    return df


# ---------------------------------------------------------------- the analysis menu

def _descriptive(df, log) -> dict:
    def counts(d):
        return {"snapshots": int(len(d)), "productions": int(d["production_id"].nunique())}

    cols = ("budget_pre_vat", "final_cost_pre_vat", "budget_gap_ils", "committed_ratio", "committed_pre_vat",
            "guest_count", "lead_time_days", "budget_revisions_to_date")
    summary = {}
    for c in cols:
        sfx = UNIT_SUFFIX[H.COLUMNS[c][2]]
        s = df[c]
        summary[c] = {f"mean{sfx}": _r(s.mean()), f"median{sfx}": _r(s.median()), f"std{sfx}": _r(s.std(ddof=1)),
                      f"min{sfx}": _r(s.min()), f"max{sfx}": _r(s.max())}
    by_stage = {st: {c: {f"mean{UNIT_SUFFIX[H.COLUMNS[c][2]]}": _r(df.loc[df.stage == st, c].mean()),
                         f"median{UNIT_SUFFIX[H.COLUMNS[c][2]]}": _r(df.loc[df.stage == st, c].median())}
                     for c in ("committed_ratio", "budget_gap_ils", "n_commitments_active", "days_to_event")}
                for st in STAGES}
    return {"population": "eligible snapshots (Policy C)", **counts(df),
            "by_group": {g: counts(df[df.split == g]) for g in GROUPS},
            "by_stage": {s: counts(df[df.stage == s]) for s in STAGES},
            "by_archetype": {a: counts(df[df.archetype == a]) for a in ARCHETYPES},
            "numeric_summary": summary, "by_stage_numeric": by_stage}


def _prevalence(df, log) -> dict:
    return {"population": "eligible snapshots (Policy C), row level",
            "overall": _prev(df), "overall_material": _prev(df, "is_material_overrun"),
            "by_group": {g: _prev(df[df.split == g]) for g in GROUPS},
            "by_stage": {s: _prev(df[df.stage == s]) for s in STAGES},
            "by_archetype": {a: _prev(df[df.archetype == a]) for a in ARCHETYPES},
            "material_by_stage": {s: _prev(df[df.stage == s], "is_material_overrun") for s in STAGES},
            "by_group_stage": {g: {s: _prev(df[(df.split == g) & (df.stage == s)]) for s in STAGES} for g in GROUPS}}


HIST_EDGES = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # equal width


def _gap(df, log) -> dict:
    share = df["gap_share"].clip(HIST_EDGES[0], HIST_EDGES[-1])
    counts, _ = np.histogram(share.to_numpy(), bins=HIST_EDGES)
    return {"population": "eligible snapshots (Policy C)",
            "gap": _q(df["budget_gap_ils"], "_ils"),
            "gap_as_budget": _q(df["gap_share"], "_share"),
            "by_stage": {s: _q(df.loc[df.stage == s, "budget_gap_ils"], "_ils") for s in STAGES},
            "by_stage_as_budget": {s: _q(df.loc[df.stage == s, "gap_share"], "_share") for s in STAGES},
            "histogram": {"edges_share": HIST_EDGES, "bin_counts": [int(c) for c in counts],
                          "note": "shares outside the first and last edge are counted in the end bins"}}


def _revisions(df, log) -> dict:
    bucket = df["budget_revisions_to_date"].map(lambda r: "0" if r == 0 else ("1" if r == 1 else "2+"))
    out = {}
    for b in ("0", "1", "2+"):
        d = df[bucket == b]
        out[b] = {**_prev(d), "mean_gap_ils": _r(d["budget_gap_ils"].mean()),
                  "median_gap_share": _r(d["gap_share"].median())}
    return {"population": "eligible snapshots (Policy C)", "by_revisions_to_date": out}


def _correlations(df, log) -> dict:
    train = df[df.split == "train"]
    feats = [c for c in H.FEATURES if H.COLUMNS[c][0] in ("integer", "number")]
    rows = []
    for c in feats:
        x = train[c].astype(float)
        if x.nunique() < 2:
            continue
        # Spearman = Pearson correlation of average ranks (tie-aware; no scipy dependency)
        sp = x.rank(method="average").corr(train["budget_gap_ils"].rank(method="average"), method="pearson")
        pb = x.corr(train["is_overrun"].astype(float), method="pearson")
        rows.append({"feature": c, "spearman": _r(sp), "pearson": _r(pb)})
    rows.sort(key=lambda r: (-abs(r["spearman"] or 0), r["feature"]))
    return {"population": "eligible snapshots of the TRAIN group only", "n": int(len(train)),
            "spearman_target": "budget_gap_ils", "pearson_target": "is_overrun (point-biserial)",
            "features": rows}


def _fees(df, log) -> dict:
    def block(d):
        return {**_prev(d), "mean_gap_ils": _r(d["budget_gap_ils"].mean())}

    return {"population": "eligible snapshots (Policy C)",
            "by_fee_mode": {m: block(df[df.fee_mode == m]) for m in ("percent", "amount")},
            "by_fee_included": {str(v): block(df[df.fee_included == v]) for v in (True, False)}}


def _split(df, log) -> dict:
    per = df.groupby(["split", "production_id"])["is_overrun"].agg(["sum", "count"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {}
    for g in GROUPS:
        arr = per.loc[g].to_numpy()
        idx = rng.integers(0, len(arr), size=(BOOTSTRAP_REPS, len(arr)))
        s = arr[idx]
        draws[g] = s[..., 0].sum(axis=1) / s[..., 1].sum(axis=1)
    groups = {}
    for g in GROUPS:
        d = df[df.split == g]
        lo, hi = np.percentile(draws[g], [2.5, 97.5])
        groups[g] = {"n": int(len(d)), "k": int(d["is_overrun"].sum()), "productions": int(d.production_id.nunique()),
                     "rate": _r(d["is_overrun"].mean()), "lo": _r(lo), "hi": _r(hi)}
    diff = draws["test"] - draws["train"]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    obs = groups["test"]["rate"] - groups["train"]["rate"]
    return {"population": "eligible snapshots (Policy C), row level; groups from contracts/frozen_split.json",
            "method": "production-level bootstrap, 95% percentile interval",
            "bootstrap_reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED, "groups": groups,
            "test_minus_train_pp": _r(100 * obs), "test_minus_train_lo_pp": _r(100 * lo),
            "test_minus_train_hi_pp": _r(100 * hi),
            "interval_includes_zero": bool(lo <= 0 <= hi)}


def production_eligibility(log: dict) -> dict:
    """Productions by Policy C eligibility of their snapshots (the narrative's production counts)."""
    prods = Counter(x["production_id"] for x in log["ineligible_snapshots"])
    scheduled, fully_excluded = log["raw_rows"]["productions"], sum(1 for v in prods.values() if v == len(STAGES))
    return {"n_productions_scheduled": scheduled,
            "n_productions_with_eligible_snapshots": scheduled - fully_excluded,
            "n_productions_fully_eligible": scheduled - len(prods),
            "n_productions_partially_eligible": len(prods) - fully_excluded,
            "n_productions_affected": len(prods), "n_productions_fully_excluded": fully_excluded}


def _exclusions(df, log) -> dict:
    by_stage = Counter(x["stage"] for x in log["ineligible_snapshots"])
    return {"scheduled": log["snapshots"]["scheduled"], "eligible": log["snapshots"]["eligible"],
            "ineligible": log["snapshots"]["ineligible"],
            "exclusion_counts": log["exclusion_counts"],
            "ineligible_by_stage_counts": {s: by_stage.get(s, 0) for s in STAGES},
            **production_eligibility(log),
            "n_target_unreliable_productions": len(log["target_unreliable_productions"]),
            "n_invalid_payments_excluded": len(log["excluded_payments"]),
            "n_records_with_problems": len(log["issues"])}


RUNNERS = {"descriptive_statistics": _descriptive, "overrun_prevalence": _prevalence,
           "signed_gap_distribution": _gap, "budget_revision_effects": _revisions,
           "feature_correlations": _correlations, "fee_breakdown": _fees,
           "frozen_split_prevalence_difference": _split, "policy_c_exclusions": _exclusions}
assert set(RUNNERS) == set(ANALYSES)


def compute_analysis(analysis_id: str, df: pd.DataFrame, log: dict) -> dict:
    if analysis_id not in RUNNERS:
        raise KeyError(analysis_id)
    return RUNNERS[analysis_id](df, log)


def load_stats(ws: Workspace) -> dict:
    if ws.has_output("out.stats"):
        return json.loads(ws.read_output("out.stats"))
    return {"synthetic": True, "definitions": DEFINITIONS, "analyses": {}}


def run_analysis(ws: Workspace, analysis_id: str) -> dict:
    """Compute one menu analysis from clean_data.csv and merge it into stats.json (order-independent)."""
    df = load_clean(ws)
    log = json.loads(ws.read_output("out.cleaning_log"))
    result = compute_analysis(analysis_id, df, log)
    stats = load_stats(ws)
    stats["analyses"][analysis_id] = result
    ws.write("out.stats", dumps(stats))
    return result


# ---------------------------------------------------------------- charts (stats.json only)

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")         # validated reference palette, slots 1-3


def _fig(title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "crew1"})
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return plt, fig, ax


def _rate_bars(ax, labels, blocks, color=SERIES[0]):
    rates = [100 * b["rate"] for b in blocks]
    err = [[100 * (b["rate"] - b["lo"]) for b in blocks], [100 * (b["hi"] - b["rate"]) for b in blocks]]
    ax.bar(labels, rates, width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(labels, rates, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for i, (r, b) in enumerate(zip(rates, blocks)):
        ax.annotate(f"{r:.1f}%", (i, b["hi"] * 100), textcoords="offset points", xytext=(0, 3), ha="center",
                    color=INK, fontsize=8)
    ax.set_ylabel("overrun prevalence (%)")
    ax.set_ylim(0, 100)


def render_chart(ws: Workspace, chart_id: str) -> bytes:
    if chart_id not in CHARTS:
        raise KeyError(chart_id)
    needed, title = CHARTS[chart_id]
    stats = load_stats(ws)["analyses"]
    if needed not in stats:
        raise LookupError(needed)
    a = stats[needed]
    plt, fig, ax = _fig(title)
    if chart_id == "prevalence_by_stage":
        _rate_bars(ax, list(STAGES), [a["by_stage"][s] for s in STAGES])
    elif chart_id == "prevalence_by_archetype":
        _rate_bars(ax, list(ARCHETYPES), [a["by_archetype"][x] for x in ARCHETYPES])
    elif chart_id == "split_prevalence":
        _rate_bars(ax, list(GROUPS), [a["groups"][g] for g in GROUPS])
    elif chart_id == "revision_effects":
        b = a["by_revisions_to_date"]
        _rate_bars(ax, [f"{k} revisions" if k != "1" else "1 revision" for k in b], list(b.values()))
    elif chart_id == "fee_breakdown":
        labels = ["fee_mode=percent", "fee_mode=amount", "fee_included", "fee on top"]
        blocks = [a["by_fee_mode"]["percent"], a["by_fee_mode"]["amount"], a["by_fee_included"]["True"],
                  a["by_fee_included"]["False"]]
        _rate_bars(ax, labels, blocks)
    elif chart_id == "gap_histogram":
        h = a["histogram"]
        edges = [100 * e for e in h["edges_share"]]
        widths = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        ax.bar(edges[:-1], h["bin_counts"], width=widths, align="edge", color=SERIES[0], edgecolor=SURFACE,
               linewidth=2)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_xlabel("signed gap as % of the as-of budget (end bins include the tails)")
        ax.set_ylabel("snapshots")
    elif chart_id == "gap_by_stage":
        boxes = [{"label": s, "med": a["by_stage"][s]["p50_ils"], "q1": a["by_stage"][s]["p25_ils"],
                  "q3": a["by_stage"][s]["p75_ils"], "whislo": a["by_stage"][s]["p05_ils"],
                  "whishi": a["by_stage"][s]["p95_ils"], "fliers": []} for s in STAGES]
        ax.bxp(boxes, showfliers=False, patch_artist=True,
               boxprops={"facecolor": SERIES[0], "edgecolor": INK2},
               medianprops={"color": SURFACE, "linewidth": 2}, whiskerprops={"color": INK2},
               capprops={"color": INK2})
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_ylabel("signed gap (ILS pre-VAT)")
    elif chart_id == "feature_correlations":
        rows = [r for r in a["features"] if r["spearman"] is not None][:12][::-1]
        ax.barh([r["feature"] for r in rows], [r["spearman"] for r in rows], color=SERIES[0], edgecolor=SURFACE,
                linewidth=2)
        ax.axvline(0, color=INK, linewidth=1)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Spearman correlation with budget_gap_ils (train group)")
        fig.set_size_inches(6.4, 4.2)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None}, facecolor=SURFACE)
    plt.close(fig)
    data = buf.getvalue()
    ws.write(f"out.chart.{chart_id}", data)
    return data


# ---------------------------------------------------------------- the report

def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _num(x, d=0) -> str:
    return "n/a" if x is None else f"{x:,.{d}f}"


def _table(head, rows) -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="computed"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def _prev_rows(blocks: dict) -> list:
    return [[k, v["n"], v["k"], _pct(v["rate"]), f"[{_pct(v['lo'])}, {_pct(v['hi'])}]"] for k, v in blocks.items()]


def computed_tables(analysis_id: str, a: dict) -> str:
    """Tables rendered from stats.json (tool output, not narrative)."""
    head = ["", "rows", "overruns", "prevalence", "95% interval"]
    if analysis_id == "descriptive_statistics":
        rows = [[g, v["snapshots"], v["productions"]] for g, v in a["by_group"].items()]
        rows += [[s, v["snapshots"], v["productions"]] for s, v in a["by_stage"].items()]
        rows += [[x, v["snapshots"], v["productions"]] for x, v in a["by_archetype"].items()]
        t1 = _table(["subset", "snapshots", "productions"], [["all", a["snapshots"], a["productions"]]] + rows)
        t2 = _table(["column", "mean", "median", "std", "min", "max"],
                    [[c] + [_num(v, 2) for v in s.values()] for c, s in a["numeric_summary"].items()])
        return t1 + t2
    if analysis_id == "policy_c_exclusions":
        rows = [["scheduled", a["scheduled"]], ["eligible", a["eligible"]], ["ineligible", a["ineligible"]]]
        rows += [[f"excluded: {r}", n] for r, n in a["exclusion_counts"].items()]
        rows += [[f"ineligible at {s}", n] for s, n in a["ineligible_by_stage_counts"].items()]
        prods = [[label, a[key]] for key, label in (
            ("n_productions_scheduled", "scheduled"),
            ("n_productions_with_eligible_snapshots", "with eligible snapshots"),
            ("n_productions_fully_eligible", "fully eligible"),
            ("n_productions_partially_eligible", "partially eligible"),
            ("n_productions_fully_excluded", "fully excluded")) if key in a]
        return _table(["snapshots", "count"], rows) + (_table(["productions", "count"], prods) if prods else "")
    if analysis_id == "overrun_prevalence":
        return (_table(head, _prev_rows({"all": a["overall"], **a["by_group"]}))
                + _table(head, _prev_rows(a["by_stage"])) + _table(head, _prev_rows(a["by_archetype"])))
    if analysis_id == "signed_gap_distribution":
        rows = [[s] + [_num(v, 0) for v in q.values()] for s, q in {"all": a["gap"], **a["by_stage"]}.items()]
        return _table(["stage", "p05", "p25", "p50", "p75", "p95", "mean", "std"], rows)
    if analysis_id == "budget_revision_effects":
        b = a["by_revisions_to_date"]
        return _table(head + ["mean gap (ILS)"],
                      [r + [_num(b[r[0]]["mean_gap_ils"], 0)] for r in _prev_rows(b)])
    if analysis_id == "fee_breakdown":
        return _table(head, _prev_rows({f"fee_mode={k}": v for k, v in a["by_fee_mode"].items()}
                                       | {f"fee_included={k}": v for k, v in a["by_fee_included"].items()}))
    if analysis_id == "feature_correlations":
        return _table(["feature", "Spearman vs gap", "Pearson vs is_overrun"],
                      [[r["feature"], _num(r["spearman"], 3), _num(r["pearson"], 3)] for r in a["features"]])
    if analysis_id == "frozen_split_prevalence_difference":
        rows = [[g, v["n"], v["k"], _pct(v["rate"]), f"[{_pct(v['lo'])}, {_pct(v['hi'])}]"]
                for g, v in a["groups"].items()]
        diff = (f"test minus train: {a['test_minus_train_pp']:+.1f} pp, 95% interval "
                f"[{a['test_minus_train_lo_pp']:+.1f}, {a['test_minus_train_hi_pp']:+.1f}] pp")
        return _table(head, rows) + f'<p class="computed">{html.escape(diff)}</p>'
    return ""


def _paragraphs(text: str) -> str:
    parts = [p.strip() for p in str(text).replace("\r\n", "\n").split("\n\n") if p.strip()]
    return "".join(f"<p>{html.escape(p)}</p>" for p in parts)


def assemble_report(ws: Workspace, sections: dict) -> str:
    """Insert the agent's interpretation text into the template with computed tables and the rendered charts."""
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise KeyError(", ".join(unknown))
    stats = load_stats(ws)["analyses"]
    blocks = []
    for sid, (title, analyses, charts) in SECTIONS.items():
        text = sections.get(sid, "")
        parts = [f'<section id="{sid}"><h2>{html.escape(title)}</h2>',
                 f'<div class="narrative" data-section="{sid}">{_paragraphs(text)}</div>']
        for aid in analyses:
            parts.append(computed_tables(aid, stats[aid]) if aid in stats else
                         f'<p class="computed missing">Analysis {html.escape(aid)} was not run.</p>')
        for cid in charts:
            if ws.has_output(f"out.chart.{cid}"):
                b64 = base64.b64encode(ws.read_output(f"out.chart.{cid}")).decode("ascii")
                parts.append(f'<figure><img alt="{html.escape(CHARTS[cid][1])}" src="data:image/png;base64,{b64}">'
                             f"<figcaption>{html.escape(CHARTS[cid][1])}</figcaption></figure>")
        parts.append("</section>")
        blocks.append("".join(parts))
    page = Template(TEMPLATE.read_text(encoding="utf-8")).substitute(sections="\n".join(blocks))
    ws.write("out.eda_report", page)
    return page
