"""The model card and its charts (deterministic). ALL DATA IS SYNTHETIC.

The card is assembled by code: a fixed header (synthetic-data notice, the demo label, the
target disclosure and the copyright notice), then each section's narrative written by the
evaluator agent, followed by the tables the tool computes from ``metrics.json``. Narrative
blocks are delimited by HTML comments so that validators check only the agent's text.
"""

from __future__ import annotations

import io
import re

SECTIONS = {
    "summary": "Summary",
    "intended_use": "Intended use and out-of-scope use",
    "data_and_target": "Data and target",
    "features": "Features",
    "protocol": "Training and evaluation protocol",
    "cv_results": "Cross-validation results",
    "selection": "Model selection",
    "test_results": "Test result (single evaluation)",
    "secondary_results": "Secondary result: material overrun",
    "limitations": "Limitations",
}
MIN_NARRATIVE = 40
TITLE = "# Model card: budget overrun probability (Crew 2, P4)"
DEMO_LABEL = "experimental / demo prediction"
HEADER = f"""{TITLE}

> **ALL DATA IS SYNTHETIC.** Every prediction of this model is an **{DEMO_LABEL}**. It was trained and
> evaluated only on synthetic productions and is not a validated real-world forecast.
>
> **Target disclosure.** The target `is_overrun` is the frozen Policy C label: the signed budget gap is
> above 0.5 ILS, a rounding tolerance only. It is **not a commercially material overrun**. The 2% material
> overrun (`is_material_overrun`) is reported as a secondary result and never affected model selection.
>
> Copyright (c) 2026 Yuval Lipa. All rights reserved. No open-source license is granted.
"""
REQUIRED_PHRASES = ("ALL DATA IS SYNTHETIC", DEMO_LABEL, "not a commercially material overrun",
                    "All rights reserved")
NARRATIVE_RE = re.compile(r"<!-- narrative:(?P<sid>[a-z_]+) -->\n(?P<text>.*?)\n<!-- /narrative -->", re.S)


def _f(x, digits=4) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def _ms(summary: dict, m: str) -> str:
    s = summary[m]
    return "n/a" if s["mean"] is None else f"{s['mean']:.4f} ± {_f(s['sd'])}"


def cv_table(metrics: dict, role: str) -> str:
    rows = ["| Run | Candidate | Feature set | Target | Brier | PR-AUC (AP) | ROC-AUC | Log loss |",
            "|---|---|---|---|---|---|---|---|"]
    for run_id, r in metrics["cv"].items():
        if r["role"] != role:
            continue
        s = r["summary"]
        rows.append(f"| `{run_id}` | {r['candidate']} | {r['feature_set']} | `{r['target']}` | {_ms(s, 'brier')} | "
                    f"{_ms(s, 'average_precision')} | {_ms(s, 'roc_auc')} | {_ms(s, 'log_loss')} |")
    return "\n".join(rows) + "\n\nMean ± sd across the 5 frozen production-grouped folds (train only).\n"


def cv_stage_table(metrics: dict) -> str:
    rows = ["| Run | " + " | ".join(f"Brier {s}" for s in ("S0", "S1", "S2", "S3")) + " |", "|---|---|---|---|---|"]
    for run_id, r in metrics["cv"].items():
        if r["role"] == "primary":
            rows.append(f"| `{run_id}` | " + " | ".join(_f(r["by_stage"][s]["brier"]["mean"])
                                                       for s in ("S0", "S1", "S2", "S3")) + " |")
    return "\n".join(rows) + "\n"


def selection_table(metrics: dict) -> str:
    sel = metrics["selection"]
    rows = ["| Candidate | Mean CV Brier | Brier minus best | Mean CV PR-AUC | In tie set |", "|---|---|---|---|---|"]
    for c, v in sel["inputs"].items():
        rows.append(f"| {c} | {v['brier']:.4f} | {sel['brier_minus_best'][c]:.4f} | {v['average_precision']:.4f} | "
                    f"{'yes' if c in sel['tie_set'] else 'no'} |")
    return "\n".join(rows) + f"\n\n**Selected:** {sel['selected']} ({sel['decided_by']}).\n"


def _test_rows(block: dict) -> list[str]:
    o, ci = block["overall"], block["bootstrap_ci"]
    rows = []
    for m, label in (("brier", "Brier"), ("log_loss", "Log loss"), ("roc_auc", "ROC-AUC"),
                     ("average_precision", "PR-AUC (AP)")):
        rows.append(f"| {label} | {_f(o[m])} | [{_f(ci[m]['lo'])}, {_f(ci[m]['hi'])}] |")
    for m in ("precision", "recall", "accuracy"):
        rows.append(f"| {m.capitalize()} at p ≥ 0.5 | {_f(o[m])} | |")
    return rows


def test_table(metrics: dict, name: str) -> str:
    t = metrics["test"]
    b = t[name]
    rows = [f"Rows {t['rows']}, productions {t['productions']}, target `{b['target']}`, "
            f"prevalence {_f(b['overall']['prevalence'])}, candidate {b['candidate']}.", "",
            "| Metric | Test value | 95% production-level bootstrap interval |", "|---|---|---|", *_test_rows(b), "",
            "| Stage | Rows | Brier | ROC-AUC | PR-AUC (AP) |", "|---|---|---|---|---|"]
    for s, v in b["by_stage"].items():
        rows.append(f"| {s} | {v['n']} | {_f(v['brier'])} | {_f(v['roc_auc'])} | {_f(v['average_precision'])} |")
    return "\n".join(rows) + f"\n\nExpected calibration error (10 bins): {_f(b['reliability']['ece'])}.\n"


def data_table(metrics: dict) -> str:
    c = metrics["data"]["counts"]
    rows = ["| Group | Rows | Productions | S0 | S1 | S2 | S3 |", "|---|---|---|---|---|---|---|"]
    for g in ("train", "test", "demo"):
        st = c[g]["by_stage"]
        rows.append(f"| {g} | {c[g]['rows']} | {c[g]['productions']} | " +
                    " | ".join(str(st.get(s, 0)) for s in ("S0", "S1", "S2", "S3")) + " |")
    return "\n".join(rows) + "\n"


def features_block(metrics: dict) -> str:
    fs = metrics["data"]["feature_sets"]
    return (f"Primary `F27` ({len(fs['F27'])} features): " + ", ".join(f"`{f}`" for f in fs["F27"]) +
            ".\n\nResearch only `F29` adds `budget_revisions_to_date` and `client_direct_share` (not computable in "
            "the product today).\n")


TABLES = {
    "data_and_target": data_table,
    "features": features_block,
    "cv_results": lambda m: cv_table(m, "primary") + "\n" + cv_stage_table(m) + "\nResearch (`F29`, never used "
                                                                               "for selection):\n\n" + cv_table(m, "research"),
    "selection": selection_table,
    "test_results": lambda m: test_table(m, "primary") + "\n![Reliability on test](charts/reliability_test.png)\n",
    "secondary_results": lambda m: cv_table(m, "secondary") + "\n" + test_table(m, "secondary"),
}


def assemble(sections: dict[str, str], metrics: dict) -> str:
    parts = [HEADER]
    for sid, title in SECTIONS.items():
        text = (sections.get(sid) or "").strip()
        parts.append(f"## {title}\n\n<!-- narrative:{sid} -->\n{text}\n<!-- /narrative -->\n")
        if sid in TABLES:
            parts.append(TABLES[sid](metrics))
    return "\n".join(parts)


def narrative(card: str) -> dict[str, str]:
    return {m["sid"]: m["text"].strip() for m in NARRATIVE_RE.finditer(card)}


# ---------------------------------------------------------------- charts (matplotlib, from metrics.json only)

def render_chart(chart_id: str, metrics: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
    if chart_id == "cv_brier":
        runs = [r for r in metrics["cv"].values() if r["role"] in ("primary", "research")]
        labels = [f"{r['candidate']}\n{r['feature_set']}" for r in runs]
        means = [r["summary"]["brier"]["mean"] for r in runs]
        sds = [r["summary"]["brier"]["sd"] or 0 for r in runs]
        ax.bar(range(len(runs)), means, yerr=sds, color=["#2a6f97" if r["role"] == "primary" else "#a9c5d6"
                                                         for r in runs])
        ax.set_xticks(range(len(runs)), labels)
        ax.set_ylabel("Mean grouped-CV Brier (lower is better)")
        ax.set_title("Grouped CV on train, is_overrun (synthetic)")
    elif chart_id == "reliability_test":
        bins = [b for b in metrics["test"]["primary"]["reliability"]["bins"] if b["n"]]
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        ax.plot([b["mean_p"] for b in bins], [b["observed_rate"] for b in bins], marker="o", color="#2a6f97")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed overrun rate")
        ax.set_title("Reliability on test, selected model (synthetic)")
    else:
        plt.close(fig)
        raise KeyError(chart_id)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()
