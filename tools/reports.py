"""Generate the measured sections of the documentation from the committed dataset.

    python tools/reports.py          # rewrite the generated sections in docs/
    python tools/reports.py --check  # exit 1 if any generated section is out of date

Descriptive statistics only. Nothing is trained or fitted.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "synthetic" / "v2"
BOOTSTRAP_SEED = 42
BOOTSTRAP_REPS = 2000
STAGES = ("S0", "S1", "S2", "S3")
GROUPS = ("train", "test", "demo")
ARCHETYPES = ("wedding", "bar_bat_mitzvah", "corporate", "conference")

BEGIN = "<!-- BEGIN GENERATED: {} -->"
END = "<!-- END GENERATED: {} -->"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    snaps = pd.read_csv(DATA / "snapshots.csv")
    prods = pd.read_csv(DATA / "productions.csv")
    return snaps, prods


def _boot_rows(snaps: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Production-level bootstrap of the row-level overrun rate, per group, and train - test."""
    per_prod = snaps.groupby(["split", "production_id"])["is_overrun"].agg(["sum", "count"])
    draws = {}
    for g in GROUPS:
        arr = per_prod.loc[g].to_numpy()
        idx = rng.integers(0, len(arr), size=(BOOTSTRAP_REPS, len(arr)))
        s = arr[idx]
        draws[g] = s[..., 0].sum(axis=1) / s[..., 1].sum(axis=1)
    return draws


def split_report() -> str:
    snaps, prods = load()
    lines = ["| Event type | train productions | train rows | test productions | test rows | demo productions | demo rows |",
             "|---|---|---|---|---|---|---|"]
    for a in ARCHETYPES:
        cells = []
        for g in GROUPS:
            n_p = int(((prods["archetype"] == a) & (prods["split"] == g)).sum())
            n_r = int(((snaps["archetype"] == a) & (snaps["split"] == g)).sum())
            cells += [str(n_p), str(n_r)]
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    tot = []
    for g in GROUPS:
        tot += [str(int((prods["split"] == g).sum())), str(int((snaps["split"] == g).sum()))]
    lines.append("| **total** | " + " | ".join(tot) + " |")

    lines += ["", "### Production-level overrun rate per stage (Wilson 95% interval)", "",
              "| Stage | train | test | demo |", "|---|---|---|---|"]
    for st in STAGES:
        cells = []
        for g in GROUPS:
            d = snaps[(snaps["stage"] == st) & (snaps["split"] == g)]
            k, n = int(d["is_overrun"].sum()), len(d)
            lo, hi = wilson(k, n)
            cells.append(f"{pct(k / n)} ({k}/{n}) [{pct(lo)}, {pct(hi)}]")
        lines.append(f"| {st} | " + " | ".join(cells) + " |")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = _boot_rows(snaps, rng)
    lines += ["", f"### Row-level overrun rate (production-level bootstrap, {BOOTSTRAP_REPS} resamples, "
              f"seed {BOOTSTRAP_SEED}, 95% percentile interval)", "",
              "| Group | Rows | Rate | 95% interval |", "|---|---|---|---|"]
    for g in GROUPS:
        d = snaps[snaps["split"] == g]
        lo, hi = np.percentile(draws[g], [2.5, 97.5])
        lines.append(f"| {g} | {len(d)} | {pct(d['is_overrun'].mean())} | [{pct(lo)}, {pct(hi)}] |")
    diff = draws["test"] - draws["train"]
    obs = snaps.loc[snaps["split"] == "test", "is_overrun"].mean() - snaps.loc[snaps["split"] == "train", "is_overrun"].mean()
    lo, hi = np.percentile(diff, [2.5, 97.5])
    lines += ["", f"Observed test − train difference (row level): **{100 * obs:+.1f} percentage points**, "
              f"95% bootstrap interval [{100 * lo:+.1f}, {100 * hi:+.1f}] percentage points."]
    return "\n".join(lines)


PERMUTATION_SEED = 20261019
PERMUTATIONS = 1000


def split_independence() -> str:
    """Descriptive diagnostics of the frozen split. Not used for any selection or tuning."""
    snaps, prods = load()
    ids = np.array(sorted(prods["production_id"]))
    lab = snaps.pivot(index="production_id", columns="stage", values="is_overrun").loc[ids].astype(float)
    grp = prods.set_index("production_id").loc[ids, "split"].to_numpy()

    def diffs(test, train):
        return [lab[test].to_numpy().mean() - lab[train].to_numpy().mean()] + \
               [lab[test][s].mean() - lab[train][s].mean() for s in STAGES]

    observed = diffs(grp == "test", grp == "train")
    rng = np.random.default_rng(PERMUTATION_SEED)
    sims = []
    for _ in range(PERMUTATIONS):
        perm = rng.permutation(len(ids))
        g = np.empty(len(ids), dtype=object)
        g[perm[:280]], g[perm[280:360]], g[perm[360:]] = "train", "test", "demo"
        sims.append(diffs(g == "test", g == "train"))
    sims = np.array(sims)
    lines = [f"### Alternative random splits ({PERMUTATIONS} draws, seed {PERMUTATION_SEED}, same 280/80/40 sizes)", "",
             "| Test − train overrun difference | Observed | Mean | 2.5th pct | 97.5th pct | Observed percentile |",
             "|---|---|---|---|---|---|"]
    for i, name in enumerate(["row level"] + [f"production level, {s}" for s in STAGES]):
        col = sims[:, i]
        pctl = 100 * ((col < observed[i]).mean() + 0.5 * (col == observed[i]).mean())
        lines.append(f"| {name} | {100 * observed[i]:+.1f} pp | {100 * col.mean():+.1f} | "
                     f"{np.percentile(col, 2.5) * 100:+.1f} | {np.percentile(col, 97.5) * 100:+.1f} | {pctl:.1f} |")
    s0 = snaps[snaps["stage"] == "S0"].set_index("production_id")
    rev = pd.read_csv(DATA / "budget_revisions.csv")
    cov = pd.DataFrame({
        "budget_pre_vat": s0["budget_pre_vat"], "guest_count": s0["guest_count"],
        "lead_time_days": s0["lead_time_days"], "event_month": s0["event_month"],
        "fee_mode = percent": (s0["fee_mode"] == "percent").astype(float),
        "fee_included": s0["fee_included"].astype(float),
        "eventual number of budget revisions": rev.groupby("production_id").size() - 1,
    })
    cov["split"] = prods.set_index("production_id")["split"]
    lines += ["", "### S0 covariate balance, test vs train (standardised mean difference)", "",
              "| Variable | Train mean | Test mean | SMD |", "|---|---|---|---|"]
    for c in cov.columns[:-1]:
        a, b = cov.loc[cov["split"] == "test", c], cov.loc[cov["split"] == "train", c]
        smd = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        lines.append(f"| {c} | {b.mean():,.3f} | {a.mean():,.3f} | {smd:+.3f} |")
    return "\n".join(lines)


def naive_baseline() -> str:
    snaps, _ = load()
    train = snaps[snaps["split"] == "train"]
    lines = ["| Stage | Rows | Prevalence | Precision | Recall | Accuracy |", "|---|---|---|---|---|---|"]
    for st in STAGES:
        d = train[train["stage"] == st]
        y, pred = d["is_overrun"].astype(bool), d["current_position_ils"] > 0
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        precision = pct(tp / (tp + fp)) if tp + fp else "n/a (no positive predictions)"
        recall = pct(tp / (tp + fn)) if tp + fn else "n/a"
        lines.append(f"| {st} | {len(d)} | {pct(y.mean())} | {precision} | {recall} | {pct((pred == y).mean())} |")
    return "\n".join(lines)


def target_audit_numbers() -> str:
    snaps, prods = load()
    rev = pd.read_csv(DATA / "budget_revisions.csv")
    per = rev.groupby("production_id").size() - 1
    dist = per.value_counts().sort_index()
    changes = snaps.groupby("production_id")["is_overrun"].nunique()
    changed = changes[changes > 1].index
    revised = set(prods.loc[prods["n_budget_revisions"] > 0, "production_id"])
    lines = [f"- Productions: {len(prods)} · snapshot rows: {len(snaps)} · budget-revision records: {len(rev)} "
             f"({len(prods)} originals + {int(per.sum())} later revisions).",
             "- Later revisions per production: " + ", ".join(f"{k} → {v} productions" for k, v in dist.items()) + ".",
             f"- Productions whose label changes across stages: **{len(changed)}**; all of them have at least one "
             f"budget revision ({len(set(changed) & revised)} of {len(changed)}).",
             "", "Examples (first three, ordered by id):", ""]
    cols = ["stage", "as_of_date", "budget_pre_vat", "fee_included", "fee_mode", "fee_contract_pre_vat",
            "final_cost_pre_vat", "budget_gap_ils", "is_overrun"]
    for pid in sorted(changed)[:3]:
        d = snaps[snaps["production_id"] == pid][cols]
        lines.append(f"**{pid}**")
        lines.append("")
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for r in d.itertuples(index=False):
            vals = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in r]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines).rstrip()


SECTIONS = {
    "docs/split_report.md": {"split-report": split_report, "split-independence": split_independence},
    "docs/model_selection_rule.md": {"naive-baseline": naive_baseline},
    "docs/target_audit.md": {"target-audit": target_audit_numbers},
}


def render(path: Path, blocks: dict) -> str:
    text = path.read_text(encoding="utf-8")
    for name, fn in blocks.items():
        b, e = BEGIN.format(name), END.format(name)
        start, stop = text.index(b) + len(b), text.index(e)
        text = text[:start] + "\n" + fn() + "\n" + text[stop:]
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = []
    for rel, blocks in SECTIONS.items():
        path = REPO / rel
        new = render(path, blocks)
        if new != path.read_text(encoding="utf-8"):
            stale.append(rel)
            if not args.check:
                path.write_text(new, encoding="utf-8", newline="\n")
    if args.check and stale:
        print("out of date:", ", ".join(stale))
        return 1
    print("updated:" if not args.check else "up to date", ", ".join(stale) if stale else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
