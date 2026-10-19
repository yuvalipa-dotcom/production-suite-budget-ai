"""Runtime leakage checks for Crew 2 (docs/p4_plan.md section f). ALL DATA IS SYNTHETIC.

These checks run on the cleaned Policy C ledgers (never on the committed clean snapshots
or the ground truth) and are also exercised by ``tests/test_crew2_leakage.py``:

* outcome-only columns never reach the model input (declared trace and dynamic trace);
* every snapshot feature is as-of: mutating or adding any ledger record dated AFTER the
  snapshot leaves its features unchanged, and the same change dated ON the snapshot day
  moves them (two-way proof); outcome-only values never move a feature;
* no production id is in two split groups or two folds, and the groups and folds equal
  ``contracts/frozen_split.json``.
"""

from __future__ import annotations

import copy
from datetime import date, timedelta

import pandas as pd

from gate import policy_c as P
from gate.checks import LEDGERS
from gate.reconstruct import FEATURE_SOURCES

from .features import FEATURE_SETS, outcome_only_columns

DATE_COLUMN = {"budget_revisions": "effective_date", "category_plans": "effective_date",
               "commitments": "selected_date", "manual_costs": "created_date"}
OUTCOME_RAW = {"commitments": ("final_pre_vat",), "productions": ("financial_close_date",)}


def _next_day(d: str, days: int = 1) -> str:
    return (date.fromisoformat(d) + timedelta(days=days)).isoformat()


def cleaned_ledgers(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, list[dict]], P.PolicyCResult]:
    result = P.apply_policy_c(raw)
    return {n: result.tables[n].to_dict("records") for n in LEDGERS}, result


def features_of(ledgers: dict[str, list[dict]], pid: str, stage: str, as_of: str) -> dict:
    return P.snapshot_features(P.index({n: pd.DataFrame(v) for n, v in ledgers.items()}), pid, stage, as_of)


def features_for(ledgers: dict[str, list[dict]], keys: list[tuple[str, str, str]]) -> dict:
    ix = P.index({n: pd.DataFrame(v) for n, v in ledgers.items()})
    return {(p, s): P.snapshot_features(ix, p, s, a) for p, s, a in keys}


# ---------------------------------------------------------------- declared and dynamic trace

def declared_trace() -> dict:
    outcome = outcome_only_columns()
    roles_bad = {f: sorted(src for src in FEATURE_SOURCES[f] if src.split(".", 1)[1] in outcome)
                 for fs in FEATURE_SETS.values() for f in fs}
    return {"feature_sets": {k: sorted(set(v) & outcome) for k, v in FEATURE_SETS.items()},
            "sources_with_outcome_only_column": {f: v for f, v in roles_bad.items() if v},
            "ok": not any(set(v) & outcome for v in FEATURE_SETS.values()) and not any(roles_bad.values())}


def dynamic_trace(ledgers: dict[str, list[dict]], handoff: pd.DataFrame) -> dict:
    """Rebuild every eligible snapshot's features with every outcome-only raw column removed."""
    stripped = {n: [{k: v for k, v in r.items() if k not in OUTCOME_RAW.get(n, ())} for r in rows]
                for n, rows in ledgers.items()}
    keys = list(zip(handoff["production_id"], handoff["stage"], handoff["as_of_date"]))
    full = features_for(ledgers, keys)
    bare = features_for(stripped, keys)
    differing = [k for k in full if full[k] != bare[k]]
    return {"snapshots": len(keys), "differing": len(differing), "ok": not differing}


# ---------------------------------------------------------------- as-of mutations

def only_production(ledgers: dict[str, list[dict]], pid: str) -> dict[str, list[dict]]:
    """The records of one production: a snapshot's features depend on nothing else (``P.index`` groups by it)."""
    return {n: [r for r in rows if r["production_id"] == pid] for n, rows in ledgers.items()}


def _post_dated(ledgers, pid, as_of, on_day: bool, manual_template: dict) -> dict[str, dict[str, list[dict]]]:
    """One mutated copy of the ledgers per mutation kind. ``on_day`` dates every change on ``as_of`` itself
    (it must then move a feature); otherwise every change is dated strictly after ``as_of``."""
    when = as_of if on_day else _next_day(as_of)
    out = {}

    def variant(name, fn):
        led = copy.deepcopy(ledgers)
        fn(led)
        out[name] = led

    def rows(led, table):
        return [r for r in led[table] if r["production_id"] == pid]

    def add_revision(led):
        revs = rows(led, "budget_revisions")
        new = dict(max(revs, key=lambda r: int(r["revision_no"])))
        new |= {"revision_no": str(max(int(r["revision_no"]) for r in revs) + 1), "effective_date": when,
                "approved_entered_amount": f"{float(new['approved_entered_amount']) * 1.5:.2f}",
                "guest_count": str(int(new["guest_count"]) + 50)}
        led["budget_revisions"].append(new)

    def add_plan(led):
        new = dict(rows(led, "category_plans")[0]) | {"category": "zz_mutation_category", "effective_date": when,
                                                       "planned_entered": "99999.00"}
        led["category_plans"].append(new)

    def add_commitment(led):
        new = dict(rows(led, "commitments")[0]) | {"commitment_id": f"{pid}-CZZ", "selected_date": when,
                                                    "cancelled_date": "", "original_pre_vat": "123456.00",
                                                    "original_entered": "123456.00"}
        led["commitments"].append(new)

    def cancel_active(led):
        for r in rows(led, "commitments"):
            if r["selected_date"] <= as_of and r["cancelled_date"] in ("", None):
                r["cancelled_date"] = when
                return
        raise LookupError("no active commitment")

    def add_manual_cost(led):
        new = dict(manual_template)                       # any manual cost row: the column layout only
        new |= {"manual_cost_id": f"{pid}-MZZ", "production_id": pid, "created_date": when, "pre_vat": "7777.00",
                "entered": "7777.00", "category": rows(led, "commitments")[0]["category"], "kind": "unexpected",
                "vat_mode": "excludes_vat"}
        led["manual_costs"].append(new)

    def revise_later_records(led):                    # existing records dated after as_of, revised in place
        if on_day:
            return add_revision(led)
        for table, col in DATE_COLUMN.items():
            for r in rows(led, table):
                if r[col] > as_of:
                    for k in ("approved_entered_amount", "planned_entered", "original_pre_vat", "pre_vat"):
                        if k in r and r[k] not in ("", None):
                            r[k] = f"{float(r[k]) * 3:.2f}"
                    if "guest_count" in r:
                        r["guest_count"] = str(int(r["guest_count"]) + 99)

    variant("budget_revision_added", add_revision)
    variant("category_plan_added", add_plan)
    variant("commitment_added", add_commitment)
    variant("manual_cost_added", add_manual_cost)
    variant("later_records_revised", revise_later_records)
    try:
        variant("active_commitment_cancelled", cancel_active)
    except LookupError:
        pass
    return out


def outcome_mutations(ledgers, pid) -> dict[str, dict[str, list[dict]]]:
    out = {}
    led = copy.deepcopy(ledgers)
    for r in led["commitments"]:
        if r["production_id"] == pid and r.get("final_pre_vat") not in ("", None):
            r["final_pre_vat"] = f"{float(r['final_pre_vat']) * 5 + 1000:.2f}"
    out["final_pre_vat_changed"] = led
    led = copy.deepcopy(ledgers)
    for r in led["productions"]:
        if r["production_id"] == pid:
            r["financial_close_date"] = "1999-01-01"
    out["financial_close_date_changed"] = led
    return out


def as_of_mutations(ledgers: dict[str, list[dict]], keys: list[tuple[str, str, str]], fn=None) -> dict:
    """For each snapshot: post-dated changes and outcome changes leave its features identical; the same changes
    dated on the snapshot day move them (two-way proof). ``fn`` replaces the feature builder (tests use it to
    prove that a deliberately leaky builder is caught)."""
    fn = fn or features_of
    template = ledgers["manual_costs"][0]
    report = {"snapshots": len(keys), "post_dated_moved": [], "outcome_moved": [], "on_day_not_moved": [],
              "mutations_checked": 0}
    for pid, stage, as_of in keys:
        own = only_production(ledgers, pid)
        base = fn(own, pid, stage, as_of)
        if base != features_of(ledgers, pid, stage, as_of) and fn is features_of:
            raise AssertionError("features depend on another production's records")
        for name, led in _post_dated(own, pid, as_of, False, template).items():
            report["mutations_checked"] += 1
            if fn(led, pid, stage, as_of) != base:
                report["post_dated_moved"].append(f"{pid}/{stage}/{name}")
        for name, led in outcome_mutations(own, pid).items():
            report["mutations_checked"] += 1
            if fn(led, pid, stage, as_of) != base:
                report["outcome_moved"].append(f"{pid}/{stage}/{name}")
        for name, led in _post_dated(own, pid, as_of, True, template).items():
            report["mutations_checked"] += 1
            if fn(led, pid, stage, as_of) == base:
                report["on_day_not_moved"].append(f"{pid}/{stage}/{name}")
    report["ok"] = not (report["post_dated_moved"] or report["outcome_moved"] or report["on_day_not_moved"])
    return report


def sample_keys(handoff: pd.DataFrame, n_per_stage: int = 10) -> list[tuple[str, str, str]]:
    """A deterministic sample: the first ``n_per_stage`` eligible snapshots of each stage, by production id."""
    out = []
    for _, g in sorted(handoff.groupby("stage")):
        g = g.sort_values("production_id").head(n_per_stage)
        out += list(zip(g["production_id"], g["stage"], g["as_of_date"]))
    return out


# ---------------------------------------------------------------- groups

def group_checks(train: pd.DataFrame, test: pd.DataFrame, demo: pd.DataFrame, frozen: dict) -> dict:
    groups = {"train": set(train["production_id"]), "test": set(test["production_id"]),
              "demo": set(demo["production_id"])}
    overlaps = {f"{a}&{b}": sorted(groups[a] & groups[b])
                for a, b in (("train", "test"), ("train", "demo"), ("test", "demo"))}
    fold_of = train.groupby("production_id")["cv_fold"].nunique()
    expected = {r["production_id"]: (r["split"], int(r["cv_fold"])) for r in frozen["productions"]}
    mismatch = sorted(p for s, ids in groups.items() for p in ids if expected[p][0] != s)
    fold_mismatch = sorted(p for p, f in zip(train["production_id"], train["cv_fold"]) if expected[p][1] != int(f))
    return {"overlaps": overlaps, "productions_in_two_folds": sorted(fold_of[fold_of > 1].index),
            "split_mismatch_with_frozen": mismatch, "fold_mismatch_with_frozen": sorted(set(fold_mismatch)),
            "ok": not any(overlaps.values()) and not (fold_of > 1).any() and not mismatch and not fold_mismatch}
