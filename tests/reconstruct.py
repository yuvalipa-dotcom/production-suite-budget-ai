"""Independent reconstruction of snapshot features and labels from dated records.

Deliberately imports NOTHING from ``generator``: VAT rules, the fee rule, the
label definition and every feature are re-implemented here from the written
specification, and applied only to records dated on or before the snapshot.

Feature coverage
* D1 (ledger-derived): rebuilt from the budget-revision, category-plan,
  commitment and manual-cost ledgers plus the production's fee terms.
* D2 (non-ledger): validated against the production master record and the
  snapshot date.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

VAT_RATE = 0.18
OVERRUN_TOLERANCE = 0.5
WARNING_THRESHOLD = 0.02

D1 = ("guest_count", "budget_pre_vat", "budget_revisions_to_date", "fee_contract_pre_vat",
      "planned_total_pre_vat", "n_categories_planned", "n_categories_committed", "n_categories_pending",
      "n_commitments_active", "n_manual_costs", "manual_costs_pre_vat", "committed_pre_vat",
      "n_categories_over_plan", "over_plan_pre_vat", "n_substitutions_to_date", "no_vat_share",
      "client_direct_share", "current_position_ils", "committed_ratio", "cost_per_guest_pre_vat")
D2 = ("stage", "archetype", "event_month", "lead_time_days", "days_since_contract", "days_to_event",
      "budget_vat_mode", "fee_included", "fee_mode")
FEATURE_RULES = {**{c: "D1" for c in D1}, **{c: "D2" for c in D2}}
LABELS = ("final_cost_pre_vat", "budget_gap_ils", "is_overrun", "is_material_overrun")


def vat_pre(amount: float, mode: str) -> float:
    return amount / (1 + VAT_RATE) if mode == "includes_vat" else amount


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _true(x) -> bool:
    return str(x) == "True"


def index(tables: dict) -> dict:
    """Group ledger rows by production id (records as plain dicts)."""
    idx = {"productions": {r["production_id"]: r for r in tables["productions"].to_dict("records")}}
    for name in ("budget_revisions", "category_plans", "commitments", "manual_costs"):
        g = defaultdict(list)
        for r in tables[name].to_dict("records"):
            g[r["production_id"]].append(r)
        idx[name] = g
    return idx


def _terms(idx: dict, pid: str, as_of: str) -> dict:
    """Budget in effect and contractual fee known at ``as_of`` (prediction-time records only)."""
    prod = idx["productions"][pid]
    revs = [r for r in idx["budget_revisions"][pid] if r["effective_date"] <= as_of]
    current = max(revs, key=lambda r: int(r["revision_no"]))
    budget_entered = float(current["approved_entered_amount"])
    budget_pre = vat_pre(budget_entered, prod["budget_vat_mode"])
    fee_typed = budget_entered * float(prod["fee_value"]) / 100 if prod["fee_mode"] == "percent" \
        else float(prod["fee_value"])
    fee_pre = vat_pre(round(fee_typed + 0.0, 2), prod["fee_vat_mode"])
    fee_in = fee_pre if _true(prod["fee_included"]) else 0.0
    return {"prod": prod, "revs": revs, "current": current, "budget_pre": budget_pre,
            "fee_pre": fee_pre, "fee_in": fee_in}


def snapshot_features(idx: dict, pid: str, stage: str, as_of: str) -> dict:
    """Features only. Must work on an index built WITHOUT any outcome-only column."""
    t = _terms(idx, pid, as_of)
    prod, revs, current = t["prod"], t["revs"], t["current"]
    budget_pre, fee_pre, fee_in = t["budget_pre"], t["fee_pre"], t["fee_in"]

    plans = {r["category"]: vat_pre(float(r["planned_entered"]), r["vat_mode"])
             for r in idx["category_plans"][pid] if r["effective_date"] <= as_of}
    active = [c for c in idx["commitments"][pid]
              if c["selected_date"] <= as_of and (c["cancelled_date"] in ("", None) or c["cancelled_date"] > as_of)]
    manual = [m for m in idx["manual_costs"][pid] if m["created_date"] <= as_of]
    committed = sum((float(c["original_pre_vat"]) for c in active), 0.0) + sum((float(m["pre_vat"]) for m in manual), 0.0)
    by_cat = defaultdict(float)
    for c in active:
        by_cat[c["category"]] += float(c["original_pre_vat"])
    for m in manual:
        by_cat[m["category"]] += float(m["pre_vat"])
    over = [by_cat[c] - p for c, p in plans.items() if by_cat[c] > p]
    committed_cats = {c["category"] for c in active}
    guests = int(current["guest_count"])
    contract, event = prod["contract_date"], prod["event_date"]
    substitutions = sum(1 for c in idx["commitments"][pid]
                        if c["cancelled_date"] not in ("", None) and c["cancelled_date"] <= as_of)

    f = {
        # D2
        "stage": stage,
        "archetype": prod["archetype"],
        "event_month": int(event[5:7]),
        "lead_time_days": _days(contract, event),
        "days_since_contract": _days(contract, as_of),
        "days_to_event": _days(as_of, event),
        "budget_vat_mode": prod["budget_vat_mode"],
        "fee_included": _true(prod["fee_included"]),
        "fee_mode": prod["fee_mode"],
        # D1
        "guest_count": guests,
        "budget_pre_vat": budget_pre,
        "budget_revisions_to_date": sum(1 for r in revs if int(r["revision_no"]) > 0),
        "fee_contract_pre_vat": fee_pre,
        "planned_total_pre_vat": sum(plans.values(), 0.0),
        "n_categories_planned": len(plans),
        "n_categories_committed": len(committed_cats & set(plans)),
        "n_categories_pending": len(set(plans) - committed_cats),
        "n_commitments_active": len(active),
        "n_manual_costs": len(manual),
        "manual_costs_pre_vat": sum((float(m["pre_vat"]) for m in manual), 0.0),
        "committed_pre_vat": committed,
        "n_categories_over_plan": len(over),
        "over_plan_pre_vat": sum(over, 0.0),
        "n_substitutions_to_date": substitutions,
        "no_vat_share": (sum(float(c["original_pre_vat"]) for c in active if c["vat_mode"] == "no_vat") / committed
                         if committed > 0 else 0.0),
        "client_direct_share": (sum(float(c["original_pre_vat"]) for c in active
                                    if c["payment_responsibility"] == "client_direct") / committed
                                if committed > 0 else 0.0),
        "current_position_ils": committed + fee_in - budget_pre,
        "committed_ratio": (committed + fee_in) / budget_pre,
        "cost_per_guest_pre_vat": committed / guests if guests > 0 else 0.0,
    }
    return f


def snapshot_labels(idx: dict, pid: str, as_of: str) -> dict:
    """Labels (approved definition A2): verified final costs (outcome records) plus the included
    contractual fee known at the snapshot, against the approved budget in effect at the snapshot."""
    t = _terms(idx, pid, as_of)
    final_cost = (sum((float(c["final_pre_vat"]) for c in idx["commitments"][pid]
                       if c["cancelled_date"] in ("", None)), 0.0)
                  + sum((float(m["pre_vat"]) for m in idx["manual_costs"][pid]), 0.0))
    gap = final_cost + t["fee_in"] - t["budget_pre"]
    return {"final_cost_pre_vat": final_cost, "budget_gap_ils": gap,
            "is_overrun": gap > OVERRUN_TOLERANCE,
            "is_material_overrun": gap > WARNING_THRESHOLD * t["budget_pre"]}


def features_at(idx: dict, pid: str, stage: str, as_of: str) -> dict:
    """Features and labels together (convenience)."""
    return {**snapshot_features(idx, pid, stage, as_of), **snapshot_labels(idx, pid, as_of)}


# Declared source columns (table.column) of every feature. Checked against the column-role
# contract: no outcome-only column may appear here (P1-R2, R3.8c).
_COMMIT_ACTIVE = {"commitments.selected_date", "commitments.cancelled_date", "commitments.original_pre_vat"}
_MANUAL_SEEN = {"manual_costs.created_date", "manual_costs.pre_vat"}
_REV = {"budget_revisions.effective_date", "budget_revisions.revision_no"}
_TERMS = _REV | {"budget_revisions.approved_entered_amount", "productions.budget_vat_mode"}
_FEE = _TERMS | {"productions.fee_mode", "productions.fee_value", "productions.fee_vat_mode",
                 "productions.fee_included"}
_PLANS = {"category_plans.category", "category_plans.planned_entered", "category_plans.vat_mode",
          "category_plans.effective_date"}
_CATS = _PLANS | _COMMIT_ACTIVE | _MANUAL_SEEN | {"commitments.category", "manual_costs.category"}
FEATURE_SOURCES = {
    "stage": {"snapshot_schedule.stage"},
    "archetype": {"productions.archetype"},
    "event_month": {"productions.event_date"},
    "lead_time_days": {"productions.contract_date", "productions.event_date"},
    "days_since_contract": {"productions.contract_date", "snapshot_schedule.as_of_date"},
    "days_to_event": {"productions.event_date", "snapshot_schedule.as_of_date"},
    "budget_vat_mode": {"productions.budget_vat_mode"},
    "fee_included": {"productions.fee_included"},
    "fee_mode": {"productions.fee_mode"},
    "guest_count": _REV | {"budget_revisions.guest_count"},
    "budget_pre_vat": _TERMS,
    "budget_revisions_to_date": _REV,
    "fee_contract_pre_vat": _FEE,
    "planned_total_pre_vat": _PLANS,
    "n_categories_planned": _PLANS,
    "n_categories_committed": _PLANS | _COMMIT_ACTIVE | {"commitments.category"},
    "n_categories_pending": _PLANS | _COMMIT_ACTIVE | {"commitments.category"},
    "n_commitments_active": _COMMIT_ACTIVE,
    "n_manual_costs": _MANUAL_SEEN,
    "manual_costs_pre_vat": _MANUAL_SEEN,
    "committed_pre_vat": _COMMIT_ACTIVE | _MANUAL_SEEN,
    "n_categories_over_plan": _CATS,
    "over_plan_pre_vat": _CATS,
    "n_substitutions_to_date": {"commitments.cancelled_date"},
    "no_vat_share": _COMMIT_ACTIVE | _MANUAL_SEEN | {"commitments.vat_mode"},
    "client_direct_share": _COMMIT_ACTIVE | _MANUAL_SEEN | {"commitments.payment_responsibility"},
    "current_position_ils": _COMMIT_ACTIVE | _MANUAL_SEEN | _FEE,
    "committed_ratio": _COMMIT_ACTIVE | _MANUAL_SEEN | _FEE,
    "cost_per_guest_pre_vat": _COMMIT_ACTIVE | _MANUAL_SEEN | _REV | {"budget_revisions.guest_count"},
}


def as_csv_text(value) -> str:
    """Render a value exactly as the dataset CSV does (fixed 2-decimal floats)."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
