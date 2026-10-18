"""Build prediction-time snapshots and production-level tables.

ALL OUTPUT IS SYNTHETIC.

A snapshot describes a production *as of* a stage day using only records that
exist on that day. Final costs, the budget gap and the overrun label are
attached as targets; they are never features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .simulate import Production
from .vat import pre_vat


def stage_days(p: Production) -> dict[str, int]:
    lead = p.lead_time_days
    return {
        "S0": 0,
        "S1": int(C.S1_FRACTION * lead),
        "S2": int(C.S2_FRACTION * lead),
        "S3": lead - C.S3_DAYS_BEFORE_EVENT,
    }


def _date(p: Production, day: int) -> str:
    return str(np.datetime64(C.ANCHOR_DATE) + np.timedelta64(p.contract_offset_days + day, "D"))


def snapshot_row(p: Production, stage: str, day: int) -> dict:
    rate = C.VAT_RATE
    active = [c for c in p.commitments
              if c.selected_day <= day and (c.cancelled_day is None or c.cancelled_day > day)]
    manual = [m for m in p.manual_costs if m.created_day <= day]
    committed = sum(c.original_pre_vat for c in active) + sum(m.pre_vat for m in manual)

    budget_pre = p.budget_pre_vat_at(day)
    fee_pre = p.fee_pre_vat_at(day)
    fee_in_budget = fee_pre if p.fee_included else 0.0

    plans_pre = {c: pre_vat(v, p.category_plan_vat_mode, rate) for c, v in p.category_plans_entered.items()}
    by_cat: dict[str, float] = {}
    for c in active:
        by_cat[c.category] = by_cat.get(c.category, 0.0) + c.original_pre_vat
    for m in manual:
        by_cat[m.category] = by_cat.get(m.category, 0.0) + m.pre_vat
    over = {c: by_cat.get(c, 0.0) - plans_pre[c] for c in plans_pre if by_cat.get(c, 0.0) > plans_pre[c]}
    committed_cats = {c.category for c in active}

    no_vat_share = (sum(c.original_pre_vat for c in active if c.vat_mode == "no_vat") / committed
                    if committed > 0 else 0.0)
    direct_share = (sum(c.original_pre_vat for c in active if c.payment_responsibility == "client_direct")
                    / committed if committed > 0 else 0.0)
    substitutions = sum(1 for c in p.commitments if c.cancelled_day is not None and c.cancelled_day <= day)

    final_cost = p.final_cost_pre_vat()
    gap = final_cost + fee_in_budget - budget_pre
    guests = p.guest_count_at(day)

    return {
        # identifiers and metadata
        "production_id": p.production_id,
        "batch_id": C.BATCH_ID,
        "is_synthetic": True,
        "stage": stage,
        "as_of_date": _date(p, day),
        "event_date": _date(p, p.lead_time_days),
        # prediction-time features
        "archetype": p.archetype,
        "event_month": p.event_month,
        "lead_time_days": p.lead_time_days,
        "days_since_contract": day,
        "days_to_event": p.lead_time_days - day,
        "guest_count": guests,
        "budget_vat_mode": p.budget_vat_mode,
        "budget_pre_vat": budget_pre,
        "budget_revisions_to_date": p.revisions_to_date(day),
        "fee_included": p.fee_included,
        "fee_mode": p.fee_mode,
        "fee_contract_pre_vat": fee_pre,
        "planned_total_pre_vat": sum(plans_pre.values()),
        "n_categories_planned": len(plans_pre),
        "n_categories_committed": len(committed_cats & set(plans_pre)),
        "n_categories_pending": len(set(plans_pre) - committed_cats),
        "n_commitments_active": len(active),
        "n_manual_costs": len(manual),
        "manual_costs_pre_vat": sum(m.pre_vat for m in manual),
        "committed_pre_vat": committed,
        "n_categories_over_plan": len(over),
        "over_plan_pre_vat": sum(over.values()),
        "n_substitutions_to_date": substitutions,
        "no_vat_share": no_vat_share,
        "client_direct_share": direct_share,
        "current_position_ils": committed + fee_in_budget - budget_pre,
        "committed_ratio": (committed + fee_in_budget) / budget_pre,
        "cost_per_guest_pre_vat": committed / guests if guests > 0 else 0.0,
        # targets (known only at financial close)
        "final_cost_pre_vat": final_cost,
        "budget_gap_ils": gap,
        "is_overrun": bool(gap > C.OVERRUN_TOLERANCE_ILS),
        "is_material_overrun": bool(gap > C.WARNING_THRESHOLD * budget_pre),
    }


def build_tables(productions: list[Production]) -> dict[str, pd.DataFrame]:
    snap_rows, prod_rows, comm_rows, man_rows, pay_rows = [], [], [], [], []
    for p in productions:
        for stage, day in stage_days(p).items():
            snap_rows.append(snapshot_row(p, stage, day))
        final_fee = p.fee_pre_vat_at(p.close_day)
        prod_rows.append({
            "production_id": p.production_id,
            "batch_id": C.BATCH_ID,
            "is_synthetic": True,
            "archetype": p.archetype,
            "contract_date": _date(p, 0),
            "event_date": _date(p, p.lead_time_days),
            "financial_close_date": _date(p, p.close_day),
            "budget_vat_mode": p.budget_vat_mode,
            "initial_budget_entered": p.initial_budget_entered,
            "final_budget_entered": p.budget_entered_at(p.close_day),
            "fee_mode": p.fee_mode,
            "fee_value": p.fee_value,
            "fee_vat_mode": p.fee_vat_mode,
            "fee_included": p.fee_included,
            "n_budget_revisions": len(p.revisions),
            "final_cost_pre_vat": p.final_cost_pre_vat(),
            "final_fee_contract_pre_vat": final_fee,
            "fee_concession_pre_vat": p.fee_concession_pre_vat,
            "final_company_profit_pre_vat": final_fee - p.fee_concession_pre_vat,
        })
        for c in p.commitments:
            comm_rows.append({
                "commitment_id": c.commitment_id, "production_id": c.production_id,
                "batch_id": C.BATCH_ID, "is_synthetic": True, "category": c.category,
                "vat_mode": c.vat_mode, "payment_responsibility": c.payment_responsibility,
                "original_entered": c.original_entered, "original_pre_vat": c.original_pre_vat,
                "selected_date": _date(p, c.selected_day),
                "cancelled_date": _date(p, c.cancelled_day) if c.cancelled_day is not None else "",
                "replaces_commitment_id": c.replaces or "",
                "final_pre_vat": c.final_pre_vat,
            })
        for m in p.manual_costs:
            man_rows.append({
                "manual_cost_id": m.manual_cost_id, "production_id": m.production_id,
                "batch_id": C.BATCH_ID, "is_synthetic": True, "category": m.category, "kind": m.kind,
                "vat_mode": m.vat_mode, "entered": m.entered, "pre_vat": m.pre_vat,
                "created_date": _date(p, m.created_day),
            })
        for pay in p.payments:
            pay_rows.append({
                "payment_id": pay.payment_id, "production_id": pay.production_id,
                "commitment_id": pay.commitment_id, "batch_id": C.BATCH_ID, "is_synthetic": True,
                "direction": pay.direction, "amount_gross": pay.amount_gross,
                "paid_date": _date(p, pay.paid_day),
            })
    order = {s: i for i, s in enumerate(C.STAGES)}
    snaps = pd.DataFrame(snap_rows)
    snaps = snaps.assign(_o=snaps["stage"].map(order)).sort_values(["production_id", "_o"]).drop(columns="_o")
    return {
        "snapshots": snaps.reset_index(drop=True),
        "productions": pd.DataFrame(prod_rows).sort_values("production_id").reset_index(drop=True),
        "commitments": pd.DataFrame(comm_rows).sort_values("commitment_id").reset_index(drop=True),
        "manual_costs": pd.DataFrame(man_rows).sort_values("manual_cost_id").reset_index(drop=True),
        "payments": pd.DataFrame(pay_rows).sort_values("payment_id").reset_index(drop=True),
    }
