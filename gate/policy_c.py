"""Reference cleaning of the raw export under Policy C (P1-R2).

Independent of ``generator`` and of the ground truth: it uses only what a
consumer can see in the raw tables plus docs/raw_data_dictionary.md and
docs/raw_cleaning_policy.md.

Policy C
1. Target rule: every non-cancelled commitment's final cost and every manual
   cost amount stay in the target, even when other fields are missing.
2. Target-unreliable safeguard: if a record needed for the target has no usable
   amount, or its cancellation status cannot be determined, the production's
   target is unreliable and every snapshot of it is ineligible.
3. Feature eligibility: a snapshot is ineligible when a feature cannot be
   rebuilt exactly from records available as of the snapshot (a commitment with
   a missing field while it is active; a manual cost without a creation date in
   every snapshot; a manual cost with another missing field once it exists).
4. Payments feed neither target nor features; invalid payment rows are logged
   and excluded from the cleaned payments table, with no eligibility effect.
5. No fabrication: nothing is imputed. Missing stays missing.
6. Groups are not touched here; eligibility is per snapshot only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

VAT_RATE = 0.18
PAYMENT_MAX_GROSS = 1_000_000.0
STAGES = ("S0", "S1", "S2", "S3")

ARCHETYPES = {"wedding", "bar_bat_mitzvah", "corporate", "conference"}
CATEGORIES = {"venue_catering", "music_entertainment", "photo_video", "decor_design", "lighting_sound",
              "attire_beauty", "logistics_transport", "gifts_hospitality", "staffing_security", "staging_av",
              "printing_branding"}
VAT = {"includes_vat", "excludes_vat", "no_vat"}
CODED = {"archetype": ARCHETYPES, "category": CATEGORIES, "vat_mode": VAT, "budget_vat_mode": VAT,
         "fee_vat_mode": VAT, "fee_mode": {"percent", "amount"}, "kind": {"unexpected", "cancellation_fee"},
         "payment_responsibility": {"company", "client_direct"}, "direction": {"out"},
         "stage": set(STAGES)}
DATES = {"contract_date", "event_date", "financial_close_date", "effective_date", "selected_date",
         "cancelled_date", "created_date", "paid_date", "as_of_date"}
OPTIONAL = {"cancelled_date", "replaces_commitment_id"}
KEYS = {
    "productions": ["production_id"], "budget_revisions": ["production_id", "revision_no"],
    "category_plans": ["production_id", "category"], "commitments": ["commitment_id"],
    "manual_costs": ["manual_cost_id"], "payments": ["payment_id"],
    "snapshot_schedule": ["production_id", "stage"],
}
VAT_PAIRS = {"commitments": ("original_entered", "original_pre_vat"), "manual_costs": ("entered", "pre_vat")}
TARGET_AMOUNTS = {"commitments": ("original_entered", "original_pre_vat", "final_pre_vat"),
                  "manual_costs": ("entered", "pre_vat")}
COST_TABLES = ("commitments", "manual_costs")


def _normalize_code(column: str, value: str) -> str | None:
    if column == "stage":
        v = value.upper()
    elif column in ("archetype", "category"):
        v = re.sub(r"[\s\-]+", "_", value.lower())
    else:
        v = value.lower()
    return v if v in CODED[column] else None


def _parse_date(value: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _pid(table: str, rec: dict) -> str:
    return rec["production_id"]


@dataclass
class PolicyCResult:
    tables: dict[str, pd.DataFrame]
    issues: list[dict] = field(default_factory=list)          # the cleaning log
    excluded_payments: set[str] = field(default_factory=set)
    target_reliable: dict[str, bool] = field(default_factory=dict)
    eligibility: pd.DataFrame | None = None


def _clean_record(table: str, rec: dict) -> tuple[dict, list[str]]:
    """Normalise one record; return (record, problems). Problems never cause a value to be filled."""
    rec = {k: (v.strip() if isinstance(v, str) else v) for k, v in rec.items()}
    problems: list[str] = []
    for col, val in rec.items():
        if val == "":
            if col not in OPTIONAL:
                problems.append(f"missing:{col}")
            continue
        if col in CODED:
            norm = _normalize_code(col, val)
            if norm is None:
                problems.append(f"invalid:{col}")
            else:
                rec[col] = norm
        elif col in DATES:
            parsed = _parse_date(val)
            if parsed is None:
                problems.append(f"invalid:{col}")
            else:
                rec[col] = parsed
    if table in VAT_PAIRS:
        entered_col, pre_col = VAT_PAIRS[table]
        if _is_number(rec[entered_col]) and _is_number(rec[pre_col]) and rec.get("vat_mode") in VAT:
            entered, stored = float(rec[entered_col]), float(rec[pre_col])
            expected = round(entered / (1 + VAT_RATE), 2) if rec["vat_mode"] == "includes_vat" else round(entered, 2)
            if abs(stored - expected) > 0.005:
                if rec["vat_mode"] == "includes_vat" and abs(stored - entered) <= 0.005:
                    rec[pre_col] = f"{expected:.2f}"         # gross recorded as pre-VAT; the rate is documented
                else:
                    problems.append(f"inconsistent:{pre_col}")
    if table == "payments" and _is_number(rec.get("amount_gross", "")):
        amount = float(rec["amount_gross"])
        if amount < 0 or amount > PAYMENT_MAX_GROSS:
            problems.append("out_of_range:amount_gross")
    return rec, problems


def _target_unusable(table: str, rec: dict, problems: list[str]) -> bool:
    """A cost record whose amount or cancellation status cannot be established."""
    if table not in COST_TABLES:
        return False
    for col in TARGET_AMOUNTS[table]:
        if not _is_number(rec.get(col, "")) or f"inconsistent:{col}" in problems:
            return True
    return "invalid:cancelled_date" in problems


def apply_policy_c(raw: dict[str, pd.DataFrame]) -> PolicyCResult:
    tables: dict[str, pd.DataFrame] = {}
    issues: list[dict] = []
    excluded_payments: set[str] = set()
    unreliable: set[str] = set()
    record_problems: dict[str, dict[str, list[str]]] = {t: {} for t in COST_TABLES}

    for table, df in raw.items():
        kept: dict[str, dict] = {}
        conflicts: set[str] = set()
        for rec in df.to_dict("records"):
            rec, problems = _clean_record(table, rec)
            key = "#".join(rec[k] for k in KEYS[table])
            if key in kept:
                if kept[key] != rec:
                    conflicts.add(key)                       # same key, different content: not a duplicate
                continue                                     # exact duplicate (after normalisation): dropped
            kept[key] = rec
            if not problems:
                continue
            issues.append({"table": table, "record_id": key, "problems": problems})
            if table == "payments":
                excluded_payments.add(key)
            elif table in COST_TABLES:
                record_problems[table][key] = problems
                if _target_unusable(table, rec, problems):
                    unreliable.add(_pid(table, rec))
            else:
                unreliable.add(rec.get("production_id", ""))  # master/plan/revision/schedule issue
        for key in conflicts:
            issues.append({"table": table, "record_id": key, "problems": ["conflicting_duplicate"]})
            if table == "payments":
                excluded_payments.add(key)
            else:
                unreliable.add(kept[key].get("production_id", ""))
        rows = [kept[k] for k in sorted(kept) if not (table == "payments" and k in excluded_payments)]
        tables[table] = pd.DataFrame(rows, columns=list(df.columns))

    pids = sorted(tables["productions"]["production_id"])
    target_reliable = {p: p not in unreliable for p in pids}

    schedule = {(r["production_id"], r["stage"]): r["as_of_date"] for r in tables["snapshot_schedule"].to_dict("records")}
    reasons: dict[tuple[str, str], set[str]] = {}

    def mark(pid: str, stages, reason: str) -> None:
        for s in stages:
            reasons.setdefault((pid, s), set()).add(reason)

    for rec in tables["commitments"].to_dict("records"):
        probs = record_problems["commitments"].get(rec["commitment_id"])
        if not probs:
            continue
        pid = rec["production_id"]
        selected, cancelled = rec["selected_date"], rec["cancelled_date"]
        if "missing:selected_date" in probs or "invalid:selected_date" in probs:
            mark(pid, STAGES, "commitment_undated")
            continue
        fields = sorted({p.split(":", 1)[1] for p in probs} - {"cancelled_date"})
        active = [s for s in STAGES
                  if selected <= schedule[(pid, s)] and (cancelled == "" or cancelled > schedule[(pid, s)])]
        for f in fields:
            mark(pid, active, f"commitment_{f}_missing")
    for rec in tables["manual_costs"].to_dict("records"):
        probs = record_problems["manual_costs"].get(rec["manual_cost_id"])
        if not probs:
            continue
        pid = rec["production_id"]
        if "missing:created_date" in probs or "invalid:created_date" in probs:
            mark(pid, STAGES, "manual_cost_undated")
            continue
        existing = [s for s in STAGES if rec["created_date"] <= schedule[(pid, s)]]
        for f in sorted({p.split(":", 1)[1] for p in probs}):
            mark(pid, existing, f"manual_cost_{f}_missing")
    for pid in pids:
        if not target_reliable[pid]:
            mark(pid, STAGES, "target_unreliable")

    elig = pd.DataFrame([{"production_id": p, "stage": s, "as_of_date": schedule[(p, s)],
                          "eligible": (p, s) not in reasons,
                          "reason": "+".join(sorted(reasons.get((p, s), set())))}
                         for p in pids for s in STAGES])
    return PolicyCResult(tables=tables, issues=issues, excluded_payments=excluded_payments,
                         target_reliable=target_reliable, eligibility=elig)


def target_records_dropped(raw: dict[str, pd.DataFrame], cleaned: dict[str, pd.DataFrame]) -> list[str]:
    """Policy A guard: cost records with a present amount that are missing from the cleaned target ledgers."""
    dropped = []
    for table in COST_TABLES:
        key = KEYS[table][0]
        present = {r[key].strip() for r in raw[table].to_dict("records")
                   if all(_is_number(r[c].strip()) for c in TARGET_AMOUNTS[table])}
        dropped += sorted(present - set(cleaned[table][key]))
    return dropped

# Re-export the independently specified snapshot reconstruction.  Keeping this
# import here gives gate callers one Policy-C surface while the original test
# modules remain backwards compatible.
from .reconstruct import (D1, D2, FEATURE_RULES, FEATURE_SOURCES, LABELS,
                          as_csv_text, features_at, index, snapshot_features,
                          snapshot_labels)
