"""Reference application of the documented repair policy to the raw export.

Independent of ``generator`` and of the defect log: it uses only the rules a
consumer can derive from docs/raw_data_dictionary.md. A deterministic repair is
applied only when the raw data itself carries enough evidence; otherwise the
record is excluded. Nothing is imputed.
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd

VAT_RATE = 0.18
PAYMENT_MAX_GROSS = 1_000_000.0

ARCHETYPES = {"wedding", "bar_bat_mitzvah", "corporate", "conference"}
CATEGORIES = {"venue_catering", "music_entertainment", "photo_video", "decor_design", "lighting_sound",
              "attire_beauty", "logistics_transport", "gifts_hospitality", "staffing_security", "staging_av",
              "printing_branding"}
CODED = {"archetype": ARCHETYPES, "category": CATEGORIES,
         "vat_mode": {"includes_vat", "excludes_vat", "no_vat"}, "budget_vat_mode": {"includes_vat", "excludes_vat", "no_vat"},
         "fee_vat_mode": {"includes_vat", "excludes_vat", "no_vat"}, "fee_mode": {"percent", "amount"},
         "kind": {"unexpected", "cancellation_fee"}, "payment_responsibility": {"company", "client_direct"},
         "direction": {"out"}, "stage": {"S0", "S1", "S2", "S3"}}
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


def repair_table(table: str, raw: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    """Return (repaired rows, one per key; keys excluded under the documented rules)."""
    rows, excluded = [], set()
    for rec in raw.to_dict("records"):
        rec = {k: (v.strip() if isinstance(v, str) else v) for k, v in rec.items()}
        bad = False
        for col, val in rec.items():
            if val == "" and col not in OPTIONAL:
                bad = True                                   # missing required value: exclude, never impute
            elif col in CODED and val != "":
                norm = _normalize_code(col, val)
                bad |= norm is None
                rec[col] = norm if norm is not None else val
            elif col in DATES and val != "":
                parsed = _parse_date(val)
                bad |= parsed is None
                rec[col] = parsed if parsed is not None else val
        if not bad and table in VAT_PAIRS:
            entered_col, pre_col = VAT_PAIRS[table]
            entered, stored = float(rec[entered_col]), float(rec[pre_col])
            expected = round(entered / (1 + VAT_RATE), 2) if rec["vat_mode"] == "includes_vat" else round(entered, 2)
            if abs(stored - expected) > 0.005:
                if rec["vat_mode"] == "includes_vat" and abs(stored - entered) <= 0.005:
                    rec[pre_col] = f"{expected:.2f}"          # gross recorded as pre-VAT; rate is documented
                else:
                    bad = True
        if not bad and table == "payments":
            amount = float(rec["amount_gross"])
            bad = amount < 0 or amount > PAYMENT_MAX_GROSS
        key = "#".join(rec[k] for k in KEYS[table])        # key after normalisation
        if bad:
            excluded.add(key)
        rows.append((key, rec))
    out: dict[str, dict] = {}
    conflicts = set()
    for key, rec in rows:
        if key in excluded:
            continue
        if key in out and out[key] != rec:
            conflicts.add(key)                               # same key, different content: not a duplicate
        out.setdefault(key, rec)
    for key in conflicts:
        out.pop(key, None)
    excluded |= conflicts
    repaired = pd.DataFrame([out[k] for k in sorted(out)], columns=list(raw.columns))
    return repaired, excluded
