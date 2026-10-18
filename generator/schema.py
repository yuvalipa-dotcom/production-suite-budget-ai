"""Machine-readable column roles for the snapshot table.

role:          feature  -- may be used as a model input
               target   -- an outcome; never a model input
               metadata -- identifiers, provenance, dates; never a model input
available_at:  as_of           -- known at the snapshot's prediction time
               financial_close -- known only after the production is financially closed
"""

from __future__ import annotations

F, T, M = "feature", "target", "metadata"
AS_OF, CLOSE = "as_of", "financial_close"

SNAPSHOT_COLUMNS: dict[str, dict] = {
    "production_id":            {"role": M, "available_at": AS_OF, "type": "string"},
    "batch_id":                 {"role": M, "available_at": AS_OF, "type": "string"},
    "is_synthetic":             {"role": M, "available_at": AS_OF, "type": "boolean"},
    "stage":                    {"role": F, "available_at": AS_OF, "type": "string",
                                 "allowed": ["S0", "S1", "S2", "S3"]},
    "as_of_date":               {"role": M, "available_at": AS_OF, "type": "date"},
    "event_date":               {"role": M, "available_at": AS_OF, "type": "date"},
    "archetype":                {"role": F, "available_at": AS_OF, "type": "string",
                                 "allowed": ["wedding", "bar_bat_mitzvah", "corporate", "conference"]},
    "event_month":              {"role": F, "available_at": AS_OF, "type": "integer", "min": 1, "max": 12},
    "lead_time_days":           {"role": F, "available_at": AS_OF, "type": "integer", "unit": "days"},
    "days_since_contract":      {"role": F, "available_at": AS_OF, "type": "integer", "unit": "days"},
    "days_to_event":            {"role": F, "available_at": AS_OF, "type": "integer", "unit": "days"},
    "guest_count":              {"role": F, "available_at": AS_OF, "type": "integer", "unit": "people"},
    "budget_vat_mode":          {"role": F, "available_at": AS_OF, "type": "string",
                                 "allowed": ["includes_vat", "excludes_vat", "no_vat"]},
    "budget_pre_vat":           {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT"},
    "budget_revisions_to_date": {"role": F, "available_at": AS_OF, "type": "integer"},
    "fee_included":             {"role": F, "available_at": AS_OF, "type": "boolean"},
    "fee_mode":                 {"role": F, "available_at": AS_OF, "type": "string",
                                 "allowed": ["percent", "amount"]},
    "fee_contract_pre_vat":     {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT",
                                 "note": "contractual fee as known at as_of; manager-only in the product"},
    "planned_total_pre_vat":    {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT"},
    "n_categories_planned":     {"role": F, "available_at": AS_OF, "type": "integer"},
    "n_categories_committed":   {"role": F, "available_at": AS_OF, "type": "integer"},
    "n_categories_pending":     {"role": F, "available_at": AS_OF, "type": "integer"},
    "n_commitments_active":     {"role": F, "available_at": AS_OF, "type": "integer"},
    "n_manual_costs":           {"role": F, "available_at": AS_OF, "type": "integer"},
    "manual_costs_pre_vat":     {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT"},
    "committed_pre_vat":        {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT",
                                 "note": "accepted quotes + recorded manual costs; NOT a verified final cost"},
    "n_categories_over_plan":   {"role": F, "available_at": AS_OF, "type": "integer"},
    "over_plan_pre_vat":        {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT"},
    "n_substitutions_to_date":  {"role": F, "available_at": AS_OF, "type": "integer"},
    "no_vat_share":             {"role": F, "available_at": AS_OF, "type": "number", "unit": "ratio"},
    "client_direct_share":      {"role": F, "available_at": AS_OF, "type": "number", "unit": "ratio"},
    "current_position_ils":     {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT",
                                 "note": "committed + fee if included - budget (deterministic actual-overrun level)"},
    "committed_ratio":          {"role": F, "available_at": AS_OF, "type": "number", "unit": "ratio"},
    "cost_per_guest_pre_vat":   {"role": F, "available_at": AS_OF, "type": "number", "unit": "ILS pre-VAT"},
    "final_cost_pre_vat":       {"role": T, "available_at": CLOSE, "type": "number", "unit": "ILS pre-VAT",
                                 "note": "simulated verified final supplier costs + all manual costs"},
    "budget_gap_ils":           {"role": T, "available_at": CLOSE, "type": "number", "unit": "ILS pre-VAT",
                                 "note": "final_cost + (fee_contract if fee_included) - budget, signed"},
    "is_overrun":               {"role": T, "available_at": CLOSE, "type": "boolean",
                                 "note": "budget_gap_ils > 0.5 (rounding tolerance only)"},
    "is_material_overrun":      {"role": T, "available_at": CLOSE, "type": "boolean",
                                 "note": "budget_gap_ils > 2% of budget; a WARNING level, not the overrun definition"},
    "split":                    {"role": M, "available_at": AS_OF, "type": "string",
                                 "allowed": ["train", "test", "demo"]},
    "cv_fold":                  {"role": M, "available_at": AS_OF, "type": "integer",
                                 "note": "0..4 for train productions, -1 for test and demo"},
}

# Any column whose name contains one of these must never be a feature.
FORBIDDEN_FEATURE_SUBSTRINGS = ("final_", "gap", "overrun", "close", "concession", "profit")


def schema_document() -> dict:
    return {
        "table": "snapshots",
        "row_grain": "one row per (production_id, stage)",
        "primary_key": ["production_id", "stage"],
        "currency": "ILS",
        "vat_basis": "pre_vat",
        "synthetic": True,
        "columns": SNAPSHOT_COLUMNS,
    }


def feature_columns() -> list[str]:
    return [c for c, s in SNAPSHOT_COLUMNS.items() if s["role"] == F]
