"""The approved Crew-1 handoff layout: ``clean_data.csv`` plus ``dataset_contract.json``.

The gate validates a delivered contract against this specification, so a
contract that is internally consistent with corrupted data still fails.
One row per eligible ``(production_id, stage)`` snapshot. Every listed column is
required and non-nullable.
"""

from __future__ import annotations

from .reconstruct import D1, D2, LABELS

DATA_FILE = "clean_data.csv"
CONTRACT_FILE = "dataset_contract.json"
PRIMARY_KEY = ["production_id", "stage"]
DTYPES = ("string", "integer", "number", "boolean", "date")

_MONEY = "ILS pre-VAT"
# name: (dtype, role, unit)
COLUMNS: dict[str, tuple[str, str, str]] = {
    "production_id": ("string", "id", ""),
    "stage": ("string", "feature", ""),
    "as_of_date": ("date", "metadata", ""),
    "archetype": ("string", "feature", ""),
    "event_month": ("integer", "feature", ""),
    "lead_time_days": ("integer", "feature", "days"),
    "days_since_contract": ("integer", "feature", "days"),
    "days_to_event": ("integer", "feature", "days"),
    "guest_count": ("integer", "feature", "people"),
    "budget_vat_mode": ("string", "feature", ""),
    "budget_pre_vat": ("number", "feature", _MONEY),
    "budget_revisions_to_date": ("integer", "feature", ""),
    "fee_included": ("boolean", "feature", ""),
    "fee_mode": ("string", "feature", ""),
    "fee_contract_pre_vat": ("number", "feature", _MONEY),
    "planned_total_pre_vat": ("number", "feature", _MONEY),
    "n_categories_planned": ("integer", "feature", ""),
    "n_categories_committed": ("integer", "feature", ""),
    "n_categories_pending": ("integer", "feature", ""),
    "n_commitments_active": ("integer", "feature", ""),
    "n_manual_costs": ("integer", "feature", ""),
    "manual_costs_pre_vat": ("number", "feature", _MONEY),
    "committed_pre_vat": ("number", "feature", _MONEY),
    "n_categories_over_plan": ("integer", "feature", ""),
    "over_plan_pre_vat": ("number", "feature", _MONEY),
    "n_substitutions_to_date": ("integer", "feature", ""),
    "no_vat_share": ("number", "feature", "ratio"),
    "client_direct_share": ("number", "feature", "ratio"),
    "current_position_ils": ("number", "feature", _MONEY),
    "committed_ratio": ("number", "feature", "ratio"),
    "cost_per_guest_pre_vat": ("number", "feature", _MONEY),
    "final_cost_pre_vat": ("number", "target", _MONEY),
    "budget_gap_ils": ("number", "target", _MONEY),
    "is_overrun": ("boolean", "target", ""),
    "is_material_overrun": ("boolean", "target", ""),
    "split": ("string", "metadata", ""),
    "cv_fold": ("integer", "metadata", ""),
}

ALLOWED = {
    "stage": {"S0", "S1", "S2", "S3"},
    "archetype": {"wedding", "bar_bat_mitzvah", "corporate", "conference"},
    "budget_vat_mode": {"includes_vat", "excludes_vat", "no_vat"},
    "fee_mode": {"percent", "amount"},
    "split": {"train", "test", "demo"},
}

_COUNTS = ("budget_revisions_to_date", "n_categories_planned", "n_categories_committed", "n_categories_pending",
           "n_commitments_active", "n_manual_costs", "n_categories_over_plan", "n_substitutions_to_date")
_NON_NEGATIVE = ("days_since_contract", "days_to_event", "fee_contract_pre_vat", "planned_total_pre_vat",
                 "manual_costs_pre_vat", "committed_pre_vat", "over_plan_pre_vat", "committed_ratio",
                 "cost_per_guest_pre_vat", "final_cost_pre_vat")
# name: (min, max); None means unbounded on that side.
RANGES: dict[str, tuple[float | None, float | None]] = {
    **{c: (0, None) for c in _COUNTS + _NON_NEGATIVE},
    "event_month": (1, 12),
    "lead_time_days": (1, None),
    "guest_count": (1, None),
    "budget_pre_vat": (0.01, None),
    "no_vat_share": (0, 1),
    "client_direct_share": (0, 1),
    "cv_fold": (-1, 4),
}

FEATURES = tuple(c for c, (_, role, _) in COLUMNS.items() if role == "feature")
TARGETS = tuple(c for c, (_, role, _) in COLUMNS.items() if role == "target")
assert set(FEATURES) == set(D1) | set(D2) and set(TARGETS) == set(LABELS)
