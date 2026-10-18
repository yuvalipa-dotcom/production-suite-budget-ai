"""Raw export: provider-style tables with realistic data-quality defects.

ALL DATA IS SYNTHETIC.

The raw export is derived from the clean ledgers. Defects are drawn from a
SEPARATE random stream ([seed, RAW_STREAM]), so the clean dataset is never
affected. Injection never receives labels, targets or snapshot features: it
only sees ledger and master tables, and every record has the same chance of a
defect regardless of the production's outcome.

A defect log and a defect catalogue (ground truth) are returned to the caller.
They are written only when explicitly requested, to a git-ignored location.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from . import config as C

RAW_STREAM = 3
PAYMENT_MAX_GROSS = 1_000_000.0   # documented per-payment maximum (ILS, gross)

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

# Columns of each raw table (production master drops derived aggregates and evaluation-protocol fields).
RAW_COLUMNS = {
    "productions": ["production_id", "batch_id", "is_synthetic", "archetype", "contract_date", "event_date",
                    "financial_close_date", "budget_vat_mode", "fee_mode", "fee_value", "fee_vat_mode",
                    "fee_included", "fee_concession_pre_vat"],
    "budget_revisions": None,
    "category_plans": None,
    "commitments": None,
    "manual_costs": None,
    "payments": None,
    "snapshot_schedule": ["production_id", "batch_id", "is_synthetic", "stage", "as_of_date"],
}

KEYS = {
    "productions": ["production_id"],
    "budget_revisions": ["production_id", "revision_no"],
    "category_plans": ["production_id", "category"],
    "commitments": ["commitment_id"],
    "manual_costs": ["manual_cost_id"],
    "payments": ["payment_id"],
    "snapshot_schedule": ["production_id", "stage"],
}

# (defect_type, column, rate, recoverable, expected_action)
PLAN = {
    "productions": [
        ("category_variant", "archetype", 0.03, True, "normalize_category"),
        ("whitespace", "fee_mode", 0.02, True, "strip_whitespace"),
    ],
    "budget_revisions": [
        ("date_format", "effective_date", 0.03, True, "parse_date"),
    ],
    "category_plans": [
        ("category_variant", "category", 0.03, True, "normalize_category"),
        ("whitespace", "vat_mode", 0.02, True, "strip_whitespace"),
    ],
    "commitments": [
        ("whitespace", "payment_responsibility", 0.02, True, "strip_whitespace"),
        ("category_variant", "category", 0.02, True, "normalize_category"),
        ("date_format", "selected_date", 0.02, True, "parse_date"),
        ("vat_inclusive_in_pre_vat", "original_pre_vat", 0.02, True, "recompute_pre_vat_from_entered"),
        ("exact_duplicate", "*", 0.005, True, "drop_duplicate"),
        ("missing_value", "category", 0.004, False, "retain_for_target_mark_features_unavailable_while_active"),
    ],
    "manual_costs": [
        ("whitespace", "kind", 0.02, True, "strip_whitespace"),
        ("category_variant", "category", 0.02, True, "normalize_category"),
        ("date_format", "created_date", 0.02, True, "parse_date"),
        ("vat_inclusive_in_pre_vat", "pre_vat", 0.02, True, "recompute_pre_vat_from_entered"),
        ("exact_duplicate", "*", 0.01, True, "drop_duplicate"),
        ("near_duplicate", "category", 0.005, True, "drop_duplicate"),
        ("missing_value", "created_date", 0.005, False, "retain_for_target_mark_all_snapshots_ineligible"),
    ],
    "payments": [
        ("date_format", "paid_date", 0.02, True, "parse_date"),
        ("exact_duplicate", "*", 0.01, True, "drop_duplicate"),
        ("near_duplicate", "direction", 0.005, True, "drop_duplicate"),
        ("negative_amount", "amount_gross", 0.005, False, "exclude_record"),
        ("out_of_range_amount", "amount_gross", 0.003, False, "exclude_record"),
    ],
    "snapshot_schedule": [
        ("date_format", "as_of_date", 0.02, True, "parse_date"),
    ],
}


def _as_text(df: pd.DataFrame) -> pd.DataFrame:
    """The exact text a CSV export would contain (fixed 2-decimal numbers)."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.2f", lineterminator="\n")
    buf.seek(0)
    return pd.read_csv(buf, dtype=str, keep_default_na=False)


def _alt_date(iso: str, rng: np.random.Generator) -> str:
    y, m, d = iso.split("-")
    return f"{y}/{m}/{d}" if rng.random() < 0.5 else f"{d}-{MONTHS[int(m) - 1]}-{y}"


def _variant(value: str, rng: np.random.Generator) -> str:
    words = value.split("_")
    style = int(rng.integers(0, 3))
    if style == 0:
        return " ".join(w.capitalize() for w in words)
    if style == 1:
        return "-".join(words)
    return value.upper()


def _pad(value: str, rng: np.random.Generator) -> str:
    return [" " + value, value + " ", "  " + value + " "][int(rng.integers(0, 3))]


def _key(row: pd.Series, table: str) -> str:
    return "#".join(row[k] for k in KEYS[table])


def _eligible(defect: str, row: pd.Series) -> bool:
    if defect == "vat_inclusive_in_pre_vat":
        return row["vat_mode"] == "includes_vat"
    if defect == "out_of_range_amount":
        return float(row["amount_gross"]) * 100 > PAYMENT_MAX_GROSS
    return True


def inject(ledgers: dict[str, pd.DataFrame], seed: int = C.SEED) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Return (raw tables as text, defect log). ``ledgers`` must not contain labels or features."""
    rng = np.random.default_rng([seed, RAW_STREAM])
    raw: dict[str, pd.DataFrame] = {}
    log: list[dict] = []
    for table in RAW_COLUMNS:
        src = ledgers[table]
        cols = RAW_COLUMNS[table] or list(src.columns)
        text = _as_text(src[cols])
        out_rows: list[dict] = []
        for _, row in text.iterrows():
            row = row.copy()
            record = _key(row, table)
            u = float(rng.random())
            acc, chosen = 0.0, None
            for spec in PLAN[table]:
                acc += spec[2]
                if u < acc:
                    chosen = spec
                    break
            extra: dict | None = None
            if chosen is not None and _eligible(chosen[0], row):
                defect, column, _, recoverable, action = chosen
                original = row[column] if column != "*" else ""
                if defect == "whitespace":
                    row[column] = _pad(original, rng)
                elif defect == "category_variant":
                    row[column] = _variant(original, rng)
                elif defect == "date_format":
                    row[column] = _alt_date(original, rng)
                elif defect == "vat_inclusive_in_pre_vat":
                    entered_col = "original_entered" if table == "commitments" else "entered"
                    row[column] = row[entered_col]
                elif defect == "missing_value":
                    row[column] = ""
                elif defect == "negative_amount":
                    row[column] = "-" + original
                elif defect == "out_of_range_amount":
                    row[column] = f"{float(original) * 100:.2f}"
                elif defect in ("exact_duplicate", "near_duplicate"):
                    extra = row.to_dict()
                    if defect == "near_duplicate":
                        extra[column] = _pad(original, rng).upper() if column == "direction" else _variant(original, rng)
                log.append({
                    "table": table, "record_id": record, "column": column, "defect_type": defect,
                    "original_value": original,
                    "defective_value": (extra[column] if extra and column != "*" else ("" if column == "*" else row[column])),
                    "recoverable": recoverable, "expected_action": action,
                })
            out_rows.append(row.to_dict())
            if extra is not None:
                out_rows.append(extra)
        raw[table] = pd.DataFrame(out_rows, columns=cols)
    return raw, log


def catalogue() -> dict:
    """Description of every defect type: rate, injection rule, detection rule, repair or exclusion."""
    detection = {
        "whitespace": "leading/trailing spaces in a coded text field",
        "category_variant": "value not in the closed list until lower-cased and spaces/hyphens mapped to underscores",
        "date_format": "date not in ISO YYYY-MM-DD; parses unambiguously as YYYY/MM/DD or DD-Mon-YYYY",
        "vat_inclusive_in_pre_vat": "vat_mode includes_vat and the pre-VAT column equals the typed (gross) amount",
        "exact_duplicate": "two identical rows with the same key",
        "near_duplicate": "two rows with the same key that become identical after whitespace/case/category normalisation",
        "missing_value": "required field empty",
        "negative_amount": "payment amount below zero",
        "out_of_range_amount": "payment amount above the documented per-payment maximum",
    }
    entries = []
    for table, specs in PLAN.items():
        for defect, column, rate, recoverable, action in specs:
            entries.append({"table": table, "column": column, "defect_type": defect, "rate_per_record": rate,
                            "injection_rule": "independent per-record draw from the raw stream; one defect at most per record",
                            "detection_rule": detection[defect], "recoverable": recoverable,
                            "expected_action": action})
    return {"seed_stream": [C.SEED, RAW_STREAM], "synthetic": True, "defects": entries}
