"""Feature sets, targets and the model input matrix (Addendum P4-1, sections 1-2).

The 29 handoff features are already as-of-snapshot values (``gate/reconstruct.py``;
docs/p4_plan.md section c). ``F27`` drops the two features the product cannot compute
today (``PROPOSED`` in docs/product_field_mapping.md); ``F29`` is research only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gate import handoff_spec as H

REPO = Path(__file__).resolve().parents[2]
PROPOSED_IN_PRODUCT = ("budget_revisions_to_date", "client_direct_share")
F29 = tuple(H.FEATURES)
F27 = tuple(c for c in F29 if c not in PROPOSED_IN_PRODUCT)
FEATURE_SETS = {"F27": F27, "F29": F29}
PRIMARY_FEATURE_SET = "F27"

CATEGORICAL = ("stage", "archetype", "budget_vat_mode", "fee_mode")
BOOLEAN = ("fee_included",)
TARGETS = ("is_overrun", "is_material_overrun")
PRIMARY_TARGET = "is_overrun"
ROW = "_row"                     # handoff row position; dropped by the preprocessing, recorded by the fit spy

assert len(F27) == 27 and len(F29) == 29


def numeric(features: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in features if c not in CATEGORICAL and c not in BOOLEAN)


def outcome_only_columns(repo: Path = REPO) -> set[str]:
    """Every outcome-only column name of every table in ``contracts/column_roles.json``."""
    roles = json.loads((repo / "contracts" / "column_roles.json").read_text(encoding="utf-8"))["tables"]
    return {c for cols in roles.values() for c, spec in cols.items() if spec["role"] == "outcome-only"}


def model_input(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """The model input: the row id plus the feature columns of one feature set, booleans as 0/1."""
    cols = FEATURE_SETS[feature_set]
    X = df.loc[:, [ROW, *cols]].copy()
    for c in BOOLEAN:
        X[c] = X[c].astype(int)
    return X
