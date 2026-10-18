"""Independent reference handoff (test-only). ALL DATA IS SYNTHETIC.

Built from sources the runtime gate never reads, so a passing gate run is a
real cross-check rather than the gate agreeing with itself:
* values: the committed clean v2 snapshots (written by the generator);
* eligibility: derived from the generated ground truth (``expected_ineligible``);
* column specification: the dataset's own ``schema.json``.
Nothing from ``gate`` is imported here.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from tests.conftest import ACTIVE_DATA, REPO

NOT_DELIVERED = ("batch_id", "is_synthetic", "event_date")   # metadata outside the approved handoff
ID_COLUMNS = ("production_id",)


def reference_frames(expected_ineligible: dict) -> tuple[pd.DataFrame, dict]:
    snaps = pd.read_csv(ACTIVE_DATA / "snapshots.csv", dtype=str, keep_default_na=False)
    schema = json.loads((ACTIVE_DATA / "schema.json").read_text(encoding="utf-8"))["columns"]
    roles = json.loads((REPO / "contracts" / "column_roles.json").read_text(encoding="utf-8"))["tables"]
    keep = ~pd.Series(list(zip(snaps["production_id"], snaps["stage"]))).isin(set(expected_ineligible)).values
    cols = [c for c in snaps.columns if c not in NOT_DELIVERED]
    data = snaps.loc[keep, cols].reset_index(drop=True)
    columns = []
    for c in cols:
        s = schema[c]
        spec = {"name": c, "dtype": s["type"], "role": "id" if c in ID_COLUMNS else s["role"],
                "unit": s.get("unit", ""), "nullable": False}
        spec |= {k: s[k] for k in ("allowed", "min", "max") if k in s}
        columns.append(spec)
    raw_manifest = ACTIVE_DATA / "raw" / "manifest.json"
    contract = {
        "dataset": "production-suite-budget-ai Crew-1 handoff (synthetic)",
        "version": "v2",
        "raw_manifest_sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
        "primary_key": ["production_id", "stage"],
        "row_count": int(len(data)),
        "exclusion_counts": dict(sorted(Counter(expected_ineligible.values()).items())),
        "columns": columns,
        "split_policy": "frozen: contracts/frozen_split.json",
        "outcome_only_exclusions": sorted(f"{t}.{c}" for t, x in roles.items() for c, s in x.items()
                                          if s["role"] == "outcome-only"),
    }
    return data, contract


def write_handoff(data: pd.DataFrame, contract: dict, out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    data_path, contract_path = out / "clean_data.csv", out / "dataset_contract.json"
    data.to_csv(data_path, index=False, lineterminator="\n")
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return data_path, contract_path
