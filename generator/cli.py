"""Generate the synthetic dataset.

    python -m generator.cli --version v2 --out data/synthetic/v2
    python -m generator.cli --version v2 --out data/synthetic/v2 --ground-truth-out ground_truth/v2
    python -m generator.cli --version v1 --out data/synthetic/v1      # frozen, for traceability

ALL OUTPUT IS SYNTHETIC. Output is byte-identical for a given seed and version:
rows are sorted, numbers use a fixed format, line endings are LF, and no
timestamp or machine-specific value is written.

Versions
* v1 -- the original export (snapshots + ledgers), kept byte-identical.
* v2 -- v1 plus the approved-budget revision ledger and the category-plan
  ledger, so that every snapshot feature and label can be rebuilt from dated
  records; plus a raw export with documented data-quality defects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import config as C
from .raw import RAW_STREAM, catalogue, inject
from .schema import SNAPSHOT_COLUMNS, schema_document
from .simulate import simulate_all
from .snapshots import build_tables
from .split import assign_folds, assign_split

FILES_V1 = ("snapshots", "productions", "commitments", "manual_costs", "payments")
FILES_V2 = FILES_V1 + ("budget_revisions", "category_plans")
FILES = FILES_V2
RAW_FILES = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs",
             "payments", "snapshot_schedule")
GENERATOR_VERSION_BY_DATASET = {"v1": "1.0.0", "v2": "1.1.0"}

V2_REASON = ("v1 preserved only the initial and final approved budget and a revision count. v2 adds the "
             "approved-budget revision ledger (original budget plus every approved revision with its effective "
             "date) and the category-plan ledger, so that every snapshot feature and label can be rebuilt from "
             "records dated on or before the snapshot. Labels, signed gaps, production ids, split groups, folds "
             "and every calibration parameter are unchanged; the snapshot table is byte-identical to v1.")


def generate(seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    tables = build_tables(simulate_all(seed))
    groups = assign_split(tables["productions"]["production_id"].tolist(), seed)
    train_ids = [pid for pid, g in groups.items() if g == "train"]
    folds = assign_folds(train_ids, C.N_FOLDS, seed)
    for name in ("snapshots", "productions"):
        df = tables[name]
        df["split"] = df["production_id"].map(groups)
        df["cv_fold"] = df["production_id"].map(lambda pid: folds.get(pid, -1)).astype(int)
    tables["snapshots"] = tables["snapshots"][list(SNAPSHOT_COLUMNS)]
    tables["snapshot_schedule"] = tables["snapshots"][["production_id", "batch_id", "is_synthetic",
                                                       "stage", "as_of_date"]].copy()
    return tables


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, float_format="%.2f", lineterminator="\n")


def _write_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write(tables: dict[str, pd.DataFrame], out: Path, seed: int, version: str = "v1") -> dict:
    files = FILES_V1 if version == "v1" else FILES_V2
    out.mkdir(parents=True, exist_ok=True)
    for name in files:
        _write_csv(tables[name], out / f"{name}.csv")
    _write_json(schema_document(), out / "schema.json")
    snaps = tables["snapshots"]
    counts = {
        "productions": int(tables["productions"].shape[0]),
        "snapshots_per_production": len(C.STAGES),
        "snapshot_rows": int(snaps.shape[0]),
        "commitments": int(tables["commitments"].shape[0]),
        "manual_costs": int(tables["manual_costs"].shape[0]),
        "payments": int(tables["payments"].shape[0]),
    }
    manifest = {
        "dataset": "production-suite-budget-ai synthetic snapshots",
        "synthetic": True,
        "notice": "ALL DATA IS SYNTHETIC. No real customer, vendor or event data.",
        "generator_version": GENERATOR_VERSION_BY_DATASET[version],
        "seed": seed,
        "batch_id": C.BATCH_ID,
        "vat_rate": C.VAT_RATE,
        "counts": counts,
        "split_productions": {g: int(n) for g, n in
                              tables["productions"]["split"].value_counts().sort_index().items()},
        "files_sha256": {f"{name}.csv": _sha256(out / f"{name}.csv") for name in files}
        | {"schema.json": _sha256(out / "schema.json")},
    }
    if version == "v2":
        counts["budget_revisions"] = int(tables["budget_revisions"].shape[0])
        counts["category_plans"] = int(tables["category_plans"].shape[0])
        manifest["dataset_version"] = "v2"
        manifest["supersedes"] = "v1"
        manifest["reason"] = V2_REASON
    _write_json(manifest, out / "manifest.json")
    return manifest


def write_raw(tables: dict[str, pd.DataFrame], out: Path, seed: int) -> tuple[dict, list[dict]]:
    """Write the provider-style raw export. Returns (raw manifest, defect log)."""
    raw, log = inject({name: tables[name] for name in RAW_FILES}, seed)
    out.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        raw[name].to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    manifest = {
        "dataset": "production-suite-budget-ai synthetic raw export",
        "synthetic": True,
        "notice": "ALL DATA IS SYNTHETIC. No real customer, vendor or event data.",
        "seed": seed,
        "random_stream": [seed, RAW_STREAM],
        "batch_id": C.BATCH_ID,
        "vat_rate": C.VAT_RATE,
        "rows": {f"{name}.csv": int(raw[name].shape[0]) for name in RAW_FILES},
        "files_sha256": {f"{name}.csv": _sha256(out / f"{name}.csv") for name in RAW_FILES},
    }
    _write_json(manifest, out / "manifest.json")
    return manifest, log


def write_ground_truth(log: list[dict], out: Path) -> None:
    """Defect log + catalogue. Destination must be git-ignored; never published."""
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(out / "defect_log.csv", index=False, lineterminator="\n")
    _write_json(catalogue(), out / "defect_catalogue.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset (ALL DATA IS SYNTHETIC).")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ground-truth-out", type=Path, default=None,
                        help="also write the defect log and catalogue here (must be a git-ignored location)")
    args = parser.parse_args()
    out = args.out or Path("data/synthetic") / args.version
    tables = generate(args.seed)
    manifest = write(tables, out, args.seed, args.version)
    if args.version == "v2":
        _, log = write_raw(tables, out / "raw", args.seed)
        if args.ground_truth_out is not None:
            write_ground_truth(log, args.ground_truth_out)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
