"""Generate the synthetic dataset.

    python -m generator.cli --seed 42 --out data/synthetic/v1

ALL OUTPUT IS SYNTHETIC. Output is byte-identical for a given seed: rows are
sorted, numbers use a fixed format, line endings are LF, and no timestamp or
machine-specific value is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import GENERATOR_VERSION
from . import config as C
from .schema import SNAPSHOT_COLUMNS, schema_document
from .simulate import simulate_all
from .snapshots import build_tables
from .split import assign_folds, assign_split

FILES = ("snapshots", "productions", "commitments", "manual_costs", "payments")


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
    return tables


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(tables: dict[str, pd.DataFrame], out: Path, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        tables[name].to_csv(out / f"{name}.csv", index=False, float_format="%.2f", lineterminator="\n")
    (out / "schema.json").write_text(json.dumps(schema_document(), indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8", newline="\n")
    snaps = tables["snapshots"]
    manifest = {
        "dataset": "production-suite-budget-ai synthetic snapshots",
        "synthetic": True,
        "notice": "ALL DATA IS SYNTHETIC. No real customer, vendor or event data.",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "batch_id": C.BATCH_ID,
        "vat_rate": C.VAT_RATE,
        "counts": {
            "productions": int(tables["productions"].shape[0]),
            "snapshots_per_production": len(C.STAGES),
            "snapshot_rows": int(snaps.shape[0]),
            "commitments": int(tables["commitments"].shape[0]),
            "manual_costs": int(tables["manual_costs"].shape[0]),
            "payments": int(tables["payments"].shape[0]),
        },
        "split_productions": {g: int(n) for g, n in
                              tables["productions"]["split"].value_counts().sort_index().items()},
        "files_sha256": {f"{name}.csv": _sha256(out / f"{name}.csv") for name in FILES}
        | {"schema.json": _sha256(out / "schema.json")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset (ALL DATA IS SYNTHETIC).")
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic/v1"))
    args = parser.parse_args()
    manifest = write(generate(args.seed), args.out, args.seed)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
