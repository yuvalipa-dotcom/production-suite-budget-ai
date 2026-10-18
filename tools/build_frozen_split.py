"""Build (or check) the frozen split reference contract ``contracts/frozen_split.json``.

    python -m tools.build_frozen_split            # write the contract
    python -m tools.build_frozen_split --check    # fail if the committed contract differs

ALL DATA IS SYNTHETIC. The contract records the existing train/test/demo groups
and grouped CV folds; it does not create or change any assignment.

Two independent sources must agree exactly, or the build stops:
1. the committed assignments in ``data/synthetic/v2/productions.csv`` (and the
   frozen v1 copy, which must be identical);
2. a fresh run of the seeded rules in ``generator/split.py`` over the same ids.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from generator import config as C
from generator.split import FOLD_STREAM, SPLIT_STREAM, assign_folds, assign_split

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "contracts" / "frozen_split.json"
SOURCES = ("data/synthetic/v2/productions.csv", "data/synthetic/v1/productions.csv")
NON_TRAIN_FOLD = -1
CONTRACT_VERSION = "1.0.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed(rel: str) -> dict[str, tuple[str, int]]:
    df = pd.read_csv(REPO / rel, dtype=str, keep_default_na=False)
    return {r.production_id: (r.split, int(r.cv_fold)) for r in df.itertuples()}


def _regenerated(ids: list[str]) -> dict[str, tuple[str, int]]:
    groups = assign_split(ids, C.SEED)
    folds = assign_folds([p for p, g in groups.items() if g == "train"], C.N_FOLDS, C.SEED)
    return {p: (groups[p], folds.get(p, NON_TRAIN_FOLD)) for p in ids}


def build() -> str:
    v2, v1 = (_committed(rel) for rel in SOURCES)
    if v2 != v1:
        raise SystemExit("STOP: v1 and v2 committed assignments disagree")
    ids = sorted(v2)
    if _regenerated(ids) != v2:
        raise SystemExit("STOP: committed assignments disagree with the seeded rules in generator/split.py")
    rows = [{"production_id": p, "split": v2[p][0], "cv_fold": v2[p][1]} for p in ids]
    counts = {g: sum(1 for r in rows if r["split"] == g) for g in ("train", "test", "demo")}
    folds = {str(k): sum(1 for r in rows if r["split"] == "train" and r["cv_fold"] == k) for k in range(C.N_FOLDS)}
    doc = {
        "contract": "frozen_split",
        "version": CONTRACT_VERSION,
        "synthetic": True,
        "description": ("Frozen production-level train/test/demo groups and grouped CV folds (train only). "
                        "Every snapshot inherits its production's group and fold. Not a dataset version."),
        "non_train_cv_fold": NON_TRAIN_FOLD,
        "n_folds": C.N_FOLDS,
        "counts": {"productions": len(rows), **counts, "train_folds": folds},
        "provenance": {
            "source_files_sha256": {rel: _sha256(REPO / rel) for rel in SOURCES},
            "cross_check": "identical to a fresh run of generator/split.py (assign_split, assign_folds)",
            "seed": C.SEED,
            "split_random_stream": [C.SEED, SPLIT_STREAM],
            "fold_random_stream": [C.SEED, FOLD_STREAM],
            "split_ratios": dict(sorted(C.SPLIT_RATIOS.items())),
        },
        "productions": rows,
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = build()
    if args.check:
        same = OUT.is_file() and OUT.read_bytes() == text.encode("utf-8")
        print("frozen_split.json up to date" if same else "frozen_split.json differs from a fresh build")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
