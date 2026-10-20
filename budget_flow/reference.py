"""Build (or check) the pinned monetary scale reference ``money_scale_reference.v1.json``.

    python -m budget_flow.reference --check    # fail if the committed reference differs from a fresh build
    python -m budget_flow.reference --write    # (re)write it; then pin its sha256 in budget_flow/contract.py

The source is the approved Policy C handoff: the committed Crew-1 output of the published run p3-live-005,
which must be byte-identical to the pinned P4 input (``crew2.ml.data.EXPECTED_CLEAN_SHA256``). Only TRAIN rows
are used, so the reference carries no test or demo outcome. ALL DATA IS SYNTHETIC.
"""

from __future__ import annotations

import argparse
import json
import sys

from crew2.ml.data import EXPECTED_CLEAN_SHA256

from . import contract as C

SOURCE = C.REPO / "artifacts" / "crew1" / "p3-live-005" / "clean_data.csv"


def build_reference() -> dict:
    data = SOURCE.read_bytes()
    if C.sha256_hex(data) != EXPECTED_CLEAN_SHA256:
        raise SystemExit("the reference source is not the approved Policy C handoff (sha256 differs from the pinned "
                         "P4 input)")
    _, rows = C.parse_and_redact(data)
    profile = C.scale_profile(rows)
    columns = {}
    for c in C.MONEY_COLUMNS:
        p = profile[c]
        stat = "median_abs" if p["median_abs"] else "p90_abs"
        if not p[stat]:
            raise SystemExit(f"{c}: no non-zero statistic; the column cannot be scale-checked")
        columns[c] = {"unit": C.MONEY_UNIT, "statistic": stat, "value": p[stat], **p}
    return {
        "version": C.REFERENCE_VERSION,
        "synthetic": True,
        "description": "Pinned scale reference for the monetary columns of a Crew-1 handoff (Flow contract flow-1). "
                       "statistic is the median of absolute values, or the nearest-rank p90 of absolute values when "
                       "the median is 0; TRAIN rows only.",
        "source": {"run": "p3-live-005", "file": "artifacts/crew1/p3-live-005/clean_data.csv",
                   "identity": "byte-identical to the pinned P4 input (crew2.ml.data.EXPECTED_CLEAN_SHA256)",
                   "rows": "train split only", "train_rows": sum(1 for r in rows if r["split"] == C.TRAIN)},
        "currency": dict(C.CURRENCY),
        "band": list(C.SCALE_BAND),
        "columns": columns,
    }


def render(ref: dict) -> bytes:
    return (json.dumps(ref, indent=2) + "\n").encode("utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    fresh = render(build_reference())
    if args.write:
        C.REFERENCE_FILE.write_bytes(fresh)
        print(f"wrote {C.REFERENCE_FILE.name}; pin its sha256 {C.sha256_hex(fresh)} in budget_flow/contract.py")
        return 0
    same = C.REFERENCE_FILE.is_file() and C.REFERENCE_FILE.read_bytes() == fresh
    pinned = same and C.sha256_hex(fresh) == C.REFERENCE_SHA256
    print("reference matches a fresh build" if same else "reference DIFFERS from a fresh build",
          "and its sha256 is pinned" if pinned else "but its sha256 is NOT the pinned value" if same else "")
    return 0 if pinned else 1


if __name__ == "__main__":
    sys.exit(main())
