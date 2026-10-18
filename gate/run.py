"""Command-line entry point for the deterministic P2 gates.

    python -m gate.run --stage source  --raw data/synthetic/v2/raw --out <dir>
    python -m gate.run --stage handoff --raw data/synthetic/v2/raw --out <dir> \\
        --data <dir>/clean_data.csv --contract <dir>/dataset_contract.json

Writes ``gate_result.json`` and ``gate_report.md`` (byte-stable for identical inputs).
Exit codes: 0 every HARD check passed; 1 at least one HARD check failed or could
not run (including a missing or unreadable input); 2 invalid command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checks import HARD, PASS, _contracts, handoff_checks, sha, source_checks
from .handoff_spec import CONTRACT_FILE, DATA_FILE

LIMITATIONS = ["Offline deterministic validation; synthetic data only.",
               "Expected values are rebuilt from the raw export under Policy C; no model is validated."]


def _digest(path) -> str | None:
    p = Path(path)
    return sha(p) if p.is_file() else None


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deterministic P2 quality gate (synthetic data only).")
    p.add_argument("--stage", choices=("source", "handoff"), required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data")
    p.add_argument("--contract")
    a = p.parse_args(argv)
    if a.stage == "handoff" and (not a.data or not a.contract):
        p.error("--data and --contract are required for handoff")
    root = Path(__file__).resolve().parents[1]
    checks = source_checks(root, a.raw) if a.stage == "source" else handoff_checks(root, a.data, a.contract, a.raw)
    contracts = _contracts(root)
    inputs = {"raw/manifest.json": _digest(Path(a.raw) / "manifest.json"),
              "contracts/column_roles.json": _digest(contracts["roles"]),
              "contracts/frozen_split.json": _digest(contracts["frozen"])}
    if a.stage == "handoff":
        inputs |= {DATA_FILE: _digest(a.data), CONTRACT_FILE: _digest(a.contract),
                   "contracts/dataset_contract.schema.json": _digest(contracts["schema"])}
    overall = "PASS" if all(c["status"] == PASS for c in checks if c["severity"] == HARD) else "FAIL"
    obj = {"stage": a.stage, "status": overall, "input_sha256s": inputs, "checks": checks, "limitations": LIMITATIONS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_result.json").write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    lines = [f"# Quality gate: {a.stage}", "", f"**Status:** {overall}", "",
             "| ID | Severity | Status | Details |", "|---|---|---|---|"]
    lines += [f"| {c['id']} | {c['severity']} | {c['status']} | {_cell(c['details'])} |" for c in checks]
    lines += ["", "## Limitations"] + [f"- {x}" for x in LIMITATIONS] + [""]
    (out / "gate_report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
