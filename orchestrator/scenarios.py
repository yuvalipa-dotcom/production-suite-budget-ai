"""Deliberate contract breaks for the Flow demonstration (P5-E). ALL DATA IS SYNTHETIC.

Each scenario tampers with the Flow's own isolated copy of the handoff (``<run>/handoff/``), made from the
files Crew 1 wrote in the SAME run. Crew 1's outputs (``<run>/crew1/``) and every approved historical file
are never modified; the Flow records their hashes before and after.

* ``column_rename``: a required column is renamed in ``clean_data.csv`` (``budget_pre_vat`` ->
  ``budget_ils``);
* ``money_x100``: every ``budget_pre_vat`` value is multiplied by 100 (e.g. agorot instead of shekels) AND
  the attacker regenerates a matching ``flow_contract.json`` from the tampered data, so every digest and
  declared profile agrees with it; only the pinned reference can reject it;
* ``row_deletion``: one required snapshot row (a train row of the frozen split) is deleted from
  ``clean_data.csv``;
* ``contract_column_deletion``: the definition of a required column (``budget_pre_vat``) is deleted from
  ``dataset_contract.json`` itself; the data are unchanged;
* ``label_leakage``: an outcome label is leaked as an extra input column (``overrun_hint``, a copy of
  ``is_overrun``) in ``clean_data.csv``, and the attacker declares it as a feature in ``dataset_contract.json``.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from budget_flow import contract as C

SCENARIOS = ("column_rename", "money_x100", "row_deletion", "contract_column_deletion", "label_leakage")


def _rows(data: bytes) -> tuple[list[str], list[dict]]:
    r = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    return list(r.fieldnames), list(r)


def _csv(header: list[str], rows: list[dict]) -> bytes:
    out = io.StringIO()
    w = csv.DictWriter(out, header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue().encode("utf-8")


def apply(scenario: str, handoff: Path) -> dict:
    """Tamper with the isolated handoff copy in ``handoff``; returns what was changed (for the manifest)."""
    clean_p, base_p, flow_p = handoff / "clean_data.csv", handoff / "dataset_contract.json", handoff / "flow_contract.json"
    header, rows = _rows(clean_p.read_bytes())
    regenerated = False
    if scenario == "column_rename":
        header = ["budget_ils" if c == "budget_pre_vat" else c for c in header]
        for r in rows:
            r["budget_ils"] = r.pop("budget_pre_vat")
        clean_p.write_bytes(_csv(header, rows))
        change = "column budget_pre_vat renamed to budget_ils in clean_data.csv"
    elif scenario == "money_x100":
        for r in rows:
            r["budget_pre_vat"] = repr(round(float(r["budget_pre_vat"]) * 100, 2))
        tampered = _csv(header, rows)
        clean_p.write_bytes(tampered)
        flow_p.write_text(json.dumps(C.build_flow_contract(tampered, base_p.read_bytes()), indent=2) + "\n",
                          encoding="utf-8", newline="\n")
        regenerated = True
        change = "budget_pre_vat multiplied by 100 in clean_data.csv; flow_contract.json regenerated from it"
    elif scenario == "row_deletion":
        idx = next(i for i, r in enumerate(rows) if r["split"] == "train")
        removed = rows.pop(idx)
        clean_p.write_bytes(_csv(header, rows))
        change = f"required snapshot row {idx} (train, stage {removed['stage']}) deleted from clean_data.csv"
    elif scenario == "contract_column_deletion":
        base = json.loads(base_p.read_text(encoding="utf-8"))
        base["columns"] = [c for c in base["columns"] if c["name"] != "budget_pre_vat"]
        base_p.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8", newline="\n")
        change = "the definition of the required column budget_pre_vat deleted from dataset_contract.json"
    elif scenario == "label_leakage":
        header = header + ["overrun_hint"]
        for r in rows:
            r["overrun_hint"] = r["is_overrun"]
        clean_p.write_bytes(_csv(header, rows))
        base = json.loads(base_p.read_text(encoding="utf-8"))
        template = next(c for c in base["columns"] if c["name"] == "is_overrun")
        base["columns"].append({**template, "name": "overrun_hint", "role": "feature",
                                "description": "Leaked copy of the outcome label."})
        base_p.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8", newline="\n")
        change = "outcome label is_overrun copied into a new input column overrun_hint, declared as a feature"
    else:
        raise ValueError(f"unknown scenario {scenario!r}; choose one of {SCENARIOS}")
    return {"scenario": scenario, "change": change, "attacker_regenerated_flow_contract": regenerated,
            "applied_to": "the Flow's isolated handoff copy only (Crew 1 outputs untouched)"}
