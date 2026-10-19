"""A3 tool: run the unmodified P2 handoff gate (``gate.run``) on the workspace handoff, read-only."""

from __future__ import annotations

import json

from .workspace import Workspace


def run_gate(ws: Workspace) -> dict:
    """The gate reads clean_data.csv, dataset_contract.json, the raw copy and the committed contracts."""
    from gate.run import main as gate_main

    for name in ("out.clean_data", "out.dataset_contract"):
        if not ws.has_output(name):
            raise LookupError(f"{name.split('.', 1)[1]} does not exist yet")
    out_dir = ws.root / "tmp" / "gate"
    code = gate_main(["--stage", "handoff", "--raw", str(ws.raw_dir()), "--out", str(out_dir),
                      "--data", str(ws.resolve("out.clean_data", output=True)),
                      "--contract", str(ws.resolve("out.dataset_contract", output=True))])
    result_bytes = (out_dir / "gate_result.json").read_bytes()
    ws.write("out.gate_result", result_bytes)
    result = json.loads(result_bytes)
    result["exit_code"] = code
    return result
