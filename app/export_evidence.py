"""Export verified Flow runs from ``out/flow_runs/`` to the committable ``flow_evidence/runs/`` (P5-F).

    python -m app.export_evidence p5e-run-001 p5e-break-money_x100 ...

Standard library only. A run is exported only if it verifies (``evidence.load_run``) and has a final status.
Only an allow-list of files is copied (the 8 required outputs, provenance, gate and kickoff evidence, traces;
for a stopped run, its gate evidence only), and ``evidence.json`` records every copied file's sha256. An
existing exported run is never overwritten.
"""

from __future__ import annotations

import json
import shutil
import sys

from . import evidence as E

PASS_FILES = (*E.REQUIRED, "crew1/gate_result.json", "crew1/crew_kickoff.json", "crew1/trace.jsonl",
              "handoff/flow_contract.json", "handoff/gate_report.json", "crew2/features.schema.json",
              "crew2/run_manifest.json", "crew2/crew_kickoff.json", "crew2/trace.jsonl",
              "flow_manifest.json", "flow_events.jsonl")
STOPPED_FILES = ("handoff/gate_report.json", "handoff/scenario.json", "crew1/crew_kickoff.json",
                 "flow_manifest.json", "flow_events.jsonl")


def export(run_id: str) -> dict:
    run = E.load_run("local", run_id)
    if not run.verified or run.status not in ("PASS", "STOPPED_CONTRACT"):
        raise E.EvidenceError(f"{run_id}: not exportable (status {run.status}, {run.verification.get('problems')})")
    dst = E.EVIDENCE_DIR / "runs" / run_id
    if dst.exists():
        raise E.EvidenceError(f"{run_id}: already exported; exported evidence is never overwritten")
    names = PASS_FILES if run.status == "PASS" else STOPPED_FILES
    files = {}
    for rel in names:
        src = run.path(rel)
        if not src.is_file():
            raise E.EvidenceError(f"{run_id}: {rel} is missing")
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst / rel)
        files[rel] = E.sha256_file(dst / rel)
    omitted = {rel: sha for rel, sha in (run.manifest.get("required_outputs_sha256") or {}).items() if rel not in files}
    index = {"run_id": run_id, "status": run.status, "scenario": run.manifest.get("scenario"), "synthetic": True,
             "exported_from": "out/flow_runs (verified before export)", "files_sha256": files,
             "omitted_sha256": omitted,
             "omitted_reason": "Crew 1's outputs of a stopped run are not copied (they were verified before export); "
                               "their sha256 values are recorded here and in flow_manifest.json" if omitted else None}
    (dst / "evidence.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
    check = E.load_run("canonical", run_id)
    if not check.verified:
        shutil.rmtree(dst)
        raise E.EvidenceError(f"{run_id}: the exported copy does not verify: {check.verification['problems']}")
    return index


def main(argv=None) -> int:
    run_ids = list(sys.argv[1:] if argv is None else argv)
    if not run_ids:
        print(__doc__)
        return 2
    for run_id in run_ids:
        idx = export(run_id)
        print(f"exported {run_id}: {idx['status']}, {len(idx['files_sha256'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
