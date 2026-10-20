"""The staged Crew-2 workspace: the Crew 1 isolation pattern with Crew 2's own allow-lists.

A fresh directory OUTSIDE the repository holds a copy of the raw export only (hash-checked
against its manifest). The committed contracts are read by Python inside the tools, never
through an agent argument. Agents can name outputs only through the logical names below.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

from crew1.tools.workspace import REPO, IsolationError, Workspace, _norm, sha256_bytes

RAW_TABLES = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs", "payments",
              "snapshot_schedule")
CHART_IDS = ("cv_brier", "reliability_test")
READ_NAMES = {**{f"raw.{t}": f"raw/{t}.csv" for t in RAW_TABLES}, "raw.manifest": "raw/manifest.json"}
OUTPUT_NAMES = {"out.metrics": "out/metrics.json", "out.model": "out/model.joblib",
                "out.model_card": "out/model_card.md", "out.demo_predictions": "out/demo_predictions.csv",
                "out.cv_predictions": "out/cv_predictions.csv", "out.leakage_report": "out/leakage_report.json",
                "out.feature_availability": "out/feature_availability.json",
                **{f"out.chart.{c}": f"out/charts/{c}.png" for c in CHART_IDS}}
RAW_DIR = REPO / "data" / "synthetic" / "v2" / "raw"
CONTRACTS_DIR = REPO / "contracts"


def stage_workspace(raw_dir: Path = RAW_DIR, parent: Path | None = None) -> Workspace:
    """Create a fresh workspace outside the repository and copy in only the raw export."""
    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="crew2_ws_", dir=parent)))
    if _norm(root).startswith(_norm(REPO) + os.sep) or _norm(root) == _norm(REPO):
        shutil.rmtree(root, ignore_errors=True)
        raise IsolationError("workspace_inside_repository")
    for sub in ("raw", "out", "out/charts", "tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((raw_dir / "manifest.json").read_text(encoding="utf-8"))
    shutil.copyfile(raw_dir / "manifest.json", root / "raw" / "manifest.json")
    for t in RAW_TABLES:
        shutil.copyfile(raw_dir / f"{t}.csv", root / "raw" / f"{t}.csv")
        if sha256_bytes((root / "raw" / f"{t}.csv").read_bytes()) != manifest["files_sha256"][f"{t}.csv"]:
            shutil.rmtree(root, ignore_errors=True)
            raise IsolationError("raw_copy_hash_mismatch")
    return Workspace(root, READ_NAMES, OUTPUT_NAMES)
