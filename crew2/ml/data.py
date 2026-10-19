"""Crew 2 input: the Policy C handoff, built in-process and verified (docs/p4_plan.md section b).

The clean data are rebuilt from the raw export with the deterministic Crew 1 cleaning
function (no LLM) and must equal the expected deterministic output byte for byte. The
expected sha256 is pinned below; ``tests/test_crew2_data.py`` proves it equals the
independent reference built from the committed clean snapshots and the ground truth.

The handoff is then partitioned by its frozen ``split``:
* train rows keep their labels;
* test rows lose every label column at the text level: the labels stay in a sealed
  ``TestVault`` that parses them only when opened, once, for the final evaluation;
* demo rows lose every label column and their labels are never parsed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from crew1.tools.cleaning import build_clean_data
from gate import handoff_spec as H
from gate.checks import RAW_TABLES

from .features import ROW

REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "synthetic" / "v2" / "raw"
CONTRACTS_DIR = REPO / "contracts"

EXPECTED_CLEAN_SHA256 = "2b8353247ebd7c5e9c63acec4c8a0bbc773380e846bd8c59ba21b58b75aa77e8"
EXPECTED_ROWS = 1538
EXPECTED_SCHEDULED = 1600
EXPECTED_EXCLUSIONS = {"commitment_category_missing": 50, "manual_cost_undated": 12}
LABEL_COLUMNS = tuple(H.TARGETS)
KEY = ("production_id", "stage")


class HandoffMismatch(RuntimeError):
    """The rebuilt handoff is not the expected deterministic output: the run stops."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[dict[str, pd.DataFrame], str]:
    """The raw tables (text, exactly as Crew 1 reads them) and the raw manifest sha256, hash-checked."""
    manifest_bytes = (raw_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    tables = {}
    for t in RAW_TABLES:
        data = (raw_dir / f"{t}.csv").read_bytes()
        if sha256_bytes(data) != manifest["files_sha256"][f"{t}.csv"]:
            raise HandoffMismatch(f"raw table {t} does not match its manifest hash")
        tables[t] = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    return tables, sha256_bytes(manifest_bytes)


def frozen_split(contracts_dir: Path = CONTRACTS_DIR) -> dict:
    return json.loads((contracts_dir / "frozen_split.json").read_text(encoding="utf-8"))


def build_handoff(raw_dir: Path = RAW_DIR, contracts_dir: Path = CONTRACTS_DIR) -> tuple[str, dict]:
    """(clean_data.csv text, cleaning log), rebuilt with the deterministic Policy C cleaning function."""
    raw, manifest_sha = load_raw(raw_dir)
    return build_clean_data(raw, frozen_split(contracts_dir), manifest_sha)


def verify_handoff(text: str, log: dict, expected_sha256: str = EXPECTED_CLEAN_SHA256) -> dict:
    """Raise ``HandoffMismatch`` unless the handoff is the expected deterministic output."""
    digest = sha256_bytes(text.encode("utf-8"))
    problems = []
    if digest != expected_sha256:
        problems.append(f"sha256 {digest} != expected {expected_sha256}")
    if log["snapshots"] != {"scheduled": EXPECTED_SCHEDULED, "eligible": EXPECTED_ROWS,
                            "ineligible": EXPECTED_SCHEDULED - EXPECTED_ROWS}:
        problems.append(f"snapshot counts {log['snapshots']}")
    if log["exclusion_counts"] != EXPECTED_EXCLUSIONS:
        problems.append(f"exclusion counts {log['exclusion_counts']}")
    header = text.split("\n", 1)[0].split(",")
    if header != list(H.COLUMNS):
        problems.append("columns differ from gate/handoff_spec.py")
    if problems:
        raise HandoffMismatch("; ".join(problems))
    return {"clean_data_sha256": digest, "rows": EXPECTED_ROWS, "exclusion_counts": dict(log["exclusion_counts"])}


# ---------------------------------------------------------------- typed parsing and partitioning

def _typed(column: str, value: str):
    dtype = H.COLUMNS[column][0]
    if dtype == "integer":
        return int(value)
    if dtype == "number":
        return float(value)
    if dtype == "boolean":
        if value not in ("True", "False"):
            raise HandoffMismatch(f"{column}: not a boolean")
        return value == "True"
    return value


class TestAlreadyOpened(RuntimeError):
    pass


class TestVault:
    """The sealed test labels. Parsed only by ``open``, which works once and needs the frozen model hashes."""

    __test__ = False                                  # not a pytest test class

    def __init__(self, text: str, keys: list[tuple[str, str]]):
        self._text = text
        self._keys = set(keys)
        self.opens: list[dict] = []

    @property
    def opened(self) -> bool:
        return bool(self.opens)

    def open(self, model_sha256: dict[str, str]) -> pd.DataFrame:
        if self.opens:
            raise TestAlreadyOpened("the test set has already been evaluated once; a second evaluation is refused")
        if not model_sha256 or not all(isinstance(v, str) and len(v) == 64 for v in model_sha256.values()):
            raise ValueError("the test set opens only for frozen models (sha256 of each fitted model)")
        self.opens.append({"model_sha256": dict(sorted(model_sha256.items()))})
        rows = []
        for r in csv.DictReader(io.StringIO(self._text)):
            if (r["production_id"], r["stage"]) in self._keys:
                rows.append({"production_id": r["production_id"], "stage": r["stage"],
                             **{c: _typed(c, r[c]) for c in LABEL_COLUMNS}})
        return pd.DataFrame(rows).sort_values(list(KEY)).reset_index(drop=True)


@dataclass
class Partitioned:
    train: pd.DataFrame                    # features, labels, keys, cv_fold, row id
    test_features: pd.DataFrame            # features, keys, row id; no label column
    demo_features: pd.DataFrame            # features, keys, row id; no label column
    vault: TestVault
    counts: dict = field(default_factory=dict)


def partition(text: str) -> Partitioned:
    """Split the handoff by its frozen group. Label fields of test and demo rows are dropped as text."""
    groups: dict[str, list[dict]] = {"train": [], "test": [], "demo": []}
    test_keys = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        split = r["split"]
        keep = [c for c in H.COLUMNS if split == "train" or c not in LABEL_COLUMNS]
        row = {ROW: i, **{c: _typed(c, r[c]) for c in keep}}
        groups[split].append(row)
        if split == "test":
            test_keys.append((r["production_id"], r["stage"]))
    frames = {s: pd.DataFrame(rows) for s, rows in groups.items()}
    for s in ("test", "demo"):
        assert not set(LABEL_COLUMNS) & set(frames[s].columns)
    counts = {s: {"rows": int(len(f)), "productions": int(f["production_id"].nunique()),
                  "by_stage": {k: int(v) for k, v in f["stage"].value_counts().sort_index().items()}}
              for s, f in frames.items()}
    return Partitioned(frames["train"], frames["test"], frames["demo"], TestVault(text, test_keys), counts)
