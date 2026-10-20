"""``load_handoff``: Crew 2's only entry to the data in the Flow. ALL DATA IS SYNTHETIC.

It reads exactly the two files Crew 1 wrote, given as explicit paths (plus, optionally, the Flow's
``flow-1`` contract for them). It never reads the raw export, never imports Crew 1 and never calls
Crew 1's cleaning code. The steps, in order:

1. read the delivered bytes (a path inside ``data/synthetic/`` is refused);
2. parse ``clean_data.csv`` and redact every non-train outcome label at once (``contract.parse_and_redact``);
3. validate the redacted rows and the delivered contract against the Flow contract; any violation
   raises ``ContractViolationError`` and Crew 2 does not start;
4. partition the redacted rows with the P4 parser (``crew2.ml.data.partition``), so the TRAIN rows and
   the label-free test and demo feature rows equal the P4 input exactly. The P4 test vault is replaced
   by ``SealedVault``, which holds no label and refuses to open: the Flow never evaluates on TEST.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from crew2.ml import data as D

from . import contract as C

FORBIDDEN_DIR = C.REPO / "data" / "synthetic"


class PathPolicyError(PermissionError):
    pass


class TestAccessError(RuntimeError):
    __test__ = False                                  # not a pytest test class


class SealedVault:
    """Stands in for the P4 ``TestVault`` in the Flow: it holds no label and can never be opened."""

    __test__ = False
    opens: tuple = ()
    opened = False

    def open(self, *args, **kwargs):
        raise TestAccessError("the Flow never evaluates on TEST; the historical P4 TEST result is cited instead")


@dataclass(frozen=True)
class FlowHandoff:
    parts: D.Partitioned                   # train (with labels), test/demo features (no labels), SealedVault
    report: C.Report

    @property
    def train(self):
        return self.parts.train

    @property
    def counts(self) -> dict:
        return self.parts.counts


def _read(path: Path) -> bytes:
    p = Path(path)
    real = Path(os.path.realpath(p))
    forbidden = Path(os.path.realpath(FORBIDDEN_DIR))
    if real == forbidden or forbidden in real.parents:
        raise PathPolicyError("the Flow reads only Crew 1's delivered files, never the raw export")
    if p.is_symlink() or not real.is_file():
        raise PathPolicyError(f"not a regular file: {p.name}")
    return real.read_bytes()


def validate_handoff(clean_csv: Path, contract_json: Path, flow_contract_json: Path | None = None) -> C.Report:
    """The Flow contract report for the delivered files (no exception; see ``report.ok``)."""
    report, _, _ = _validated(clean_csv, contract_json, flow_contract_json)
    return report


def _validated(clean_csv, contract_json, flow_contract_json):
    data = _read(clean_csv)
    clean_sha = C.sha256_hex(data)
    header, rows = C.parse_and_redact(data)            # non-train labels are gone from here on
    del data
    base_bytes = _read(contract_json)
    flow = json.loads(_read(flow_contract_json).decode("utf-8")) if flow_contract_json is not None else None
    return C.validate(header, rows, clean_sha, base_bytes, flow), header, rows


def load_handoff(clean_csv: Path, contract_json: Path, flow_contract_json: Path | None = None) -> FlowHandoff:
    """Validated, label-redacted Crew-1 handoff, partitioned exactly as the P4 input."""
    report, header, rows = _validated(clean_csv, contract_json, flow_contract_json)
    if not report.ok:
        raise C.ContractViolationError(report)
    parts = D.partition(C.redacted_csv_text(header, rows))
    parts.vault = SealedVault()
    return FlowHandoff(parts, report)
