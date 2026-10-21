"""P5-B: the file-only boundary between Crew 1 and Crew 2's Flow path. Offline. ALL DATA IS SYNTHETIC.

* ``load_handoff`` on the approved Crew-1 files equals the P4 input exactly (TRAIN rows and the label-free
  test and demo feature rows);
* no non-train outcome label reaches any Flow component, and the test vault cannot be opened;
* importing and running the Flow handoff path loads no Crew 1 module (not even through a package
  ``__init__``) and opens no file of the raw export.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from budget_flow import contract as C
from budget_flow import handoff as FH
from tests.conftest import REPO

APPROVED = REPO / "artifacts" / "crew1" / "p3-live-005"


@pytest.fixture
def delivered(tmp_path):
    """Crew 1's two files, copied outside the repository as a Flow run would hand them over."""
    for name in ("clean_data.csv", "dataset_contract.json"):
        shutil.copyfile(APPROVED / name, tmp_path / name)
    return tmp_path / "clean_data.csv", tmp_path / "dataset_contract.json"


# ---------------------------------------------------------------- equivalence with the P4 input

def test_the_approved_files_equal_the_p4_input(delivered):
    from crew2.ml import data as D
    h = FH.load_handoff(*delivered)
    ref = D.partition(D.build_handoff()[0])                  # the historical in-process P4 input
    for name in ("train", "test_features", "demo_features"):
        pd.testing.assert_frame_equal(getattr(h.parts, name), getattr(ref, name), check_exact=True)
    assert h.counts == ref.counts and h.counts["train"]["rows"] == 1084
    assert h.report.ok and h.report.clean_data_sha256 == D.EXPECTED_CLEAN_SHA256


# ---------------------------------------------------------------- test labels never reach the Flow

def test_non_train_labels_are_redacted_before_any_check(delivered, monkeypatch):
    seen = {}
    real = C.validate

    def spy(header, rows, *args, **kwargs):
        seen["rows"] = rows
        return real(header, rows, *args, **kwargs)

    monkeypatch.setattr(C, "validate", spy)
    h = FH.load_handoff(*delivered)
    non_train = [r for r in seen["rows"] if r["split"] != "train"]
    assert len(non_train) == 1538 - 1084
    assert all(r[c] is None for r in non_train for c in C.LABEL_COLUMNS)
    assert all(r[c] not in (None, "") for r in seen["rows"] if r["split"] == "train" for c in C.LABEL_COLUMNS)
    for name in ("test_features", "demo_features"):
        assert not set(C.LABEL_COLUMNS) & set(getattr(h.parts, name).columns)


def test_the_redacted_text_given_to_the_p4_parser_holds_no_non_train_label(delivered, monkeypatch):
    from crew2.ml import data as D
    captured = {}
    real = D.partition

    def spy(text):
        captured["text"] = text
        return real(text)

    monkeypatch.setattr(D, "partition", spy)
    FH.load_handoff(*delivered)
    rows = list(__import__("csv").DictReader(__import__("io").StringIO(captured["text"])))
    assert all(r[c] == "" for r in rows if r["split"] != "train" for c in C.LABEL_COLUMNS)


def test_the_test_vault_is_sealed(delivered):
    h = FH.load_handoff(*delivered)
    assert isinstance(h.parts.vault, FH.SealedVault) and not h.parts.vault.opened
    with pytest.raises(FH.TestAccessError):
        h.parts.vault.open({"primary": "0" * 64})


def test_a_raw_export_path_is_refused():
    raw = REPO / "data" / "synthetic" / "v2" / "raw" / "productions.csv"
    with pytest.raises(FH.PathPolicyError):
        FH.load_handoff(raw, APPROVED / "dataset_contract.json")


# ---------------------------------------------------------------- import isolation

def test_no_flow_module_imports_crew1_statically():
    for path in sorted((REPO / "budget_flow").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n == "crew1" or n.startswith("crew1.") for n in names), (path.name, names)


_CHILD = r"""
import importlib.abc, json, os, sys
sys.path.insert(0, REPO)
blocked = []
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("crew1", "crewai"):
            blocked.append(name)
            raise ImportError("import of " + name + " is forbidden in the Flow handoff path")
        return None
sys.meta_path.insert(0, Block())
opened = []
norm = lambda p: os.path.realpath(p).replace("\\", "/").lower()
FORBIDDEN = [norm(os.path.join(REPO, d)) for d in ("data", "crew1")]
def hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        p = norm(os.fsdecode(args[0]))
        opened.append(p)
        if any(p == f or p.startswith(f + "/") for f in FORBIDDEN) or "/raw/" in p:
            raise PermissionError("forbidden read in the Flow handoff path: " + p)
sys.addaudithook(hook)
from budget_flow.handoff import load_handoff
h = load_handoff(CLEAN, CONTRACT)
print(json.dumps({"ok": h.report.ok, "train": h.counts["train"]["rows"], "blocked": blocked,
                  "crew_modules": sorted(m for m in sys.modules if m.split(".")[0] in ("crew1", "crewai")),
                  "read_inputs": [p for p in opened if p in (norm(CLEAN), norm(CONTRACT))]}))
"""


def _run_child(clean, contract) -> dict:
    code = (f"REPO = {str(REPO)!r}\nCLEAN = {str(clean)!r}\nCONTRACT = {str(contract)!r}\n") + _CHILD
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(clean.parent))
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_the_flow_handoff_path_loads_no_crew1_module_and_reads_no_raw_file(delivered):
    out = _run_child(*delivered)
    assert out["ok"] and out["train"] == 1084
    assert out["blocked"] == [] and out["crew_modules"] == []
    assert len(out["read_inputs"]) == 2                      # it did read exactly Crew 1's two delivered files


def test_the_isolation_harness_does_catch_a_forbidden_read(tmp_path):
    """Control: the same child process fails when it is pointed at a file of the raw export."""
    raw = REPO / "data" / "synthetic" / "v2" / "raw" / "productions.csv"
    code = (f"REPO = {str(REPO)!r}\nCLEAN = {str(raw)!r}\nCONTRACT = {str(raw)!r}\n"
            "import sys; sys.path.insert(0, REPO)\n"
            "import os\n"
            + _CHILD.split("from budget_flow.handoff")[0]
            + "open(CLEAN).read(1)\n")
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert res.returncode != 0 and "forbidden read" in res.stderr


def test_importing_crew2_no_longer_imports_crew1():
    code = ("import sys; sys.path.insert(0, %r)\nimport crew2, crew2.ml.protocol, crew2.ml.data, crew2.ml.leakage\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('crew1', 'crewai')))" % str(REPO))
    res = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip() == "[]"
