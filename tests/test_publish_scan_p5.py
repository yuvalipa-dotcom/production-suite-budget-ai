"""The P5-F scanner exemption is exact, and every other secret-like value is still detected.

Proposed for owner approval (P5-F): the sha256 provenance digests in 29 JSON records of the six canonical
Flow runs under ``flow_evidence/runs/``. Each spared value is the sha256 of a file, a pinned P4 model digest,
or a digest that a fresh run reproduces (the scripted-LLM request digests and the tampered-copy digests of
the break runs: asserted in tests/test_flow_e2e.py). Planted values are assembled at runtime, so this file
contains no secret-shaped literal. No real key is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_scan", REPO / "tools" / "publish_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

FILES = tuple(scan.APPROVED_P5_FLOW_DIGESTS)
OTHER_DIGEST = hashlib.sha256(b"planted-p5-other").hexdigest()
FAKE_GH = "gh" + "p_" + "Pf5" * 12
TYPE = "Hex High Entropy String"


def _tree_secrets(files: dict) -> list:
    return scan._scan_tree(files, scan.Denylist([]), "t")["secret_findings"]


def _bytes(logical: str) -> bytes:
    p = REPO / logical
    if not p.is_file():
        pytest.skip("the canonical Flow evidence is not in this checkout")
    return p.read_bytes()


def _flagged(logical: str, tmp_path) -> list[dict]:
    disk = tmp_path / Path(logical).name
    disk.write_bytes(_bytes(logical))
    return scan._secret_findings(disk, logical)


def test_p5_exemption_is_exact():
    assert scan.P5_EVIDENCE == "flow_evidence/runs" and scan.P5_DIGEST_TYPE == TYPE
    runs = {f.split("/")[2] for f in FILES}
    assert runs == {"p5e-run-001", *(f"p5e-break-{s}" for s in ("column_rename", "money_x100", "row_deletion",
                                                                   "contract_column_deletion", "label_leakage"))}
    assert len(FILES) == 29 and all(f.startswith("flow_evidence/runs/") and f.endswith(".json") for f in FILES)
    assert all(len(blob) == 40 for blob, _ in scan.APPROVED_P5_FLOW_DIGESTS.values())


def test_p5_files_are_flagged_unexempted_and_spared_only_by_the_exemption(tmp_path):
    total = 0
    for logical in FILES:
        data = _bytes(logical)
        assert scan.git_blob_id(data) == scan.APPROVED_P5_FLOW_DIGESTS[logical][0], logical
        flagged = _flagged(logical, tmp_path)
        assert flagged and all(f["rule"] == TYPE for f in flagged), logical
        total += len(flagged)
        assert _tree_secrets({logical: data}) == [], logical
    assert total == 248


def test_p5_every_spared_value_is_an_independently_checkable_digest(tmp_path):
    """Each spared value is the sha256 of a file in the publish set, a pinned P4 model digest, or a digest that
    tests/test_flow_e2e.py proves a fresh run reproduces (request digests; tampered copies of a break run)."""
    from budget_flow.p4_reference import HIST_MODEL_SHA256
    names = subprocess.run(["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard"], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    file_digests = {hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in names if (REPO / n).is_file()}
    kinds = {"file": 0, "p4_model": 0, "request": 0, "tampered_copy": 0}
    for logical in FILES:
        text = _bytes(logical).decode("utf-8")
        lines = text.splitlines()
        for f in _flagged(logical, tmp_path):
            m = scan.DIGEST_FIELD_LINE.match(lines[f["line"] - 1])
            locs = list(scan._json_paths_of(json.loads(text), m["value"]))
            if m["value"] in file_digests:
                kinds["file"] += 1
            elif any("model_sha256" in loc for loc in locs):
                assert m["value"] in HIST_MODEL_SHA256.values(), (logical, f)
                kinds["p4_model"] += 1
            elif m["key"] == "request_sha256":
                kinds["request"] += 1
            else:
                assert "/p5e-break-" in logical and logical.endswith("handoff/gate_report.json"), (logical, f)
                assert all(loc[0] == "validated_files_sha256" for loc in locs)
                kinds["tampered_copy"] += 1
    assert kinds == {"file": 130, "p4_model": 4, "request": 107, "tampered_copy": 7}


def test_p5_exemption_requires_the_unchanged_file_at_its_path():
    logical = "flow_evidence/runs/p5e-run-001/flow_manifest.json"
    data = _bytes(logical)
    value = json.loads(data)["required_outputs_sha256"]["crew2/model.joblib"]
    # a different digest in an approved slot (same shape, same length)
    assert _tree_secrets({logical: data.replace(value.encode(), OTHER_DIGEST.encode())})
    # any other change to the file, even outside the digests
    assert _tree_secrets({logical: data.replace(b'"status": "PASS"', b'"status": "PAS5"')})
    # the same bytes at any other path: a local run, another run id, another name, another folder
    for other in ("out/flow_runs/p5e-run-001/flow_manifest.json", "flow_evidence/runs/p5e-run-002/flow_manifest.json",
                  "flow_evidence/runs/p5e-run-001/flow_manifest_copy.json", "flow_manifest.json",
                  "flow_evidence/runs/p5e-run-001/evidence.json", "artifacts/crew1/p3-live-005/run_manifest.json"):
        assert _tree_secrets({other: data}), other


def test_p5_files_still_detect_other_secrets():
    for logical in FILES:
        data = _bytes(logical)
        planted = data.replace(b"{", ('{\n  "token": "' + FAKE_GH + '",').encode(), 1)
        assert _tree_secrets({logical: planted}), logical
        assert "github_token" in {f["rule"] for f in scan._content_findings(logical, planted.decode("utf-8"), [])}
        hexed = data.replace(b"{", ('{\n  "api_secret": "' + OTHER_DIGEST + '",').encode(), 1)
        assert _tree_secrets({logical: hexed}), logical


def test_p5_exemption_never_spares_another_finding_type_or_line(tmp_path):
    logical = "flow_evidence/runs/p5e-run-001/crew2/run_manifest.json"
    data = _bytes(logical)
    line = _flagged(logical, tmp_path)[0]["line"]
    assert scan._approved_p5_flow_digest(logical, data, {"rule": TYPE, "line": line})
    for finding in ({"rule": "Secret Keyword", "line": line}, {"rule": "Base64 High Entropy String", "line": line},
                    {"rule": TYPE, "line": 1}, {"rule": TYPE, "line": 0}, {"rule": TYPE, "line": 10 ** 6}):
        assert not scan._approved_p5_flow_digest(logical, data, finding), finding


def test_p5_exemption_ignores_synthetic_files_at_the_approved_paths():
    """Without the exact approved bytes nothing is spared, whatever the structure."""
    for logical in FILES:
        fake = ('{\n  "files_sha256": {\n    "handoff/gate_report.json": "' + OTHER_DIGEST + '"\n  },\n'
                '  "llm_transport": {"exchanges": [\n    {\n      "request_sha256": "'
                + hashlib.sha256(b"planted-p5-request").hexdigest() + '"\n    }\n  ]}\n}\n').encode()
        assert len(_tree_secrets({logical: fake})) == 2, logical


def test_p5_location_patterns_are_strict():
    pats = scan.P5_DIGEST_LOCATIONS["crew1/crew_kickoff.json"]
    assert scan._p5_location_ok(("llm_transport", "exchanges", 3, "request_sha256"), pats)
    assert scan._p5_location_ok(("handoff_sha256", "clean_data.csv"), pats)
    for bad in (("llm_transport", "exchanges", "3", "request_sha256"),     # a key where a list index belongs
                ("llm_transport", "exchanges", True, "request_sha256"),
                ("llm_transport", "exchanges", 3, "api_key"),
                ("llm_transport", "exchanges", 3, "request_sha256", "x"),
                ("handoff_sha256", "a b"), ("handoff_sha256", "../x\n"), ("handoff_sha256",),
                ("other", "clean_data.csv")):
        assert not scan._p5_location_ok(bad, pats), bad


def test_earlier_exemptions_are_unchanged():
    assert len(scan.APPROVED_P3_RUN_DIGESTS) == 4 and scan.P3_RUN == "artifacts/crew1/p3-live-005"
    assert scan.P4_CLEAN_DIGEST_PATH == "crew2/ml/data.py" and len(scan.APPROVED_COMMIT_IDS) == 2
    assert set(scan.APPROVED_DIGEST_FIELDS) == {"tests/fixtures/policy_c_baseline.json", "contracts/frozen_split.json"}
