"""The live-run scanner exemption is exact, and every other secret-like value is still detected.

Owner-approved (live-LLM follow-up, 2026-10-28): the sha256 provenance digests in the 11 JSON records of the
documented live-LLM Flow run under ``flow_evidence/live/p5-live-001`` (a byte-for-byte copy of the verified run). Each
spared value is the sha256 of a file in the publish set, a pinned P4 model digest, or the digest of a recorded
provider request. Planted values are assembled at runtime, so this file contains no secret-shaped literal. No
real key is used.
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

RUN = "flow_evidence/live/p5-live-001"
FILES = tuple(scan.APPROVED_LIVE_FLOW_DIGESTS)
OTHER_DIGEST = hashlib.sha256(b"planted-live-other").hexdigest()
FAKE_GH = "gh" + "p_" + "Lv1" * 12
TYPE = "Hex High Entropy String"


def _tree_secrets(files: dict) -> list:
    return scan._scan_tree(files, scan.Denylist([]), "t")["secret_findings"]


def _bytes(logical: str) -> bytes:
    p = REPO / logical
    if not p.is_file():
        pytest.skip("the live Flow evidence is not in this checkout")
    return p.read_bytes()


def _flagged(logical: str, tmp_path) -> list[dict]:
    disk = tmp_path / Path(logical).name
    disk.write_bytes(_bytes(logical))
    return scan._secret_findings(disk, logical)


def test_live_exemption_is_exact():
    assert scan.LIVE_EVIDENCE == "flow_evidence/live"
    assert {f.split("/")[2] for f in FILES} == {"p5-live-001"}
    assert len(FILES) == 11 and all(f.startswith(RUN + "/") and f.endswith(".json") for f in FILES)
    assert set(scan.APPROVED_P5_FLOW_DIGESTS).isdisjoint(FILES)


def test_the_whole_live_run_scans_clean_and_only_through_the_exemption(tmp_path):
    files = {p.relative_to(REPO).as_posix(): p.read_bytes() for p in (REPO / RUN).rglob("*") if p.is_file()}
    if not files:
        pytest.skip("the live Flow evidence is not in this checkout")
    assert len(files) == 24 and _tree_secrets(files) == []
    total = 0
    for logical in FILES:
        assert scan.git_blob_id(files[logical]) == scan.APPROVED_LIVE_FLOW_DIGESTS[logical][0], logical
        flagged = _flagged(logical, tmp_path)
        assert flagged and all(f["rule"] == TYPE for f in flagged), logical
        total += len(flagged)
    assert total == 97


def test_live_every_spared_value_is_an_independently_checkable_digest(tmp_path):
    from budget_flow.p4_reference import HIST_MODEL_SHA256
    names = subprocess.run(["git", "-C", str(REPO), "ls-files", "-co", "--exclude-standard"], capture_output=True,
                           text=True, check=True).stdout.splitlines()
    file_digests = {hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in names if (REPO / n).is_file()}
    kinds = {"file": 0, "p4_model": 0, "request": 0}
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
            else:
                assert m["key"] == "request_sha256", (logical, f)
                kinds["request"] += 1
    assert kinds["request"] == 32 and kinds["p4_model"] == 4 and sum(kinds.values()) == 97, kinds


def test_live_exemption_requires_the_unchanged_file_at_its_path():
    logical = f"{RUN}/flow_manifest.json"
    data = _bytes(logical)
    value = json.loads(data)["required_outputs_sha256"]["crew2/model.joblib"]
    assert _tree_secrets({logical: data.replace(value.encode(), OTHER_DIGEST.encode())})
    assert _tree_secrets({logical: data.replace(b'"status": "PASS"', b'"status": "PAS5"')})
    for other in ("out/flow_runs/live-001/flow_manifest.json", "flow_evidence/live/p5-live-002/flow_manifest.json",
                  "flow_evidence/runs/p5-live-001/flow_manifest.json", f"{RUN}/flow_manifest_copy.json"):
        assert _tree_secrets({other: data}), other


def test_live_files_still_detect_other_secrets():
    for logical in FILES:
        data = _bytes(logical)
        planted = data.replace(b"{", ('{\n  "token": "' + FAKE_GH + '",').encode(), 1)
        assert _tree_secrets({logical: planted}), logical
        hexed = data.replace(b"{", ('{\n  "api_secret": "' + OTHER_DIGEST + '",').encode(), 1)
        assert _tree_secrets({logical: hexed}), logical


def test_live_exemption_never_spares_another_finding_type_or_line(tmp_path):
    logical = f"{RUN}/crew2/crew_kickoff.json"
    data = _bytes(logical)
    line = _flagged(logical, tmp_path)[0]["line"]
    assert scan._approved_p5_flow_digest(logical, data, {"rule": TYPE, "line": line})
    for finding in ({"rule": "Secret Keyword", "line": line}, {"rule": "Base64 High Entropy String", "line": line},
                    {"rule": TYPE, "line": 1}, {"rule": TYPE, "line": 0}, {"rule": TYPE, "line": 10 ** 6}):
        assert not scan._approved_p5_flow_digest(logical, data, finding), finding
