"""The P4 scanner exemption is exact, and every other secret-like value is still detected.

Owner-approved (P4 security closure): the quoted sha256 digest of the deterministic Policy C clean
data in ONE historical file version, ``crew2/ml/data.py`` of commit b2146c5. Planted values are
assembled at runtime, so this file contains no secret-shaped literal. No real key is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_scan", REPO / "tools" / "publish_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

COMMIT = "b2146c5"
PATH = scan.P4_CLEAN_DIGEST_PATH
VALUE = scan.APPROVED_P4_CLEAN_DIGEST
OTHER_DIGEST = hashlib.sha256(b"planted-p4-other").hexdigest()
FAKE_GH = "gh" + "p_" + "Zq7" * 12


def _git(*args) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True).stdout


@pytest.fixture(scope="module")
def historical() -> bytes:
    try:
        return _git("show", f"{COMMIT}:{PATH}")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history unavailable")


def _findings(tmp_path, logical: str, data: bytes) -> list[dict]:
    disk = tmp_path / "scanned.py"
    disk.write_bytes(data)
    return scan._secret_findings(disk, logical)


# ---------------------------------------------------------------- the exemption is exactly this value

def test_the_approved_value_is_the_digest_of_the_deterministic_clean_data():
    """Independent verification: recompute the clean data in-process and hash it."""
    from crew2.ml import data as D
    text, log = D.build_handoff()
    assert VALUE == hashlib.sha256(text.encode("utf-8")).hexdigest() == D.EXPECTED_CLEAN_SHA256
    assert log["snapshots"]["eligible"] == 1538


def test_the_approved_blob_is_the_historical_file_version(historical):
    assert _git("rev-parse", f"{COMMIT}:{PATH}").decode().strip() == scan.P4_CLEAN_DIGEST_BLOB
    assert scan.git_blob_id(historical) == scan.P4_CLEAN_DIGEST_BLOB
    assert historical.decode().splitlines()[35] == scan.P4_CLEAN_DIGEST_LINE.format(VALUE)


def test_the_historical_finding_is_exempted(tmp_path, historical):
    assert _findings(tmp_path, PATH, historical) == []
    # the detector does flag it: only the exemption spares it
    flagged = _findings(tmp_path, "elsewhere/data.py", historical)
    assert [(f["line"], f["rule"]) for f in flagged] == [(36, scan.P4_CLEAN_DIGEST_TYPE)]


def test_the_current_file_needs_no_exemption(tmp_path):
    assert _findings(tmp_path, PATH, (REPO / PATH).read_bytes()) == []
    assert _findings(tmp_path, "elsewhere/data.py", (REPO / PATH).read_bytes()) == []


# ---------------------------------------------------------------- everything else is still detected

def test_the_same_file_at_another_path_is_detected(tmp_path, historical):
    for logical in ("crew2/ml/data_copy.py", "crew2/data.py", "tools/publish_scan.py", "docs/crew2.md"):
        assert _findings(tmp_path, logical, historical), logical


def test_a_different_digest_in_the_approved_line_and_path_is_detected(tmp_path, historical):
    swapped = historical.replace(VALUE.encode(), OTHER_DIGEST.encode())
    assert [f["line"] for f in _findings(tmp_path, PATH, swapped)] == [36]


def test_any_other_change_to_the_file_voids_the_exemption(tmp_path, historical):
    assert _findings(tmp_path, PATH, historical + b"# one more line\n")
    edited = historical.replace(b"Policy C", b"Policy-C", 1)
    assert edited != historical and len(edited) == len(historical)        # same size, one byte different
    assert _findings(tmp_path, PATH, edited)


def test_a_repeated_value_is_detected(tmp_path, historical):
    repeated = historical + f'SECOND = "{VALUE}"\n'.encode()
    assert len(_findings(tmp_path, PATH, repeated)) >= 1


def test_a_different_secret_like_value_in_the_approved_file_is_still_detected(tmp_path, historical):
    """Regression (owner step 6): a deliberately introduced, synthetic secret-like value is caught, in the
    approved path, next to the approved digest."""
    planted = historical + f'GITHUB_TOKEN = "{FAKE_GH}"\nOTHER_DIGEST = "{OTHER_DIGEST}"\n'.encode()
    found = _findings(tmp_path, PATH, planted)
    n = len(historical.decode().splitlines())
    assert {n + 1, n + 2} <= {f["line"] for f in found}
    content = scan._content_findings(PATH, planted.decode(), [])
    assert "github_token" in {f["rule"] for f in content}


def test_the_exemption_requires_the_exact_finding_type_value_and_line(historical):
    line = scan.P4_CLEAN_DIGEST_LINE.format(VALUE)
    assert scan._approved_p4_clean_digest(PATH, line, historical, scan.P4_CLEAN_DIGEST_TYPE, VALUE)
    assert not scan._approved_p4_clean_digest(PATH, line, historical, "Secret Keyword", VALUE)
    assert not scan._approved_p4_clean_digest(PATH, line, historical, scan.P4_CLEAN_DIGEST_TYPE, OTHER_DIGEST)
    assert not scan._approved_p4_clean_digest(PATH, line, historical, scan.P4_CLEAN_DIGEST_TYPE, None)
    assert not scan._approved_p4_clean_digest(PATH, "  " + line, historical, scan.P4_CLEAN_DIGEST_TYPE, VALUE)
    assert not scan._approved_p4_clean_digest("crew2/ml/Data.py", line, historical, scan.P4_CLEAN_DIGEST_TYPE,
                                              VALUE)


def test_the_earlier_exemptions_are_unchanged():
    assert scan.APPROVED_DIGEST_FIELDS.keys() == {"tests/fixtures/policy_c_baseline.json", "contracts/frozen_split.json"}
    assert len(scan.APPROVED_COMMIT_IDS) == 2
    assert scan.GENERIC_PATTERNS["github_token"].search(FAKE_GH)
