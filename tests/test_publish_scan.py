"""The pre-publication scan must catch what it should, and pass healthy content.

Planted values are assembled at runtime so that this file itself contains no
secret-shaped literal.
"""

import base64
import hashlib
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("publish_scan", REPO / "tools" / "publish_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

FAKE_GH = "gh" + "p_" + "Ab1" * 12


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


FAKE_JWT = ".".join([_b64(b'{"alg":"HS256","typ":"JWT"}'), _b64(b'{"sub":"planted-test-value"}'),
                     _b64(hashlib.sha256(b"planted").digest())])
FAKE_DB = "postgres" + "ql://" + "user:pw@host:5432/db"
DIGEST = hashlib.sha256(b"planted").hexdigest()


def rules(findings):
    return {f["rule"] for f in findings}


def test_content_scan_catches_planted_values():
    text = "\n".join([f"token = {FAKE_GH}", f"auth: {FAKE_JWT}", f"url = {FAKE_DB}",
                      "path C:" + "\\Users\\someone\\x", "import x from '" + "@/lib/x'"])
    found = rules(scan._content_findings("docs/note.md", text, []))
    assert {"github_token", "jwt", "postgres_url", "local_windows_path", "product_code_import"} <= found


def test_content_scan_catches_env_files_and_private_denylist():
    assert "env_file_name" in rules(scan._content_findings("config/." + "env", "", []))
    assert "env_file_name" in rules(scan._content_findings("." + "env.local", "", []))
    assert "private_denylist[0]" in rules(scan._content_findings("a.md", "see Example-Corp here", ["example-corp"]))


def test_content_scan_passes_healthy_content():
    healthy = "production_id,stage,budget_pre_vat\nSYN-0001,S0,254237.29\n"
    assert scan._content_findings("data/synthetic/v1/snapshots.csv", healthy, ["example-corp"]) == []
    assert scan._content_findings(".gitignore", "." + "env*\n", []) == []


def test_secret_scan_flags_a_token_and_spares_only_manifest_digests(tmp_path):
    secret_file = tmp_path / "notes.txt"
    secret_file.write_text(f'github_token = "{FAKE_GH}"\n', encoding="utf-8")
    assert scan._secret_findings(secret_file, "notes.txt")

    manifest = tmp_path / "manifest.json"
    manifest.write_text('{\n  "files_sha256": {\n    "snapshots.csv": "' + DIGEST + '"\n  }\n}\n', encoding="utf-8")
    assert scan._secret_findings(manifest, "data/synthetic/v1/manifest.json") == []

    elsewhere = tmp_path / "other.json"
    elsewhere.write_text('{\n  "api_secret": "' + DIGEST + '"\n}\n', encoding="utf-8")
    assert scan._secret_findings(elsewhere, "other.json"), "the same digest outside the manifest must be flagged"


def test_private_patterns_with_allow_list():
    deny = scan.Denylist([r"re:\bzork\b", r"re:zork[-_]", r"allow:unzork|zorkish", "plainword"])
    assert len(deny) == 3
    assert "private_pattern[0]" in rules(scan._content_findings("a.md", "met Zork today", deny))
    assert "private_pattern[1]" in rules(scan._content_findings("a.md", "zork-team", deny))
    assert scan._content_findings("a.md", "unzork_x is ok", deny) == []      # allow-listed line
    assert scan._content_findings("a.md", "zorkless word", deny) == []          # word boundary respected
    # allow never suppresses a literal entry
    assert "private_denylist[0]" in rules(scan._content_findings("a.md", "unzork plainword", deny))
