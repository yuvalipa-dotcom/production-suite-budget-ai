"""Pre-publication scan: secrets and content, over every commit of every branch.

    python tools/publish_scan.py

* Secret scan: detect-secrets (all default plugins) on every file of every commit
  reachable from any local branch, plus the current working tree.
* Content scan: generic patterns (environment files, key/token shapes, database
  URLs, local absolute paths, code copied from a product codebase) plus an
  optional PRIVATE denylist read from ``scan_output/private_denylist.txt``
  (one case-insensitive literal per line). The denylist lives in an ignored
  directory so the sensitive strings themselves are never published.

Raw findings are written to ``scan_output/`` (git-ignored). Only counts are printed.
Exit code 0 = no findings, 1 = findings.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.settings import default_settings

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "scan_output"

# Narrow, documented exclusion: the dataset manifest lists sha256 digests of the
# data files, which are high-entropy hex by design and are not secrets.
MANIFEST_DIGEST_LINE = re.compile(r'^\s*"[a-z_]+\.(csv|json)": "[0-9a-f]{64}",?\s*$')

GENERIC_PATTERNS = {
    "env_file_name": re.compile(r"(^|/)\.env(\.|$)"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "anthropic_or_openai_key": re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{20,}"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    "supabase_url": re.compile(r"[a-z0-9]{15,}\.supabase\.(co|in)", re.I),
    "service_role": re.compile(r"service[_-]?role", re.I),
    "postgres_url": re.compile(r"postgres(ql)?://", re.I),
    "local_windows_path": re.compile(r"[A-Za-z]:\\\\?Users\\\\?", re.I),
    "local_unix_home": re.compile(r"/(home|Users)/[A-Za-z0-9_.-]+/"),
    "product_code_import": re.compile(r"""from ["']@/|require\(["']@/|import ["']server-only["']"""),
}

SELF = "tools/publish_scan.py"   # this file necessarily contains the generic patterns


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def _denylist() -> list[str]:
    f = OUT / "private_denylist.txt"
    if not f.exists():
        return []
    return [ln.strip().lower() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _content_findings(path: str, text: str, deny: list[str]) -> list[dict]:
    findings = []
    if GENERIC_PATTERNS["env_file_name"].search(path) and not path.endswith(".gitignore"):
        findings.append({"file": path, "line": 0, "rule": "env_file_name"})
    for n, line in enumerate(text.splitlines(), 1):
        if path != SELF:
            for rule, rx in GENERIC_PATTERNS.items():
                if rule != "env_file_name" and rx.search(line):
                    findings.append({"file": path, "line": n, "rule": rule})
        low = line.lower()
        for i, word in enumerate(deny):
            if word in low:
                findings.append({"file": path, "line": n, "rule": f"private_denylist[{i}]"})
    return findings


def _secret_findings(path_on_disk: Path, logical: str) -> list[dict]:
    coll = SecretsCollection()
    with default_settings():
        coll.scan_file(str(path_on_disk))
    out = []
    lines = path_on_disk.read_text(encoding="utf-8", errors="replace").splitlines()
    for _, secret in coll:
        line = lines[secret.line_number - 1] if 0 < secret.line_number <= len(lines) else ""
        if logical.endswith("manifest.json") and MANIFEST_DIGEST_LINE.match(line):
            continue
        out.append({"file": logical, "line": secret.line_number, "rule": secret.type})
    return out


def _scan_tree(files: dict[str, bytes], deny: list[str], label: str) -> dict:
    secrets, content = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for logical, blob in files.items():
            disk = Path(tmp) / logical
            disk.parent.mkdir(parents=True, exist_ok=True)
            disk.write_bytes(blob)
            secrets += _secret_findings(disk, logical)
            content += _content_findings(logical, blob.decode("utf-8", errors="replace"), deny)
    return {"tree": label, "files": len(files), "secret_findings": secrets, "content_findings": content}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    deny = _denylist()
    results = []
    commits = _git("rev-list", "--branches").split()
    for sha in commits:
        names = _git("ls-tree", "-r", "--name-only", sha).split("\n")
        files = {n: subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:{n}"], check=True,
                                   capture_output=True).stdout for n in names if n}
        results.append(_scan_tree(files, deny, f"commit {sha[:12]}"))
    tracked = [n for n in _git("ls-files").split("\n") if n]
    results.append(_scan_tree({n: (REPO / n).read_bytes() for n in tracked}, deny, "working tree (tracked)"))
    total_s = sum(len(r["secret_findings"]) for r in results)
    total_c = sum(len(r["content_findings"]) for r in results)
    (OUT / "publish_scan_raw.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"commits_scanned": len(commits), "trees_scanned": len(results),
                      "private_denylist_entries": len(deny),
                      "secret_findings": total_s, "content_findings": total_c}, indent=2))
    return 0 if total_s == 0 and total_c == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
