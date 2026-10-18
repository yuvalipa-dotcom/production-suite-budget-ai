"""Pre-publication scan: secrets and content, over every commit of every branch.

    python tools/publish_scan.py

* Secret scan: detect-secrets (all default plugins) on every file of every commit
  reachable from any local branch, plus the current working tree.
* Content scan: generic patterns (environment files, key/token shapes, database
  URLs, local absolute paths, code copied from a product codebase) plus an
  optional PRIVATE denylist read from ``scan_output/private_denylist.txt``.
  One entry per line: a case-insensitive literal; or ``re:<regex>`` (case-
  insensitive regular expression); or ``allow:<regex>``, which suppresses
  ``re:`` matches on a line it matches (never literal matches). The denylist
  lives in an ignored directory so the sensitive strings are never published.

Raw findings are written to ``scan_output/`` (git-ignored). Only counts are printed.
Exit code 0 = no findings, 1 = findings.
"""

from __future__ import annotations

import argparse
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


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


class Denylist:
    def __init__(self, lines: list[str]):
        self.literals = [ln.lower() for ln in lines if not ln.startswith(("re:", "allow:"))]
        self.patterns = [re.compile(ln[3:], re.I) for ln in lines if ln.startswith("re:")]
        self.allow = [re.compile(ln[6:], re.I) for ln in lines if ln.startswith("allow:")]

    def __len__(self) -> int:
        return len(self.literals) + len(self.patterns)

    def hits(self, line: str) -> list[str]:
        low = line.lower()
        out = [f"private_denylist[{i}]" for i, w in enumerate(self.literals) if w in low]
        if not any(a.search(line) for a in self.allow):
            out += [f"private_pattern[{i}]" for i, rx in enumerate(self.patterns) if rx.search(line)]
        return out


def _denylist(f: Path | None = None) -> Denylist:
    f = f or OUT / "private_denylist.txt"
    if not f.exists():
        return Denylist([])
    return Denylist([ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()])


def _content_findings(path: str, text: str, deny) -> list[dict]:
    if isinstance(deny, list):
        deny = Denylist(deny)
    findings = []
    if GENERIC_PATTERNS["env_file_name"].search(path) and not path.endswith(".gitignore"):
        findings.append({"file": path, "line": 0, "rule": "env_file_name"})
    for n, line in enumerate(text.splitlines(), 1):
        if path != SELF:
            for rule, rx in GENERIC_PATTERNS.items():
                if rule != "env_file_name" and rx.search(line):
                    findings.append({"file": path, "line": n, "rule": rule})
        for rule in deny.hits(line):
            findings.append({"file": path, "line": n, "rule": rule})
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


def _scan_tree(files: dict[str, bytes], deny: Denylist, label: str) -> dict:
    secrets, content = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for logical, blob in files.items():
            disk = Path(tmp) / logical
            disk.parent.mkdir(parents=True, exist_ok=True)
            disk.write_bytes(blob)
            secrets += _secret_findings(disk, logical)
            content += _content_findings(logical, blob.decode("utf-8", errors="replace"), deny)
    return {"tree": label, "files": len(files), "secret_findings": secrets, "content_findings": content}


def scan_history(repo: Path, deny: Denylist) -> tuple[list[dict], int, int]:
    """Scan every file of every commit reachable from any local branch.

    Each distinct (blob, path) pair is scanned once and its result is attributed to every
    commit that contains it, so coverage equals scanning every tree in full.
    Returns (per-commit results, number of commits, number of distinct blob/path pairs).
    """
    commits = _git(repo, "rev-list", "--branches").split()
    cache: dict[tuple[str, str], dict] = {}
    results = []
    for sha in commits:
        entries = [ln.split("\t", 1) for ln in _git(repo, "ls-tree", "-r", sha).splitlines() if ln]
        secrets, content = [], []
        for meta, path in entries:
            blob = meta.split()[2]
            key = (blob, path)
            if key not in cache:
                data = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", blob], check=True,
                                      capture_output=True).stdout
                cache[key] = _scan_tree({path: data}, deny, "blob")
            secrets += cache[key]["secret_findings"]
            content += cache[key]["content_findings"]
        results.append({"tree": f"commit {sha[:12]}", "files": len(entries),
                        "secret_findings": secrets, "content_findings": content})
    return results, len(commits), len(cache)


def scan_worktree(root: Path, deny: Denylist, label: str) -> dict:
    """Scan every file on disk under ``root`` (the .git directory excluded)."""
    files = {p.relative_to(root).as_posix(): p.read_bytes()
             for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}
    return _scan_tree(files, deny, label)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--denylist", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT / "publish_scan_raw.json",
                    help="raw output file (must be outside any published tree)")
    ap.add_argument("--worktree-label", default=None,
                    help="also scan the files on disk in --repo under this label")
    args = ap.parse_args()
    deny = _denylist(args.denylist)
    results, n_commits, n_blobs = scan_history(args.repo, deny)
    if args.worktree_label:
        results.append(scan_worktree(args.repo, deny, args.worktree_label))
    else:
        tracked = [n for n in _git(args.repo, "ls-files").splitlines() if n]
        results.append(_scan_tree({n: (args.repo / n).read_bytes() for n in tracked}, deny,
                                  "working tree (tracked)"))
    total_s = sum(len(r["secret_findings"]) for r in results)
    total_c = sum(len(r["content_findings"]) for r in results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"commits_scanned": n_commits, "distinct_blob_path_pairs": n_blobs,
                      "trees_scanned": len(results), "private_denylist_entries": len(deny),
                      "secret_findings": total_s, "content_findings": total_c}, indent=2))
    return 0 if total_s == 0 and total_c == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
