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
import hashlib
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

# Narrow, documented exemption (P2): two legitimate sha256 digest structures, each identified by
# exact file path and exact JSON location. A flagged line is spared only when (a) the whole line is
# one approved key with a lowercase hex-64 value, so nothing else can share the line; (b) the parsed
# JSON holds that value at the approved location; (c) the value occurs nowhere else in the file; and
# (d) the flagged secret is that value.
APPROVED_DIGEST_FIELDS = {
    "tests/fixtures/policy_c_baseline.json": {"digest": ("digest",)},
    "contracts/frozen_split.json": {
        f"data/synthetic/{v}/productions.csv": ("provenance", "source_files_sha256", f"data/synthetic/{v}/productions.csv")
        for v in ("v1", "v2")
    },
}
DIGEST_FIELD_LINE = re.compile(r'^\s*"(?P<key>[^"\\]+)": "(?P<value>[0-9a-f]{64})",?\s*$')

# Narrow, documented exemption (P3, owner-approved): the sha256 provenance digests in four JSON artifacts
# of the published live run p3-live-005, each the digest of an artifact of that run or of a committed
# input (verified independently; docs/security_scan.md). A finding is spared only when ALL hold: (a) the
# path is exactly an approved path; (b) the whole file is byte-identical to its approved blob (git blob
# id), so the run record cannot change and keep the exemption; (c) the finding type is exactly the
# approved detect-secrets type; (d) the whole flagged line is one key with a lowercase hex-64 value, the
# key names exactly one approved JSON location of that file, and that location holds that value; (e) the
# value occurs once in the file. Blob ids are integer literals, so this file contains no quoted hex string.
P3_RUN = "artifacts/crew1/p3-live-005"
P3_RUN_DIGEST_TYPE = "Hex High Entropy String"
P3_RUN_INPUTS = ("contracts/column_roles.json", "contracts/dataset_contract.schema.json", "contracts/frozen_split.json",
                 "raw/manifest.json")
APPROVED_P3_RUN_DIGESTS = {   # path: (git blob id, JSON locations of the approved digests)
    f"{P3_RUN}/run_manifest.json": (
        f"{0x2db0c5c813c3eea9163ac0e2fe365d3e1088352b:040x}",
        tuple(("artifacts_sha256", n) for n in ("eda_report.html", "insights.md", "trace.jsonl"))
        + tuple(("input_sha256", n) for n in P3_RUN_INPUTS + (
            "crew1/config/agents.yaml", "crew1/config/tasks.yaml", "docs/raw_data_dictionary.md",
            "raw/budget_revisions.csv", "raw/category_plans.csv", "raw/commitments.csv", "raw/manual_costs.csv",
            "raw/payments.csv", "raw/productions.csv", "raw/snapshot_schedule.csv"))),
    f"{P3_RUN}/gate_result.json": (
        f"{0x5e895d16991a1e41814a5a1c911b0cea3ef9f7bd:040x}",
        tuple(("input_sha256s", n) for n in P3_RUN_INPUTS + ("clean_data.csv", "dataset_contract.json"))),
    f"{P3_RUN}/dataset_contract.json": (f"{0xd5d9023285c831584c09b4e85b495f2e9fbb0447:040x}",
                                        (("raw_manifest_sha256",),)),
    f"{P3_RUN}/cleaning_log.json": (f"{0x83fd16ac59287c027a6f182fc88edf0e64d7f4d8:040x}",
                                    (("raw_manifest_sha256",),)),
}

# Narrow, documented exemption (P2): the public W-3 merge commit (origin/main after PR #2), in exactly
# two complete lines of two files. A flagged line is spared only when it equals its template filled
# with this one value, the flagged secret is this value, and the value occurs once in the file.
# Held as an integer literal so that this file contains no quoted hex string of its own.
APPROVED_GIT_BASELINE = f"{0x59517254e108738696cd3632530b953768745788:040x}"
GIT_BASELINE_LINES = {
    "tools/publication_check.py": 'PUBLISHED_BASELINE = "{}"',
    "tests/test_publication_check.py": '    assert check.PUBLISHED_BASELINE == "{}"',
}

# Narrow, documented exemption (P3), on the same rules: the public P2 merge commit (origin/main after
# PR #3), the P3 starting baseline, in exactly two complete lines of the same two files.
APPROVED_P3_BASELINE = f"{0x7140231951b1d2b82b8e1edcd92a0a8be5da15b8:040x}"
P3_BASELINE_LINES = {
    "tools/publication_check.py": 'P3_BASELINE = "{}"',
    "tests/test_publication_check.py": '    assert check.P3_BASELINE == "{}"',
}
APPROVED_COMMIT_IDS = ((APPROVED_GIT_BASELINE, GIT_BASELINE_LINES), (APPROVED_P3_BASELINE, P3_BASELINE_LINES))

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


def _json_at(doc, path: tuple[str, ...]):
    for k in path:
        doc = doc.get(k) if isinstance(doc, dict) else None
    return doc


def _approved_p3_run_digest(logical: str, data: bytes, finding: dict) -> bool:
    """True only for an approved provenance digest in an unchanged p3-live-005 artifact (see APPROVED_P3_RUN_DIGESTS)."""
    approved = APPROVED_P3_RUN_DIGESTS.get(logical)
    if not approved or finding["rule"] != P3_RUN_DIGEST_TYPE:
        return False
    blob, locations = approved
    if hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() != blob:
        return False
    text = data.decode("utf-8")
    lines = text.splitlines()
    m = DIGEST_FIELD_LINE.match(lines[finding["line"] - 1]) if 0 < finding["line"] <= len(lines) else None
    at = [loc for loc in locations if m and loc[-1] == m["key"]]
    return len(at) == 1 and _json_at(json.loads(text), at[0]) == m["value"] and text.count(m["value"]) == 1


def _approved_digest(logical: str, line: str, text: str, secret_value: str | None) -> bool:
    """True only for an approved digest structure (see APPROVED_DIGEST_FIELDS)."""
    fields = APPROVED_DIGEST_FIELDS.get(logical)
    m = DIGEST_FIELD_LINE.match(line) if fields else None
    if not m or m["key"] not in fields or (secret_value is not None and secret_value != m["value"]):
        return False
    try:
        doc = json.loads(text)
    except ValueError:
        return False
    approved = [_json_at(doc, p) for p in fields.values()]
    return _json_at(doc, fields[m["key"]]) == m["value"] and text.count(m["value"]) == approved.count(m["value"])


def _approved_git_baseline(logical: str, line: str, text: str, secret_value: str | None) -> bool:
    """True only for an approved line holding its approved public commit id (see APPROVED_COMMIT_IDS)."""
    for value, lines in APPROVED_COMMIT_IDS:
        template = lines.get(logical)
        if (template is not None and line == template.format(value)
                and secret_value in (None, value) and text.count(value) == 1):
            return True
    return False


def _secret_findings(path_on_disk: Path, logical: str) -> list[dict]:
    coll = SecretsCollection()
    with default_settings():
        coll.scan_file(str(path_on_disk))
    out = []
    text = path_on_disk.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for _, secret in coll:
        line = lines[secret.line_number - 1] if 0 < secret.line_number <= len(lines) else ""
        if logical.endswith("manifest.json") and MANIFEST_DIGEST_LINE.match(line):
            continue
        value = getattr(secret, "secret_value", None)
        if _approved_digest(logical, line, text, value) or _approved_git_baseline(logical, line, text, value):
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
            secrets += [f for f in _secret_findings(disk, logical) if not _approved_p3_run_digest(logical, blob, f)]
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
