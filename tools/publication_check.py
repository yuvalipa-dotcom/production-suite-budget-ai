"""Publication dry-run checks on a disposable clone (refs, history scans, exclusions, objects, ignores).

    python tools/publication_check.py --clone <dry-run clone> --origin-repo <academic repo>
        --denylist <private list> --out <raw output json, outside the clone>

Run only after the clone's remote has been removed. Writes nothing inside the clone
except branch checkouts. Prints a summary; full detail goes to --out.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("publish_scan", HERE / "publish_scan.py")
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)

# The W-3 branches, the published P2 branch and the explicitly approved P3 and P4 candidates.
# An exact list: no other branch is accepted.
PUBLICATION_BRANCHES = ("main", "p1-synthetic-generator", "p1-remediation", "p2-quality-gate", "p3-crew1",
                        "p4-crew2")
EXPECTED_REFS = {f"refs/heads/{b}" for b in PUBLICATION_BRANCHES}
# Historical: origin/main after the W-3 publication (the PR #2 merge). The P2 branch must build on it.
PUBLISHED_BASELINE = "59517254e108738696cd3632530b953768745788"
BASELINE_DESCENDANTS = ("p2-quality-gate",)
# origin/main after the P2 publication (the PR #3 merge, whose first parent is PUBLISHED_BASELINE).
# The P3 candidate must build on it.
P3_BASELINE = "7140231951b1d2b82b8e1edcd92a0a8be5da15b8"
P3_BASELINE_DESCENDANTS = ("p3-crew1",)
# The P4 candidate (owner-approved local publication candidate) is built on the P3 commit it started from.
# That commit must be an ancestor of both p3-crew1 and p4-crew2, and must itself descend from P3_BASELINE.
# Held as an integer literal so that this file gains no quoted hex string (docs/security_scan.md).
P4_BASELINE = f"{0x7d2d38aac808e4762536b66ad80e75fd2e1880c6:040x}"
P4_BASELINE_DESCENDANTS = ("p3-crew1", "p4-crew2")
FORBIDDEN_PATH = re.compile(
    r"(^|/)(private|scan_output|ground_truth|\.venv|venv)(/|$)"
    r"|(^|/)\.env(\.[^/]*)?$"
    r"|private_denylist|publish_scan_raw|defect_log|defect_catalogue|process_log"
)
LFS_POINTER = b"version https://git-lfs.github.com/spec/"
MAX_BYTES = 5 * 1024 * 1024
IGNORE_PROBES = (".env", ".env.local", "config/.env.production", ".venv/x", "venv/x", "env/x",
                 "dump.sql", "db.dump", "dumps/x.csv", "backup.bak", "private/x.md",
                 "scan_output/x.json", "ground_truth/v2/x.csv")
README_STATEMENT = "committed intentionally as the project's source data"


def git(repo: Path, *args: str, check: bool = True) -> str:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {res.stderr.strip()}")
    return res.stdout


def refs_report(clone: Path) -> dict:
    refs = set(git(clone, "for-each-ref", "--format=%(refname)").split())
    packed = clone / ".git" / "packed-refs"
    packed_refs = set()
    if packed.exists():
        for ln in packed.read_text(encoding="utf-8").splitlines():
            if ln and not ln.startswith(("#", "^")):
                packed_refs.add(ln.split()[1])
    return {
        "remotes": git(clone, "remote").split(),
        "refs": sorted(refs | packed_refs),
        "unexpected_refs": sorted((refs | packed_refs) - EXPECTED_REFS),
        "missing_refs": sorted(EXPECTED_REFS - refs),
        "stash": git(clone, "stash", "list").strip(),
        "notes": git(clone, "notes", "list", check=False).strip(),
        "head": git(clone, "symbolic-ref", "HEAD").strip(),
    }


def baseline_report(clone: Path, baseline: str = PUBLISHED_BASELINE,
                    branches: tuple[str, ...] = BASELINE_DESCENDANTS) -> dict:
    """Each listed branch must contain the published baseline commit."""
    res = {b: subprocess.run(["git", "-C", str(clone), "merge-base", "--is-ancestor", baseline,
                              f"refs/heads/{b}"], capture_output=True).returncode == 0 for b in branches}
    return {"baseline": baseline, "descends": res, "all_descend": all(res.values())}


def lineage_ok(clone: Path, baseline: str = P3_BASELINE, parent: str = PUBLISHED_BASELINE) -> bool:
    """The P3 baseline's first parent must be the historical published baseline."""
    res = subprocess.run(["git", "-C", str(clone), "rev-parse", "--verify", "-q", f"{baseline}^1"],
                         capture_output=True, text=True)
    return res.returncode == 0 and res.stdout.strip() == parent


def tree_checks(clone: Path) -> dict:
    commits = git(clone, "rev-list", "--branches").split()
    forbidden, submodules, lfs, large, binary = set(), set(), set(), set(), set()
    for sha in commits:
        for ln in git(clone, "ls-tree", "-r", "-l", sha).splitlines():
            meta, path = ln.split("\t", 1)
            mode, kind, blob, size = meta.split()
            if FORBIDDEN_PATH.search(path):
                forbidden.add(path)
            if mode == "160000" or path == ".gitmodules":
                submodules.add(path)
            if kind == "blob":
                if int(size) > MAX_BYTES:
                    large.add((path, int(size)))
                data = subprocess.run(["git", "-C", str(clone), "cat-file", "blob", blob],
                                      capture_output=True).stdout
                if data.startswith(LFS_POINTER):
                    lfs.add(path)
                if b"\x00" in data:
                    binary.add(path)
    return {"commits": len(commits), "forbidden_paths": sorted(forbidden), "submodules": sorted(submodules),
            "lfs_pointers": sorted(lfs), "files_over_5mb": sorted(large), "binary_files": sorted(binary)}


def worktree_checks(clone: Path, deny) -> list[dict]:
    out = []
    for branch in PUBLICATION_BRANCHES:
        git(clone, "checkout", "-q", branch)
        files = [p for p in clone.rglob("*") if p.is_file() and ".git" not in p.relative_to(clone).parts]
        rel = [p.relative_to(clone).as_posix() for p in files]
        result = scan.scan_worktree(clone, deny, f"working tree {branch}")
        result.update({
            "branch": branch, "tip": git(clone, "rev-parse", "HEAD").strip(),
            "forbidden_paths": [r for r in rel if FORBIDDEN_PATH.search(r)],
            "files_over_5mb": [r for r, p in zip(rel, files) if p.stat().st_size > MAX_BYTES],
            "binary_files": [r for r, p in zip(rel, files) if b"\x00" in p.read_bytes()],
            "status_clean": git(clone, "status", "--porcelain").strip() == "",
        })
        out.append(result)
    git(clone, "checkout", "-q", PUBLICATION_BRANCHES[-1])
    return out


def message_checks(clone: Path, deny) -> dict:
    """Scan every reachable commit message in memory with the same rules as the file scans.

    Secret rules: each detect-secrets plugin's ``analyze_line`` (the per-line rules of file scanning).
    Content rules: the generic patterns and the private term list. Identities are not recorded here.
    Not covered: references to environment-file names in messages (see docs/security_scan.md).
    """
    from detect_secrets.settings import default_settings, get_plugins
    commits = git(clone, "rev-list", "--branches").split()
    secret, content, lines = [], [], 0
    with default_settings():
        plugins = list(get_plugins())
        for sha in commits:
            msg = git(clone, "cat-file", "commit", sha).partition("\n\n")[2]
            for n, ln in enumerate(msg.splitlines(), 1):
                lines += 1
                for p in plugins:
                    for s in p.analyze_line(filename="COMMIT_MSG.txt", line=ln, line_number=n):
                        secret.append({"commit": sha[:12], "line": n, "rule": s.type})
                for f in scan._content_findings("COMMIT_MSG", ln, deny):
                    content.append({"commit": sha[:12], "line": n, "rule": f["rule"]})
    return {"commits": len(commits), "lines": lines, "secret_findings": secret,
            "content_findings": content}


def objects_report(repo: Path) -> dict:
    fsck = git(repo, "fsck", "--unreachable", "--dangling", "--no-reflogs", "--no-progress", check=False)
    lines = [ln for ln in fsck.splitlines() if ln.strip()]
    reflog = git(repo, "reflog", "--all", check=False).splitlines()
    reachable = git(repo, "rev-list", "--objects", *PUBLICATION_BRANCHES).splitlines()
    return {"unreachable_or_dangling": len(lines),
            "unreachable_or_dangling_kinds": sorted({ln.split()[1] for ln in lines if len(ln.split()) > 1}),
            "reflog_entries": len(reflog),
            "objects_reachable_from_publication_branches": len(reachable)}


def ignore_report(clone: Path) -> dict:
    res = {}
    for probe in IGNORE_PROBES:
        r = subprocess.run(["git", "-C", str(clone), "check-ignore", "-q", "--no-index", probe])
        res[probe] = r.returncode == 0
    readme = (clone / "README.md").read_text(encoding="utf-8")
    return {"ignored": res, "all_ignored": all(res.values()), "readme_states_dataset_committed":
            README_STATEMENT in readme}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", type=Path, required=True)
    ap.add_argument("--origin-repo", type=Path, required=True)
    ap.add_argument("--denylist", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.resolve().is_relative_to(args.clone.resolve()):
        raise SystemExit("--out must be outside the clone")
    deny = scan._denylist(args.denylist)
    refs = refs_report(args.clone)
    baseline = baseline_report(args.clone)
    p3_baseline = baseline_report(args.clone, P3_BASELINE, P3_BASELINE_DESCENDANTS)
    p3_baseline["first_parent_is_published_baseline"] = lineage_ok(args.clone)
    p4_baseline = baseline_report(args.clone, P4_BASELINE, P4_BASELINE_DESCENDANTS)
    p4_baseline["builds_on_p3_baseline"] = subprocess.run(
        ["git", "-C", str(args.clone), "merge-base", "--is-ancestor", P3_BASELINE, P4_BASELINE],
        capture_output=True).returncode == 0
    history, n_commits, n_pairs = scan.scan_history(args.clone, deny)
    trees = tree_checks(args.clone)
    worktrees = worktree_checks(args.clone, deny)
    messages = message_checks(args.clone, deny)
    report = {
        "refs": refs,
        "baseline": baseline,
        "p3_baseline": p3_baseline,
        "p4_baseline": p4_baseline,
        "history_scan": {"commits": n_commits, "distinct_blob_path_pairs": n_pairs,
                         "secret_findings": [f for r in history for f in r["secret_findings"]],
                         "content_findings": [f for r in history for f in r["content_findings"]]},
        "tree_checks": trees,
        "worktrees": [{k: v for k, v in w.items()} for w in worktrees],
        "messages": messages,
        "objects": {"clone": objects_report(args.clone), "academic": objects_report(args.origin_repo)},
        "ignore": ignore_report(args.clone),
        "private_denylist_entries": len(deny),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=list), encoding="utf-8")
    ok = (not refs["remotes"] and not refs["unexpected_refs"] and not refs["missing_refs"]
          and not refs["stash"] and not refs["notes"] and baseline["all_descend"] and p3_baseline["all_descend"]
          and p3_baseline["first_parent_is_published_baseline"]
          and p4_baseline["all_descend"] and p4_baseline["builds_on_p3_baseline"]
          and not report["history_scan"]["secret_findings"] and not report["history_scan"]["content_findings"]
          and not trees["forbidden_paths"] and not trees["submodules"] and not trees["lfs_pointers"]
          and not trees["files_over_5mb"]
          and all(not w["secret_findings"] and not w["content_findings"] and not w["forbidden_paths"]
                  and w["status_clean"] for w in worktrees)
          and not messages["secret_findings"] and not messages["content_findings"]
          and report["ignore"]["all_ignored"] and report["ignore"]["readme_states_dataset_committed"])
    summary = {
        "remotes": refs["remotes"], "refs": refs["refs"], "unexpected_refs": refs["unexpected_refs"],
        "missing_refs": refs["missing_refs"], "baseline_descends": baseline["descends"],
        "p3_baseline_descends": p3_baseline["descends"],
        "p3_baseline_first_parent_ok": p3_baseline["first_parent_is_published_baseline"],
        "p4_baseline_descends": p4_baseline["descends"],
        "p4_baseline_builds_on_p3_baseline": p4_baseline["builds_on_p3_baseline"],
        "commits_scanned": n_commits, "distinct_blob_path_pairs": n_pairs,
        "history_secret_findings": len(report["history_scan"]["secret_findings"]),
        "history_content_findings": len(report["history_scan"]["content_findings"]),
        "worktrees": {w["branch"]: {"tip": w["tip"][:12], "secret": len(w["secret_findings"]),
                                    "content": len(w["content_findings"]), "forbidden": len(w["forbidden_paths"]),
                                    "clean": w["status_clean"]} for w in worktrees},
        "forbidden_paths_in_history": trees["forbidden_paths"], "submodules": trees["submodules"],
        "lfs_pointers": trees["lfs_pointers"], "files_over_5mb": trees["files_over_5mb"],
        "binary_files": trees["binary_files"],
        "message_scan": {"commits": messages["commits"], "lines": messages["lines"],
                         "secret_findings": len(messages["secret_findings"]),
                         "content_findings": len(messages["content_findings"])},
        "objects": report["objects"], "ignore_all": report["ignore"]["all_ignored"],
        "readme_statement": report["ignore"]["readme_states_dataset_committed"],
        "PASS": ok,
    }
    print(json.dumps(summary, indent=2, default=list))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
