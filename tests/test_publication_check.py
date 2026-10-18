"""The publication dry-run checker accepts exactly the approved branch set, and P2 must build on the
published baseline. Uses throwaway repositories under tmp_path only."""

import importlib.util
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("publication_check", REPO / "tools" / "publication_check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

IDENTITY = {"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "test@users.noreply.github.com",
            "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "test@users.noreply.github.com"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env={**os.environ, **IDENTITY}).stdout.strip()


def _repo(tmp_path: Path, branches) -> tuple[Path, str]:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    for b in branches:
        if b != "main":
            _git(repo, "branch", b, base)
    return repo, base


def test_branch_set_is_exact():
    assert check.PUBLICATION_BRANCHES == ("main", "p1-synthetic-generator", "p1-remediation", "p2-quality-gate")
    assert check.EXPECTED_REFS == {f"refs/heads/{b}" for b in check.PUBLICATION_BRANCHES}
    assert check.BASELINE_DESCENDANTS == ("p2-quality-gate",)
    assert check.PUBLISHED_BASELINE == "59517254e108738696cd3632530b953768745788"


def test_expected_p2_branch_is_accepted(tmp_path):
    repo, _ = _repo(tmp_path, check.PUBLICATION_BRANCHES)
    refs = check.refs_report(repo)
    assert refs["unexpected_refs"] == [] and refs["missing_refs"] == []


def test_unexpected_branch_is_still_rejected(tmp_path):
    repo, base = _repo(tmp_path, check.PUBLICATION_BRANCHES)
    for extra in ("p2-quality-gate-2", "p3-crew-1", "feature"):
        _git(repo, "branch", extra, base)
    assert check.refs_report(repo)["unexpected_refs"] == ["refs/heads/feature", "refs/heads/p2-quality-gate-2",
                                                          "refs/heads/p3-crew-1"]


def test_missing_p2_branch_is_reported(tmp_path):
    repo, _ = _repo(tmp_path, check.PUBLICATION_BRANCHES[:3])
    assert check.refs_report(repo)["missing_refs"] == ["refs/heads/p2-quality-gate"]


def test_p2_must_descend_from_the_published_baseline(tmp_path):
    repo, base = _repo(tmp_path, check.PUBLICATION_BRANCHES)
    _git(repo, "checkout", "-q", "p2-quality-gate")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "p2 work")
    assert check.baseline_report(repo, baseline=base)["all_descend"]
    _git(repo, "checkout", "-q", "--orphan", "unrelated")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "unrelated root")
    _git(repo, "branch", "-f", "p2-quality-gate", "HEAD")
    assert not check.baseline_report(repo, baseline=base)["all_descend"]
    assert not check.baseline_report(repo)["all_descend"]        # the real baseline is absent here
