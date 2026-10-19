"""The publication dry-run checker accepts p4-crew2 as the approved P4 candidate, only on its P3 base.
Uses throwaway repositories under tmp_path, plus read-only checks of this repository's history."""

from __future__ import annotations

import subprocess

from tests.test_publication_check import REPO, _git, _repo, check


def test_p4_candidate_is_in_the_exact_branch_set(tmp_path):
    assert check.PUBLICATION_BRANCHES[-1] == "p4-crew2" and len(check.PUBLICATION_BRANCHES) == 6
    assert check.P4_BASELINE_DESCENDANTS == ("p3-crew1", "p4-crew2")
    repo, _ = _repo(tmp_path, check.PUBLICATION_BRANCHES)
    refs = check.refs_report(repo)
    assert refs["unexpected_refs"] == [] and refs["missing_refs"] == []


def test_missing_p4_branch_is_reported(tmp_path):
    repo, _ = _repo(tmp_path, check.PUBLICATION_BRANCHES[:5])
    assert check.refs_report(repo)["missing_refs"] == ["refs/heads/p4-crew2"]


def test_p4_must_descend_from_its_p3_base(tmp_path):
    repo, base = _repo(tmp_path, check.PUBLICATION_BRANCHES)
    _git(repo, "checkout", "-q", "p4-crew2")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "p4 work")
    p4 = dict(baseline=base, branches=check.P4_BASELINE_DESCENDANTS)
    assert check.baseline_report(repo, **p4)["all_descend"]
    _git(repo, "checkout", "-q", "--orphan", "unrelated")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "unrelated root")
    _git(repo, "branch", "-f", "p4-crew2", "HEAD")
    assert not check.baseline_report(repo, **p4)["all_descend"]
    assert not check.baseline_report(repo, branches=check.P4_BASELINE_DESCENDANTS,
                                     baseline=check.P4_BASELINE)["all_descend"]     # absent here


def test_real_p4_base_matches_this_repository():
    """The P4 base is the commit p4-crew2 started from; it descends from the P3 baseline and is an ancestor of
    both p3-crew1 and p4-crew2. The full id is computed at runtime (no quoted hex literal here)."""
    def has(obj):
        return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{obj}^{{commit}}"]).returncode == 0
    if not (has(check.P4_BASELINE) and has("refs/heads/p4-crew2") and has("refs/heads/p3-crew1")):
        return                                                             # e.g. a reduced test checkout
    assert check.P4_BASELINE == _git(REPO, "rev-parse", "7d2d38a")
    assert check.baseline_report(REPO, check.P4_BASELINE, check.P4_BASELINE_DESCENDANTS)["all_descend"]
    assert check.baseline_report(REPO, check.P3_BASELINE, ("p4-crew2",))["all_descend"]
