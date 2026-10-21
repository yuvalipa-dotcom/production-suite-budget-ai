"""P5-G: the publication dry-run checker takes the new candidate branch as a parameter (--candidate-branch),
never as a hard-coded name. The fixed branch set is unchanged; only one validated phase-branch name is added
for a run, and that branch must build on the published main after P4. Throwaway repositories only."""

import importlib.util
import subprocess
import sys

import pytest

from tests.test_publication_check import REPO, _git, _repo

_spec = importlib.util.spec_from_file_location("publication_check", REPO / "tools" / "publication_check.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def test_the_fixed_branch_set_is_unchanged_and_names_no_p5_branch():
    assert check.publication_branches() == check.PUBLICATION_BRANCHES
    assert not any(b.startswith("p5") for b in check.PUBLICATION_BRANCHES)
    assert "p5-flow" not in (REPO / "tools" / "publication_check.py").read_text(encoding="utf-8")


def test_a_candidate_branch_is_added_only_for_the_run():
    assert check.publication_branches("p5-flow") == check.PUBLICATION_BRANCHES + ("p5-flow",)
    assert check.publication_branches("p12-final-polish")[-1] == "p12-final-polish"


@pytest.mark.parametrize("bad", ["", "main", "p4-crew2", "p3-crew1", "p1-anything", "feature", "p5", "p5-",
                                 "p5-Flow", "p5-flow/x", "refs/heads/p5-flow", "p5-flow ", "p5--flow", "p05-flow"])
def test_other_names_are_rejected(bad):
    with pytest.raises(ValueError):
        check.publication_branches(bad)


def test_refs_report_accepts_the_candidate_and_still_rejects_strangers(tmp_path):
    branches = check.publication_branches("p5-flow")
    repo, base = _repo(tmp_path, branches)
    assert check.refs_report(repo)["unexpected_refs"] == ["refs/heads/p5-flow"]      # not accepted by default
    refs = check.refs_report(repo, branches)
    assert refs["unexpected_refs"] == [] and refs["missing_refs"] == []
    _git(repo, "branch", "p5-flow-2", base)
    assert check.refs_report(repo, branches)["unexpected_refs"] == ["refs/heads/p5-flow-2"]
    _git(repo, "branch", "-D", "p5-flow")
    assert check.refs_report(repo, branches)["missing_refs"] == ["refs/heads/p5-flow"]


def test_the_candidate_must_build_on_the_published_p4_main():
    def has(obj):
        return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{obj}^{{commit}}"]).returncode == 0
    if not has(check.CANDIDATE_BASELINE):
        pytest.skip("git history unavailable")
    assert subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short=7", check.CANDIDATE_BASELINE],
                          capture_output=True, text=True).stdout.strip() == "ea7366f"
    if has("refs/heads/p5-flow"):
        assert check.baseline_report(REPO, check.CANDIDATE_BASELINE, ("p5-flow",))["all_descend"]
    for b in ("p4-crew2", "p3-crew1"):
        if has(f"refs/heads/{b}"):     # earlier phase tips predate the P4 merge: they do not descend from it
            assert not check.baseline_report(REPO, check.CANDIDATE_BASELINE, (b,))["all_descend"]


def test_the_command_line_rejects_a_bad_candidate_before_reading_anything(tmp_path):
    res = subprocess.run([sys.executable, str(REPO / "tools" / "publication_check.py"), "--clone", str(tmp_path),
                          "--origin-repo", str(tmp_path), "--denylist", str(tmp_path / "none"),
                          "--out", str(tmp_path.parent / "out.json"), "--candidate-branch", "main"],
                         capture_output=True, text=True)
    assert res.returncode != 0 and "not an acceptable candidate branch" in res.stderr
