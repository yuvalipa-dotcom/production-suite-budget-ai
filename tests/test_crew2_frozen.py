"""P4 must leave the frozen data, contracts, P2 gate, generator and every published P3 file byte-identical
to the published main it was merged with (P3 via PR #4). Offline. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import subprocess

import pytest

from tests.conftest import REPO

P3_BASE = "7d2d38a"            # the P4 branch point
PUBLISHED_MAIN = "bb7d26c"     # published main after P3 (PR #4), merged into p4-crew2
FROZEN = ("data", "contracts", "gate", "generator", "crew1", "tools", "LICENSE", "requirements.txt",
          "requirements-crew.txt", "docs/raw_cleaning_policy.md", "docs/raw_data_dictionary.md",
          "docs/feature_availability.md", "docs/split_protocol.md", "docs/quality_gate.md", "docs/crew1.md",
          "artifacts", "scripts", ".gitattributes")
DATA_AND_CONTRACTS = ("data", "contracts", "gate", "generator")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


# Owner-approved changes under frozen paths, and nothing else: the narrow scanner exemption for the P4
# clean-data digest (P4 security closure) and the p4-crew2 publication candidate (P4 readiness fixes).
APPROVED_CHANGES = {"tools/publish_scan.py", "tools/publication_check.py"}
APPROVED_P3_TEST_CHANGES = {"tests/test_publication_check.py"}


def _require(*commits):
    try:
        for c in commits:
            _git("cat-file", "-e", c)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history unavailable")


def test_frozen_paths_and_p3_files_are_byte_identical_to_the_published_main():
    _require(PUBLISHED_MAIN)
    changed = sorted(set(_git("diff", "--name-only", PUBLISHED_MAIN, "--", *FROZEN).split()) - APPROVED_CHANGES)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN).split()
    assert changed == [] and untracked == []
    p3_tests = [p for p in _git("ls-tree", "-r", "--name-only", PUBLISHED_MAIN, "tests").split()]
    assert sorted(set(_git("diff", "--name-only", PUBLISHED_MAIN, "--", *p3_tests).split())
                  - APPROVED_P3_TEST_CHANGES) == []


def test_data_contracts_gate_and_generator_are_unchanged_since_the_p4_branch_point():
    _require(P3_BASE)
    assert _git("diff", "--name-only", P3_BASE, "--", *DATA_AND_CONTRACTS).split() == []
