"""P4 must leave the frozen data, contracts, P2 gate, generator and every P3 file byte-identical to the
P3 base commit. Offline. ALL DATA IS SYNTHETIC."""

from __future__ import annotations

import subprocess

import pytest

from tests.conftest import REPO

P3_BASE = "7d2d38a"
FROZEN = ("data", "contracts", "gate", "generator", "crew1", "tools", "LICENSE", "requirements.txt",
          "requirements-crew.txt", "docs/raw_cleaning_policy.md", "docs/raw_data_dictionary.md",
          "docs/feature_availability.md", "docs/split_protocol.md", "docs/quality_gate.md", "docs/crew1.md")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def test_frozen_paths_and_p3_files_are_byte_identical_to_the_p3_base():
    try:
        _git("cat-file", "-e", P3_BASE)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history unavailable")
    changed = _git("diff", "--name-only", P3_BASE, "--", *FROZEN).split()
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *FROZEN).split()
    assert changed == [] and untracked == []
    p3_tests = [p for p in _git("ls-tree", "-r", "--name-only", P3_BASE, "tests").split()]
    assert _git("diff", "--name-only", P3_BASE, "--", *p3_tests).split() == []
