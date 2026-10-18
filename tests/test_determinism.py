"""One seed reproduces byte-identical outputs."""

import hashlib

from generator import config as C
from generator.cli import FILES, generate, write
from tests.conftest import COMMITTED_DATA

ALL_FILES = [f"{n}.csv" for n in FILES] + ["schema.json", "manifest.json"]


def _hashes(directory):
    return {f: hashlib.sha256((directory / f).read_bytes()).hexdigest() for f in ALL_FILES}


def test_two_runs_are_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(generate(C.SEED), a, C.SEED)
    write(generate(C.SEED), b, C.SEED)
    assert _hashes(a) == _hashes(b)


def test_different_seed_changes_output(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(generate(C.SEED), a, C.SEED)
    write(generate(C.SEED + 1), b, C.SEED + 1)
    assert _hashes(a)["snapshots.csv"] != _hashes(b)["snapshots.csv"]


def test_committed_dataset_matches_generator(generated):
    """The dataset in the repository is exactly what the generator produces for the seed."""
    assert _hashes(COMMITTED_DATA) == _hashes(generated["dir"])


def test_manifest_hashes_match_files(generated):
    for name, digest in generated["manifest"]["files_sha256"].items():
        assert hashlib.sha256((generated["dir"] / name).read_bytes()).hexdigest() == digest


def test_output_uses_lf_line_endings(generated):
    for f in ALL_FILES:
        assert b"\r\n" not in (generated["dir"] / f).read_bytes()
