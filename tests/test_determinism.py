"""One seed reproduces byte-identical outputs, for every version and for the raw export."""

import hashlib

from generator import config as C
from generator.cli import FILES_V1, FILES_V2, RAW_FILES, generate, write, write_raw
from tests.conftest import ACTIVE_DATA, DATA

V1_FILES = [f"{n}.csv" for n in FILES_V1] + ["schema.json", "manifest.json"]
V2_FILES = [f"{n}.csv" for n in FILES_V2] + ["schema.json", "manifest.json"]
RAW = [f"{n}.csv" for n in RAW_FILES] + ["manifest.json"]


def _hashes(directory, files):
    return {f: hashlib.sha256((directory / f).read_bytes()).hexdigest() for f in files}


def test_two_runs_are_byte_identical(tmp_path):
    for version, files in (("v1", V1_FILES), ("v2", V2_FILES)):
        a, b = tmp_path / f"a_{version}", tmp_path / f"b_{version}"
        write(generate(C.SEED), a, C.SEED, version)
        write(generate(C.SEED), b, C.SEED, version)
        assert _hashes(a, files) == _hashes(b, files), version


def test_raw_export_is_byte_identical(tmp_path):
    write_raw(generate(C.SEED), tmp_path / "a", C.SEED)
    write_raw(generate(C.SEED), tmp_path / "b", C.SEED)
    assert _hashes(tmp_path / "a", RAW) == _hashes(tmp_path / "b", RAW)


def test_different_seed_changes_output(tmp_path):
    write(generate(C.SEED), tmp_path / "a", C.SEED, "v2")
    write(generate(C.SEED + 1), tmp_path / "b", C.SEED + 1, "v2")
    assert _hashes(tmp_path / "a", V2_FILES)["snapshots.csv"] != _hashes(tmp_path / "b", V2_FILES)["snapshots.csv"]


def test_committed_v1_is_unchanged(tmp_path):
    """v1 stays byte-identical for traceability."""
    write(generate(C.SEED), tmp_path / "v1", C.SEED, "v1")
    assert _hashes(DATA / "v1", V1_FILES) == _hashes(tmp_path / "v1", V1_FILES)


def test_committed_active_dataset_matches_generator(generated):
    assert _hashes(ACTIVE_DATA, V2_FILES) == _hashes(generated["dir"], V2_FILES)
    assert _hashes(ACTIVE_DATA / "raw", RAW) == _hashes(generated["dir"] / "raw", RAW)


def test_manifest_hashes_match_files(generated):
    for name, digest in generated["manifest"]["files_sha256"].items():
        assert hashlib.sha256((generated["dir"] / name).read_bytes()).hexdigest() == digest
    for name, digest in generated["raw_manifest"]["files_sha256"].items():
        assert hashlib.sha256((generated["dir"] / "raw" / name).read_bytes()).hexdigest() == digest


def test_output_uses_lf_line_endings(generated):
    for f in V2_FILES:
        assert b"\r\n" not in (generated["dir"] / f).read_bytes()
    for f in RAW:
        assert b"\r\n" not in (generated["dir"] / "raw" / f).read_bytes()
