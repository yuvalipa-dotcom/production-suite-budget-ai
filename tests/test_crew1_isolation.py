"""Crew 1 isolation (P3 2.9.2, docs/crew1_isolation.md). Offline. ALL DATA IS SYNTHETIC.

Canary tokens are planted in copies of the clean v2 tables and in ground-truth files of a mirror
repository; the tools run against that mirror and no canary may appear in any tool output, trace
or workspace file. A deliberately misconfigured allow-list shows that the canary check detects a leak.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from crew1.crew import Recorder, isolated_process_state, minimal_env
from crew1.tools import SPECS, ToolContext, build_tools
from crew1.tools.workspace import READ_NAMES, REPO, IsolationError, Sources, Workspace, classify_name, \
    stage_workspace
from tests.crew1_helpers import no_network, scripted_runner  # noqa: F401

pytestmark = pytest.mark.usefixtures("no_network")

CANARY = "CANARY-7f3e2a91-crew1-must-never-see-this"
CLEAN_V2 = ("snapshots.csv", "productions.csv", "commitments.csv", "manual_costs.csv", "budget_revisions.csv",
            "category_plans.csv", "payments.csv", "schema.json", "manifest.json")


def make_mirror(root: Path) -> Sources:
    """A mirror repository: real raw export, dictionary and contracts; canaries in clean v2 and ground truth."""
    v2 = REPO / "data" / "synthetic" / "v2"
    shutil.copytree(v2 / "raw", root / "data" / "synthetic" / "v2" / "raw")
    for name in CLEAN_V2:
        text = (v2 / name).read_text(encoding="utf-8")
        (root / "data" / "synthetic" / "v2" / name).write_text(text + f"\n{CANARY}\n", encoding="utf-8")
    gt = root / "ground_truth" / "v2"
    gt.mkdir(parents=True)
    (gt / "defect_log.csv").write_text(f"table,record_id,column\n{CANARY},x,y\n", encoding="utf-8")
    (gt / "defect_catalogue.json").write_text(json.dumps({"canary": CANARY}), encoding="utf-8")
    (root / "docs").mkdir()
    shutil.copyfile(REPO / "docs" / "raw_data_dictionary.md", root / "docs" / "raw_data_dictionary.md")
    shutil.copytree(REPO / "contracts", root / "contracts")
    return Sources(root / "data" / "synthetic" / "v2" / "raw", root / "docs" / "raw_data_dictionary.md",
                   root / "contracts")


def attack_names(mirror: Path) -> list[str]:
    gt = mirror / "ground_truth" / "v2" / "defect_log.csv"
    snaps = mirror / "data" / "synthetic" / "v2" / "snapshots.csv"
    return ["../ground_truth/v2/defect_log.csv", "raw/../../ground_truth/v2/defect_log.csv", "..\\snapshots.csv",
            "raw.productions/../../snapshots", str(gt), str(snaps), gt.as_posix(), "/etc/passwd",
            "\\\\localhost\\c$\\x.csv", "//localhost/share/x.csv", "\\\\?\\" + str(gt), "\\\\.\\" + str(gt),
            "RAW~1\\PRODUC~1.CSV", "raw.produc~1", "raw.productions:hidden", "raw/productions.csv",
            "clean.snapshots", "clean_v2.snapshots", "ground_truth.defect_log", "raw.manifest",
            "docs.raw_data_dictionary", "", "raw.", "RAW.PRODUCTIONS"]


@pytest.fixture
def ws():
    w = stage_workspace()
    yield w
    w.cleanup()


# ---------------------------------------------------------------- the staged workspace

def test_workspace_is_outside_the_repository_and_holds_only_the_approved_inputs(ws):
    root = Path(os.path.realpath(ws.root))
    assert not root.is_relative_to(REPO.resolve())
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    assert files == sorted([*(f"raw/{n}" for n in os.listdir(REPO / "data/synthetic/v2/raw")),
                            "docs/raw_data_dictionary.md"])
    assert "~" not in str(root)                                   # the long form, never an 8.3 alias


# ---------------------------------------------------------------- name and path attacks

def test_every_traversal_absolute_unc_device_short_and_stream_attempt_is_rejected(tmp_path, ws):
    allowed = {k for k in READ_NAMES if k.startswith("raw.") and k != "raw.manifest"}
    for name in attack_names(tmp_path):
        assert classify_name(name, allowed) is not None, name
    tool = {t.name: t for ts in build_tools(ToolContext(ws, Sources.default())).values() for t in ts}["load_raw_table"]
    for name in attack_names(tmp_path):
        assert tool._run(name=name).startswith("ERROR: rejected"), name
    codes = {n: classify_name(n, allowed) for n in ("../x", "C:\\x", "/x", "\\\\srv\\s\\x", "\\\\?\\C:\\x",
                                                    "RAW~1", "raw.productions:s", "raw/productions.csv", "clean.x")}
    assert codes == {"../x": "traversal", "C:\\x": "absolute_path", "/x": "absolute_path",
                     "\\\\srv\\s\\x": "unc_or_device_path", "\\\\?\\C:\\x": "unc_or_device_path",
                     "RAW~1": "short_name", "raw.productions:s": "alternate_stream",
                     "raw/productions.csv": "path_not_name", "clean.x": "not_allowlisted"}


def test_output_names_cannot_escape_the_output_folder(ws):
    for name in ("out/../../x", "out.clean_data/../x", "C:\\x", "out.stats:ads", "raw.productions"):
        with pytest.raises(IsolationError):
            ws.write(name, b"x")
    for name in ("out/../raw/productions.csv", "raw.productions"):
        with pytest.raises(IsolationError):
            ws.resolve(name, output=True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions exist only on Windows")
def test_a_planted_junction_is_rejected(tmp_path, ws):
    import _winapi
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "productions.csv").write_text(CANARY, encoding="utf-8")
    shutil.rmtree(ws.root / "raw")
    _winapi.CreateJunction(str(outside), str(ws.root / "raw"))
    with pytest.raises(IsolationError) as e:
        ws.read_bytes("raw.productions")
    assert e.value.code == "link_rejected"
    tool = {t.name: t for ts in build_tools(ToolContext(ws, Sources.default())).values() for t in ts}["load_raw_table"]
    out = tool._run(name="raw.productions")
    assert out.startswith("ERROR: rejected (link_rejected)") and CANARY not in out
    os.rmdir(ws.root / "raw")                                     # removes the junction, not its target
    assert (outside / "productions.csv").exists()


def test_a_planted_symlink_is_rejected(tmp_path, ws):
    target = tmp_path / "canary.csv"
    target.write_text(CANARY, encoding="utf-8")
    link = ws.root / "raw" / "productions.csv"
    link.unlink()
    try:
        os.symlink(target, link)
    except OSError as e:                                          # needs Developer Mode or the symlink privilege
        pytest.skip(f"Windows cannot create a file symlink here without privilege: {e}")
    with pytest.raises(IsolationError) as err:
        ws.read_bytes("raw.productions")
    assert err.value.code == "link_rejected"


def test_the_symlink_branch_rejects_a_link_reported_by_lstat(ws, monkeypatch):
    """Supplements the test above where file symlinks cannot be created: a path whose lstat reports a
    symbolic link (or any reparse point) is rejected before it is opened."""
    import stat as st
    from crew1.tools import workspace as W
    target = os.path.normcase(str(ws.root / "raw" / "productions.csv"))
    real_lstat = os.lstat

    class Fake:
        def __init__(self, mode, attrs=0):
            self.st_mode, self.st_file_attributes = mode, attrs

    for fake in (Fake(st.S_IFLNK | 0o644), Fake(st.S_IFREG | 0o644, W.REPARSE)):
        monkeypatch.setattr(W.os, "lstat", lambda p, f=fake: f if os.path.normcase(str(p)) == target else real_lstat(p))
        with pytest.raises(IsolationError) as e:
            ws.read_bytes("raw.productions")
        assert e.value.code == "link_rejected"
    monkeypatch.undo()
    assert ws.read_bytes("raw.productions")


@pytest.mark.skipif(sys.platform != "win32", reason="8.3 short names exist only on Windows")
def test_a_real_short_name_alias_is_rejected(ws):
    import ctypes
    long_path = str(ws.root / "raw" / "budget_revisions.csv")
    buf = ctypes.create_unicode_buffer(32768)
    n = ctypes.windll.kernel32.GetShortPathNameW(long_path, buf, len(buf))
    short = buf.value if n else long_path
    for candidate in {short, os.path.basename(short), "BUDGET~1.CSV"}:
        assert classify_name(candidate, set(READ_NAMES)) is not None, candidate


def test_the_open_handle_is_checked_after_opening(tmp_path, ws):
    outside = tmp_path / "x.csv"
    outside.write_text("x", encoding="utf-8")
    with open(outside, "rb") as f:
        with pytest.raises(IsolationError):
            ws._check_handle(f, ws.root / "raw" / "productions.csv")
    with open(ws.root / "raw" / "productions.csv", "rb") as f:
        ws._check_handle(f, ws.root / "raw" / "productions.csv")


# ---------------------------------------------------------------- canaries

def _workspace_text(ws: Workspace) -> str:
    return "".join(p.read_bytes().decode("utf-8", "replace") for p in ws.root.rglob("*") if p.is_file())


def test_canaries_in_clean_v2_and_ground_truth_never_reach_any_tool_output(tmp_path, monkeypatch):
    mirror = tmp_path / "mirror"
    sources = make_mirror(mirror)
    ws = stage_workspace(sources)
    recorder = Recorder(ws.root)
    seen: list[Path] = []
    real = builtins.open

    def tracing_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            seen.append(Path(file).resolve())
        return real(file, *args, **kwargs)

    try:
        monkeypatch.setattr(builtins, "open", tracing_open)
        monkeypatch.setattr(io, "open", tracing_open)
        scripted_runner()(ws, sources, "offline", recorder)
        tool = {t.name: t for ts in build_tools(ToolContext(ws, sources), recorder).values()
                for t in ts}["load_raw_table"]
        for name in attack_names(mirror):
            assert tool._run(name=name).startswith("ERROR: rejected"), name
        monkeypatch.undo()
        outputs = [e.get("output", "") for e in recorder.events if e["event"] == "tool_call"]
        assert len(outputs) > 20 and all(CANARY not in o for o in outputs)
        assert CANARY not in recorder.jsonl() and CANARY not in _workspace_text(ws)
        assert any(p.is_relative_to((ws.root / "raw").resolve()) for p in seen)        # tracing works
        clean_v2, gt = (mirror / "data" / "synthetic" / "v2").resolve(), (mirror / "ground_truth").resolve()
        for p in seen:
            assert not p.is_relative_to(gt), p
            assert not p.is_relative_to(clean_v2) or p.is_relative_to(sources.raw_dir.resolve()), p
            assert not p.is_relative_to((REPO / "ground_truth").resolve()), p
            assert not p.is_relative_to((REPO / "data").resolve()), p
    finally:
        ws.cleanup()


def test_the_canary_check_detects_a_deliberately_misconfigured_allowlist(tmp_path):
    """Two-way proof: with the clean v2 table reachable, the same check finds the canary."""
    mirror = tmp_path / "mirror"
    sources = make_mirror(mirror)
    ws = stage_workspace(sources)
    try:
        (ws.root / "leak").mkdir()
        shutil.copyfile(mirror / "data" / "synthetic" / "v2" / "snapshots.csv", ws.root / "leak" / "snapshots.csv")
        leaky = Workspace(ws.root, read_names={**READ_NAMES, "clean.snapshots": "leak/snapshots.csv"})
        assert CANARY in leaky.read_text("clean.snapshots")
        with pytest.raises(IsolationError):
            ws.read_text("clean.snapshots")                       # the real allow-list refuses it
    finally:
        ws.cleanup()


# ---------------------------------------------------------------- configuration and environment

def test_no_tool_description_schema_or_config_reveals_a_forbidden_path():
    texts = [d + json.dumps(s.model_json_schema()) for s, d in SPECS.values()]
    texts += [(REPO / "crew1" / "config" / f).read_text(encoding="utf-8") for f in ("agents.yaml", "tasks.yaml")]
    for t in texts:
        low = t.lower().replace("\\", "/")
        assert str(REPO).lower().replace("\\", "/") not in low
        assert "ground_truth" not in low and "data/synthetic" not in low and "snapshots.csv" not in low


def test_the_crew_environment_is_minimal_and_never_reveals_the_repository(ws):
    repo = str(REPO)
    placeholder = "dummy-" + "value"                              # assembled at runtime: no key-shaped literal
    env = {"VIRTUAL_ENV": repo + "\\.venv", "PYTHONPATH": repo, "MY_REPO": repo, "PWD": repo,
           "PATH": os.pathsep.join([repo + "\\.venv\\Scripts", "C:\\Windows\\system32"]),
           "SYSTEMROOT": "C:\\Windows", "UNRELATED_VAR": "x"}
    env["ANTHROPIC_API_KEY"] = placeholder
    out = minimal_env(env, ws.root)
    assert not {"VIRTUAL_ENV", "PYTHONPATH", "MY_REPO", "PWD", "UNRELATED_VAR"} & set(out)
    assert out["PATH"] == "C:\\Windows\\system32" and out["ANTHROPIC_API_KEY"] == placeholder
    assert out["TEMP"] == out["TMP"] == str(ws.root / "tmp")
    assert all(repo.lower() not in v.lower() for v in out.values())
    cwd = os.getcwd()
    with isolated_process_state(ws.root) as active:
        assert Path(os.getcwd()).resolve() == ws.root.resolve()
        assert all(repo.lower() not in v.lower() for v in os.environ.values())
        assert os.environ.get("CREWAI_DISABLE_TELEMETRY") == "true"
        assert {k.upper() for k in os.environ} == {k.upper() for k in active}
    assert os.getcwd() == cwd and os.environ.get("CREWAI_DISABLE_TELEMETRY") == "true"
