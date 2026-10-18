"""The staged Crew-1 workspace and its logical-name allow-list (docs/crew1_isolation.md).

Every run gets a fresh directory OUTSIDE the repository holding only a copy of the
raw export and docs/raw_data_dictionary.md. Tools never accept a path: they take a
logical dataset name, which is resolved through a fixed allow-list to a file inside
the workspace. Absolute paths, ``..`` traversal, UNC and device paths, 8.3 short
names, alternate data streams, symbolic links, junctions and other reparse points
are rejected; links are checked before opening, and the opened handle's final path
is checked after opening.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW_TABLES = ("productions", "budget_revisions", "category_plans", "commitments", "manual_costs", "payments",
              "snapshot_schedule")
CHART_IDS = ("prevalence_by_stage", "prevalence_by_archetype", "split_prevalence", "gap_histogram",
             "gap_by_stage", "revision_effects", "fee_breakdown", "feature_correlations")

READ_NAMES = {**{f"raw.{t}": f"raw/{t}.csv" for t in RAW_TABLES},
              "raw.manifest": "raw/manifest.json",
              "docs.raw_data_dictionary": "docs/raw_data_dictionary.md"}
OUTPUT_NAMES = {"out.clean_data": "out/clean_data.csv", "out.cleaning_log": "out/cleaning_log.json",
                "out.stats": "out/stats.json", "out.eda_report": "out/eda_report.html",
                "out.dataset_contract": "out/dataset_contract.json", "out.insights": "out/insights.md",
                "out.gate_result": "out/gate_result.json",
                **{f"out.chart.{c}": f"out/charts/{c}.png" for c in CHART_IDS}}
NAME_RE = re.compile(r"[a-z]+(\.[a-z_]+){1,2}")
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class IsolationError(Exception):
    """A rejected name or path. ``code`` is safe to show to an agent; it never contains a path."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Sources:
    """Where the orchestrator copies the workspace inputs from. Never given to an agent."""
    raw_dir: Path
    dictionary: Path
    contracts_dir: Path

    @classmethod
    def default(cls) -> "Sources":
        return cls(REPO / "data" / "synthetic" / "v2" / "raw", REPO / "docs" / "raw_data_dictionary.md",
                   REPO / "contracts")


def classify_name(name, allowed) -> str | None:
    """The rejection code for a requested name, or None when it is an allow-listed logical name."""
    if not isinstance(name, str) or not name.strip():
        return "invalid_name"
    s = name.strip()
    if s.startswith(("\\\\", "//")) or s.lower().startswith(("\\\\?\\", "\\\\.\\", "//?/", "//./")):
        return "unc_or_device_path"
    if re.match(r"^[A-Za-z]:", s) or s.startswith(("/", "\\")):
        return "absolute_path"
    if ".." in s:
        return "traversal"
    if "~" in s:
        return "short_name"
    if ":" in s:
        return "alternate_stream"
    if "/" in s or "\\" in s:
        return "path_not_name"
    if not NAME_RE.fullmatch(s):
        return "invalid_name"
    if s not in allowed:
        return "not_allowlisted"
    return None


def _is_link(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    if getattr(st, "st_file_attributes", 0) & REPARSE:        # junctions and other reparse points
        return True
    is_junction = getattr(os.path, "isjunction", None)
    return bool(is_junction and is_junction(p))


def _final_path(fileobj) -> str:
    """The final path of an open file handle (after every link and alias is resolved)."""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes
        handle = msvcrt.get_osfhandle(fileobj.fileno())
        fn = ctypes.windll.kernel32.GetFinalPathNameByHandleW
        fn.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(32768)
        n = fn(handle, buf, len(buf), 0)
        if n == 0 or n >= len(buf):
            raise IsolationError("final_path_unavailable")
        path = buf.value
        if path.startswith("\\\\?\\UNC\\"):
            return "\\\\" + path[8:]
        return path[4:] if path.startswith("\\\\?\\") else path
    try:
        return os.readlink(f"/proc/self/fd/{fileobj.fileno()}")
    except OSError:
        return os.path.realpath(fileobj.name)


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


class Workspace:
    """A staged workspace. All agent-reachable file access goes through ``open_read`` / ``write``."""

    def __init__(self, root: Path, read_names=None, output_names=None):
        if _is_link(root):
            raise IsolationError("workspace_is_link")
        self.root = Path(os.path.realpath(root))
        self.read_names = dict(READ_NAMES if read_names is None else read_names)
        self.output_names = dict(OUTPUT_NAMES if output_names is None else output_names)

    # ---------------------------------------------------------------- resolution
    def resolve(self, name: str, *, output: bool = False) -> Path:
        table = self.output_names if output else self.read_names
        code = classify_name(name, table)
        if code:
            raise IsolationError(code)
        parts = table[name.strip()].split("/")
        cur = self.root
        for part in parts:                       # every component is checked BEFORE opening
            cur = cur / part
            if _is_link(cur):
                raise IsolationError("link_rejected")
        expected = _norm(cur)
        real = _norm(os.path.realpath(cur))
        if real != expected or not real.startswith(_norm(self.root) + os.sep):
            raise IsolationError("outside_workspace")
        return cur

    def _check_handle(self, f, path: Path) -> None:
        final = _norm(_final_path(f))
        if final != _norm(path) or not final.startswith(_norm(self.root) + os.sep):
            raise IsolationError("outside_workspace")

    # ---------------------------------------------------------------- access
    def read_bytes(self, name: str) -> bytes:
        path = self.resolve(name)
        try:
            with open(path, "rb") as f:
                self._check_handle(f, path)
                return f.read()
        except FileNotFoundError:
            raise IsolationError("not_found") from None

    def read_text(self, name: str) -> str:
        return self.read_bytes(name).decode("utf-8")

    def read_csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read_bytes(name)), dtype=str, keep_default_na=False)

    def read_output(self, name: str) -> bytes:
        path = self.resolve(name, output=True)
        try:
            with open(path, "rb") as f:
                self._check_handle(f, path)
                return f.read()
        except FileNotFoundError:
            raise IsolationError("not_found") from None

    def has_output(self, name: str) -> bool:
        return self.resolve(name, output=True).is_file()

    def write(self, name: str, data: bytes | str) -> None:
        path = self.resolve(name, output=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(path, "wb") as f:
            self._check_handle(f, path)
            f.write(data)

    def raw_dir(self) -> Path:
        """The raw folder for the in-process gate call (never shown to an agent)."""
        path = self.resolve("raw.manifest").parent
        if _is_link(path):
            raise IsolationError("link_rejected")
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_workspace(sources: Sources | None = None, parent: Path | None = None) -> Workspace:
    """Create a fresh workspace outside the repository and copy in only the approved inputs."""
    sources = sources or Sources.default()
    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="crew1_ws_", dir=parent)))
    if _norm(root).startswith(_norm(REPO) + os.sep) or _norm(root) == _norm(REPO):
        shutil.rmtree(root, ignore_errors=True)
        raise IsolationError("workspace_inside_repository")
    for sub in ("raw", "docs", "out", "out/charts", "tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    manifest = json.loads((sources.raw_dir / "manifest.json").read_text(encoding="utf-8"))
    shutil.copyfile(sources.raw_dir / "manifest.json", root / "raw" / "manifest.json")
    for t in RAW_TABLES:
        src = sources.raw_dir / f"{t}.csv"
        shutil.copyfile(src, root / "raw" / f"{t}.csv")
        if sha256_bytes((root / "raw" / f"{t}.csv").read_bytes()) != manifest["files_sha256"][f"{t}.csv"]:
            shutil.rmtree(root, ignore_errors=True)
            raise IsolationError("raw_copy_hash_mismatch")
    shutil.copyfile(sources.dictionary, root / "docs" / "raw_data_dictionary.md")
    return Workspace(root)
