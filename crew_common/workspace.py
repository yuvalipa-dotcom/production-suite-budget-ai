"""A staged run workspace with a logical-name allow-list (copied from ``crew1.tools.workspace``).

Tools never accept a path: they take a logical name, which is resolved through a fixed allow-list to a
file inside the workspace. Absolute paths, ``..`` traversal, UNC and device paths, 8.3 short names,
alternate data streams, symbolic links, junctions and other reparse points are rejected; links are
checked before opening, and the opened handle's final path is checked after opening.

The resolution and checking code is identical to Crew 1's (``tests/test_crew_common.py``); only the
staging is generic: ``stage`` copies the given files into a fresh directory outside the repository.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"[a-z]+(\.[a-z_]+){1,2}")
REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class IsolationError(Exception):
    """A rejected name or path. ``code`` is safe to show to an agent; it never contains a path."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Workspace:
    """A staged workspace. All agent-reachable file access goes through ``read_bytes`` / ``write``."""

    def __init__(self, root: Path, read_names: dict, output_names: dict):
        if _is_link(root):
            raise IsolationError("workspace_is_link")
        self.root = Path(os.path.realpath(root))
        self.read_names = dict(read_names)
        self.output_names = dict(output_names)

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

    def _read(self, path: Path) -> bytes:
        try:
            with open(path, "rb") as f:
                self._check_handle(f, path)
                return f.read()
        except FileNotFoundError:
            raise IsolationError("not_found") from None

    def read_bytes(self, name: str) -> bytes:
        return self._read(self.resolve(name))

    def read_output(self, name: str) -> bytes:
        return self._read(self.resolve(name, output=True))

    def has_output(self, name: str) -> bool:
        return self.resolve(name, output=True).is_file()

    def write(self, name: str, data: bytes | str) -> None:
        path = self.resolve(name, output=True)
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(path, "wb") as f:
            self._check_handle(f, path)
            f.write(data)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def stage(prefix: str, inputs: dict[str, Path], read_names: dict, output_names: dict,
          subdirs=("in", "out", "tmp"), parent: Path | None = None) -> Workspace:
    """A fresh workspace OUTSIDE the repository holding copies of ``inputs`` (``{relative path: source}``)."""
    root = Path(os.path.realpath(tempfile.mkdtemp(prefix=prefix, dir=parent)))
    if _norm(root).startswith(_norm(REPO) + os.sep) or _norm(root) == _norm(REPO):
        shutil.rmtree(root, ignore_errors=True)
        raise IsolationError("workspace_inside_repository")
    for sub in subdirs:
        (root / sub).mkdir(parents=True, exist_ok=True)
    for rel, src in inputs.items():
        if _is_link(Path(src)):
            shutil.rmtree(root, ignore_errors=True)
            raise IsolationError("link_rejected")
        shutil.copyfile(src, root / rel)
    return Workspace(root, read_names, output_names)
