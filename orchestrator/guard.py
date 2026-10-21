"""A runtime boundary guard for the Crew 2 stage of the Flow (P5-E).

Crew 1 and Crew 2 run in the same Flow process, so Crew 1's modules are already loaded when Crew 2 starts.
While ``crew2_boundary(...)`` is active, a process audit hook refuses (and records) every file open under the
raw export (``data/``), Crew 1's code (``crew1/``) and the current run's Crew 1 output folder, and every new
import of a Crew 1 module. Crew 2 can therefore reach only the Flow's handoff copy. The audit hook is
installed once (Python cannot remove one) and is inert outside the context. It covers the Flow process; the
joblib worker processes only fit models from data passed in memory (the guarded child-process tests cover
them too).
"""

from __future__ import annotations

import contextlib
import importlib.abc
import os
import sys
import threading
from pathlib import Path

_STATE = {"active": False, "forbidden": (), "violations": []}
_LOCK = threading.Lock()
_INSTALLED = False


def _norm(p) -> str:
    return os.path.realpath(p).replace("\\", "/").lower()


def _hook(event, args):
    if not _STATE["active"] or event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    p = _norm(os.fsdecode(args[0]))
    if any(p == f or p.startswith(f + "/") for f in _STATE["forbidden"]) or "/raw/" in p:
        _STATE["violations"].append("open " + p.rsplit("/", 2)[-2] + "/" + p.rsplit("/", 1)[-1])
        raise PermissionError("Crew 2 may read only the Flow's handoff copy")


class _BlockCrew1(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _STATE["active"] and name.split(".")[0] == "crew1":
            _STATE["violations"].append("import " + name)
            raise ImportError("Crew 2 may not import Crew 1")
        return None


def _install():
    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            sys.addaudithook(_hook)
            sys.meta_path.insert(0, _BlockCrew1())
            _INSTALLED = True


@contextlib.contextmanager
def crew2_boundary(repo: Path, crew1_out: Path):
    _install()
    _STATE.update(active=True, violations=[],
                  forbidden=tuple(_norm(p) for p in (repo / "data", repo / "crew1", crew1_out)))
    record = {"violations": _STATE["violations"]}
    try:
        yield record
    finally:
        _STATE["active"] = False
        record["violations"] = list(_STATE["violations"])
