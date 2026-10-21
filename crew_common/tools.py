"""A deterministic CrewAI tool bound to one run (the Crew 1 / Crew 2 tool pattern, crew-neutral).

The tool runs Python only; every call is recorded in the run trace with ``computed_by: python``. Errors
are returned to the agent as text, except that an error of a ``fatal`` type also stops the run: it is
stored on the shared ``RunState`` and every later tool call is refused, so the orchestrator fails the run
after kickoff whatever the (scripted) agent does next.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from .workspace import IsolationError

OUTPUT_LIMIT = 12000


def compact(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class NoArgs(BaseModel):
    pass


class RunState:
    """Shared by the tools of one run: the lock (one tool call at a time) and the first fatal error."""

    def __init__(self, fatal: tuple[type[BaseException], ...] = ()):
        self.lock = threading.Lock()
        self.fatal_types = fatal
        self.fatal: str | None = None


class RecordedTool(BaseTool):
    _fn: Callable[..., Any] = PrivateAttr()
    _recorder: Any = PrivateAttr(default=None)
    _state: Any = PrivateAttr(default=None)
    _agent: str = PrivateAttr(default="")

    def bind(self, fn, recorder, state: RunState, agent: str) -> "RecordedTool":
        self._fn, self._recorder, self._state, self._agent = fn, recorder, state, agent
        return self

    def _run(self, **kwargs) -> str:
        with self._state.lock:
            start = time.monotonic()
            if self._state.fatal:
                text, ok = f"ERROR: the run already failed ({self._state.fatal}); no further step runs.", False
            else:
                try:
                    out = self._fn(**kwargs)
                    text, ok = (out if isinstance(out, str) else compact(out)), True
                except IsolationError as e:
                    text, ok = f"ERROR: rejected ({e.code}).", False
                except Exception as e:                       # noqa: BLE001  (returned to the agent)
                    text, ok = f"ERROR: {type(e).__name__}: {e}", False
                    if isinstance(e, self._state.fatal_types):
                        self._state.fatal = f"{type(e).__name__} in {self.name}: {e}"[:500]
            if len(text) > OUTPUT_LIMIT:
                text = text[:OUTPUT_LIMIT] + " ...[truncated]"
            if self._recorder is not None:
                self._recorder.add("tool_call", tool=self.name, agent=self._agent, args=kwargs, ok=ok, output=text,
                                   computed_by="python", seconds=round(time.monotonic() - start, 3))
            return text


def build_tools(specs: dict, toolsets: dict, ctx, recorder, state: RunState) -> dict[str, list[BaseTool]]:
    """``specs[name] = (args schema, description)``; each agent gets only the tools of its own toolset."""
    out = {}
    for agent, names in toolsets.items():
        out[agent] = [RecordedTool(name=n, description=specs[n][1], args_schema=specs[n][0])
                      .bind(getattr(ctx, n), recorder, state, agent) for n in names]
    return out
