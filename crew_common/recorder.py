"""The run trace with redaction (copied from ``crew1.crew.Recorder``; identical behaviour, proven in
``tests/test_crew_common.py``). Every string is redacted when recorded: secrets, the repository path,
the workspace path and any other absolute path."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .workspace import REPO

ABS_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s\"'<>|,;)\]]*|\\\\[^\s\"'<>|,;)\]]+|"
                      r"(?<![\w.])/(?:home|Users|tmp|var|etc|root|mnt|opt|private)/[^\s\"'<>|,;)\]]*")
SECRET = re.compile(r"sk-ant-[A-Za-z0-9_\-]+|\bsk-[A-Za-z0-9_\-]{20,}")
FIELD_LIMIT = 6000


def _path_forms(p: Path) -> list[str]:
    s = str(p)
    forms = {s, s.replace("\\", "/"), s.replace("/", "\\"), s.replace("\\", "\\\\")}
    return sorted(forms, key=len, reverse=True)


class Recorder:
    """Collects trace events; every string is redacted when recorded."""

    def __init__(self, workspace_root: Path | None = None, secret_env: str = "ANTHROPIC_API_KEY"):
        self.events: list[dict] = []
        self.t0 = time.monotonic()
        self.workspace_forms = _path_forms(workspace_root) if workspace_root else []
        self.repo_forms = _path_forms(REPO)
        self.secret_env = secret_env

    def redact(self, text: str) -> str:
        secret = os.environ.get(self.secret_env)
        if secret:
            text = text.replace(secret, "<redacted-secret>")
        text = SECRET.sub("<redacted-secret>", text)
        for form in self.repo_forms:
            text = re.sub(re.escape(form), "<redacted-path>", text, flags=re.I)
        for form in self.workspace_forms:
            text = re.sub(re.escape(form), "<workspace>", text, flags=re.I)
        return ABS_PATH.sub("<redacted-path>", text)

    def _clean(self, value):
        if isinstance(value, str):
            v = self.redact(value)
            return v if len(v) <= FIELD_LIMIT else v[:FIELD_LIMIT] + " ...[truncated]"
        if isinstance(value, dict):
            return {str(k): self._clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(v) for v in value]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return self._clean(str(value))

    def add(self, kind: str, /, **fields) -> None:
        self.events.append({"seq": len(self.events) + 1, "t": round(time.monotonic() - self.t0, 3), "event": kind,
                            **self._clean(fields)})

    def step(self, step) -> None:
        """CrewAI step callback: an agent action (thought / tool call) or a final answer."""
        fields = {k: getattr(step, k) for k in ("thought", "tool", "tool_input", "result", "output", "text")
                  if getattr(step, k, None) not in (None, "")}
        self.add("agent_step", kind=type(step).__name__, **fields)

    def task(self, output) -> None:
        self.add("task_completed", task=getattr(output, "name", None) or "", agent=getattr(output, "agent", ""),
                 output=getattr(output, "raw", ""))

    def jsonl(self) -> str:
        return "".join(json.dumps(e, sort_keys=True, ensure_ascii=False) + "\n" for e in self.events)
