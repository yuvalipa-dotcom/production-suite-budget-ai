"""Shared crew infrastructure for the Flow (P5-D). ALL DATA IS SYNTHETIC.

A neutral package: it imports neither Crew 1 nor Crew 2, so the Flow's Crew 2 can use a workspace, a
redacting trace recorder, the LLM construction and the scripted transport without loading any Crew 1
module (P5 strict crew boundary).

The approved P3 files under ``crew1/`` are frozen (byte-identical to the published main), so the shared
pieces are extracted as copies, not moved: ``workspace.Workspace`` and ``recorder.Recorder`` reproduce
``crew1.tools.workspace.Workspace`` and ``crew1.crew.Recorder``, and ``llm.make_llm`` /
``llm.LLM_SETTINGS`` reproduce ``crew1.crew``. ``tests/test_crew_common.py`` proves the copies behave
identically to the originals; the historical P3/P4 entry points keep using the originals.

Importing this package forces the same telemetry and tracing settings as Crew 1 and Crew 2, before CrewAI
can be imported by any of its modules.
"""

from __future__ import annotations

import os

SAFETY_ENV = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
}


def apply_safety_env(environ=os.environ) -> None:
    """Force the safety settings (never defaults: an inherited value cannot re-enable telemetry)."""
    environ.update(SAFETY_ENV)


apply_safety_env()
