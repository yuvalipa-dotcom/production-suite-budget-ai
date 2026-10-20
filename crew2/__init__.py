"""Crew 2: the data-scientist crew (P4). ALL DATA IS SYNTHETIC.

Importing this package forces the telemetry and tracing settings BEFORE CrewAI can be
imported by any of its modules (the same values as Crew 1). Crew 2's own CrewAI app-data
folder name is set only while its crew is built and run (``crew2/crew.py``), so importing
Crew 2 never changes Crew 1's process-wide settings. The deterministic modelling core lives
in ``crew2.ml`` and never calls an LLM.

Importing this package does not import Crew 1 (P5 strict crew boundary): the file-based Flow
path (``budget_flow``) uses ``crew2.ml`` without loading any Crew 1 module. The historical P4
entry points still import Crew 1 where they need it (``crew2.crew``, ``crew2.tools``,
``crew2.validators``, and ``crew2.ml.data.build_handoff``), and Crew 1 applies its own
settings when it is imported. The forced values below are identical to Crew 1's.
"""

from __future__ import annotations

import os

SAFETY_ENV = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_STORAGE_DIR": "psba-crew2",
}


IMPORT_ENV = {k: v for k, v in SAFETY_ENV.items() if k != "CREWAI_STORAGE_DIR"}


def apply_safety_env(environ=os.environ, full: bool = False) -> None:
    """Force the safety settings (never defaults: an inherited value cannot re-enable telemetry)."""
    environ.update(SAFETY_ENV if full else IMPORT_ENV)


apply_safety_env()
