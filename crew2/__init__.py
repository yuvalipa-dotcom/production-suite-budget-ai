"""Crew 2: the data-scientist crew (P4). ALL DATA IS SYNTHETIC.

Importing this package applies the runtime safety settings BEFORE CrewAI can be
imported by any of its modules (the same settings as Crew 1, with Crew 2's own
app-data folder name). The deterministic modelling core lives in ``crew2.ml`` and
never calls an LLM.
"""

from __future__ import annotations

import os

import crew1  # noqa: F401  (Crew 1 applies its own settings on import; Crew 2's are applied after them)

SAFETY_ENV = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_STORAGE_DIR": "psba-crew2",
}


def apply_safety_env(environ=os.environ) -> None:
    """Force the safety settings (never defaults: an inherited value cannot re-enable telemetry)."""
    environ.update(SAFETY_ENV)


apply_safety_env()
