"""Crew 1: the data-analyst crew (P3). ALL DATA IS SYNTHETIC.

Importing this package applies the runtime safety settings BEFORE CrewAI can be
imported by any of its modules: telemetry, tracking and tracing are switched off,
and CrewAI's local app-data folder gets a fixed name instead of one derived from
the working directory.
"""

from __future__ import annotations

import os

SAFETY_ENV = {
    "CREWAI_DISABLE_TELEMETRY": "true",
    "CREWAI_DISABLE_TRACKING": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "CREWAI_STORAGE_DIR": "psba-crew1",
}


def apply_safety_env(environ=os.environ) -> None:
    """Force the safety settings (never defaults: an inherited value cannot re-enable telemetry)."""
    environ.update(SAFETY_ENV)


apply_safety_env()
