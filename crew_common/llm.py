"""The crews' LLM construction (copied from ``crew1.crew``: the same settings and the same provider path).

``make_llm`` builds CrewAI's native Anthropic LLM. In the Flow it is always used together with the
scripted transport (``scripted.ScriptedTransport``), which answers every provider request in-process:
the CrewAI orchestration is real, the model responses are scripted.
"""

from __future__ import annotations

# The same pinned model id as Crew 1 (crewai 1.15.28 lists it); with the scripted transport no model runs.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
LLM_SETTINGS = {"temperature": 0.0, "max_tokens": 8192, "timeout": 180.0, "max_retries": 2}


def make_llm(model: str = DEFAULT_MODEL, **overrides):
    from crewai import LLM
    return LLM(model=f"anthropic/{model}", **{**LLM_SETTINGS, **overrides})
