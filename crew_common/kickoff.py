"""Evidence that a crew really ran through ``crew.kickoff()``: CrewAI's own event bus. ALL DATA IS SYNTHETIC.

``KickoffEvidence`` subscribes (scoped to one kickoff) to the events CrewAI emits only from inside a real
kickoff: crew kickoff started / completed, task started / completed, agent execution started / completed,
tool usage finished and LLM call started. The record names the agents and tasks that executed, in order.
"""

from __future__ import annotations

import contextlib
from importlib import metadata


def _role(obj) -> str:
    agent = getattr(obj, "agent", None)
    return (getattr(agent, "role", None) or getattr(obj, "agent_role", None) or "").strip()


class KickoffEvidence:
    def __init__(self):
        self.events: list[dict] = []

    def _add(self, kind: str, **fields):
        self.events.append({"event": kind, **{k: v for k, v in fields.items() if v not in (None, "")}})

    @contextlib.contextmanager
    def listening(self):
        from crewai.events import crewai_event_bus as bus
        from crewai.events.types.agent_events import AgentExecutionCompletedEvent, AgentExecutionStartedEvent
        from crewai.events.types.crew_events import CrewKickoffCompletedEvent, CrewKickoffStartedEvent
        from crewai.events.types.llm_events import LLMCallStartedEvent
        from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
        from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

        with bus.scoped_handlers():
            @bus.on(CrewKickoffStartedEvent)
            def _ks(source, event):
                self._add("crew_kickoff_started", crew=getattr(event, "crew_name", None))

            @bus.on(CrewKickoffCompletedEvent)
            def _kc(source, event):
                self._add("crew_kickoff_completed", crew=getattr(event, "crew_name", None))

            @bus.on(TaskStartedEvent)
            def _ts(source, event):
                task = getattr(event, "task", None)
                self._add("task_started", task=getattr(task, "name", None), agent=_role(task))

            @bus.on(TaskCompletedEvent)
            def _tc(source, event):
                task = getattr(event, "task", None)
                self._add("task_completed", task=getattr(task, "name", None), agent=_role(task))

            @bus.on(AgentExecutionStartedEvent)
            def _as(source, event):
                self._add("agent_execution_started", agent=_role(event))

            @bus.on(AgentExecutionCompletedEvent)
            def _ac(source, event):
                self._add("agent_execution_completed", agent=_role(event))

            @bus.on(ToolUsageFinishedEvent)
            def _tu(source, event):
                self._add("tool_usage_finished", tool=getattr(event, "tool_name", None),
                          agent=(getattr(event, "agent_role", None) or "").strip())

            @bus.on(LLMCallStartedEvent)
            def _lc(source, event):
                self._add("llm_call_started")

            try:
                yield self
            finally:
                bus.flush()

    def summary(self) -> dict:
        def names(kind, key):
            out = []
            for e in self.events:
                if e["event"] == kind and e.get(key) and e[key] not in out:
                    out.append(e[key])
            return out
        counts = {}
        for e in self.events:
            counts[e["event"]] = counts.get(e["event"], 0) + 1
        tools = {}
        for e in self.events:
            if e["event"] == "tool_usage_finished":
                tools.setdefault(e.get("agent", ""), []).append(e.get("tool"))
        return {"crewai_version": metadata.version("crewai"), "event_counts": counts,
                "agents_executed": names("agent_execution_started", "agent"),
                "tasks_completed": names("task_completed", "task"), "tools_by_agent": tools}
