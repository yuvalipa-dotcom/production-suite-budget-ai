"""A deterministic in-process stand-in for the Anthropic Messages API: real CrewAI orchestration, scripted
model responses. NO LLM INFERENCE TAKES PLACE AND NO REQUEST LEAVES THE PROCESS.

While ``ScriptedTransport.installed()`` is active:

* ``httpx.HTTPTransport.handle_request`` (and its async twin) is replaced, so every request the Anthropic
  SDK sends through CrewAI's native provider is answered here from a fixed per-agent script;
* every outbound socket connection except loopback (CrewAI's asyncio self-pipe on Windows) and every name
  lookup is refused and counted;
* ``ANTHROPIC_API_KEY`` holds a dummy value, so no real key is ever read or sent.

The agent answering a request is identified by the exact set of tool names CrewAI offers in it (each agent
has its own tools). A turn is a list of tool calls ``[(name, args)]`` (args may be a callable of the run
context), a final text, or a callable of the context returning the final text. Every scripted text starts
with ``SCRIPTED_MARK`` so no reader can mistake it for model output, and every exchange is logged with
``source: "scripted"``.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import socket
import threading

import httpx

SCRIPTED_MARK = "[SCRIPTED RESPONSE - not LLM inference]"
DUMMY_KEY = "offline-" + "scripted-transport-not-a-key"
SOURCE = "scripted in-process transport (no LLM inference, no network)"


class NetworkBlocked(RuntimeError):
    pass


class ScriptedTransport:
    def __init__(self, plans: dict[str, list], agent_by_tools: dict[frozenset, str], context=None,
                 model: str = "scripted"):
        self.plans, self.agent_by_tools, self.context, self.model = plans, agent_by_tools, context, model
        self.pos = {k: 0 for k in plans}
        self.requests = {k: 0 for k in plans}
        self.forced_final = {k: 0 for k in plans}
        self.exchanges: list[dict] = []
        self.network_blocked: list[str] = []
        self.current: str | None = None
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    # ------------------------------------------------------------ responses
    def _message(self, content: list[dict], stop: str) -> dict:
        return {"id": f"msg_scripted_{next(self._ids)}", "type": "message", "role": "assistant", "model": self.model,
                "content": content, "stop_reason": stop, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0}}

    def _text(self, t: str) -> dict:
        return self._message([{"type": "text", "text": f"{SCRIPTED_MARK} {t}"}], "end_turn")

    def respond(self, body: dict) -> dict:
        with self._lock:
            tools = body.get("tools")
            n = len(self.exchanges) + 1
            digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
            if not tools:                                # CrewAI's forced final answer after max_iter
                agent = self.current
                self.requests[agent] = self.requests.get(agent, 0) + 1
                self.forced_final[agent] = self.forced_final.get(agent, 0) + 1
                out, kind, calls = self._text(f"Forced final answer ({agent})."), "forced_final_text", []
            else:
                agent = self.agent_by_tools[frozenset(t["name"] for t in tools)]
                self.current = agent
                self.requests[agent] += 1
                steps, i = self.plans[agent], self.pos[agent]
                self.pos[agent] = i + 1
                step = steps[i] if i < len(steps) else f"Final answer ({agent})."
                if callable(step):
                    step = step(self.context)
                if isinstance(step, str):
                    out, kind, calls = self._text(step), "final_text", []
                else:
                    calls = [(name, arg(self.context) if callable(arg) else arg) for name, arg in step]
                    out = self._message([{"type": "tool_use", "id": f"toolu_scripted_{next(self._ids)}", "name": nm,
                                          "input": a} for nm, a in calls], "tool_use")
                    kind = "tool_use"
            self.exchanges.append({"n": n, "agent": agent, "source": "scripted", "request_sha256": digest,
                                   "request_messages": len(body.get("messages") or []),
                                   "tools_offered": sorted(t["name"] for t in tools or []), "response": kind,
                                   "tool_calls": [c[0] for c in calls],
                                   "text": out["content"][0].get("text") if kind != "tool_use" else None})
            return out

    # ------------------------------------------------------------ installation
    @contextlib.contextmanager
    def installed(self):
        transport = self
        saved_sync = httpx.HTTPTransport.handle_request
        saved_async = httpx.AsyncHTTPTransport.handle_async_request
        saved_connect, saved_create, saved_gai = socket.socket.connect, socket.create_connection, socket.getaddrinfo
        saved_key = os.environ.get("ANTHROPIC_API_KEY")

        def handle_request(_self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read() or b"{}")
            return httpx.Response(200, json=transport.respond(body), request=request)

        async def handle_async_request(_self, request: httpx.Request) -> httpx.Response:
            body = json.loads(await request.aread() or b"{}")
            return httpx.Response(200, json=transport.respond(body), request=request)

        def loopback_only(sock, address):
            if isinstance(address, tuple) and address and address[0] in ("127.0.0.1", "::1"):
                return saved_connect(sock, address)
            transport.network_blocked.append(f"connect {address!r}")
            raise NetworkBlocked("network access attempted while the scripted transport is installed")

        def refuse(*args, **kwargs):
            transport.network_blocked.append(f"lookup {args[:1]!r}")
            raise NetworkBlocked("network access attempted while the scripted transport is installed")

        httpx.HTTPTransport.handle_request = handle_request
        httpx.AsyncHTTPTransport.handle_async_request = handle_async_request
        socket.socket.connect = loopback_only
        socket.create_connection = refuse
        socket.getaddrinfo = refuse
        os.environ["ANTHROPIC_API_KEY"] = DUMMY_KEY
        try:
            yield self
        finally:
            httpx.HTTPTransport.handle_request = saved_sync
            httpx.AsyncHTTPTransport.handle_async_request = saved_async
            socket.socket.connect, socket.create_connection, socket.getaddrinfo = saved_connect, saved_create, saved_gai
            if saved_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = saved_key

    def summary(self) -> dict:
        return {"source": SOURCE, "requests": len(self.exchanges), "requests_by_agent": dict(self.requests),
                "forced_final_answers": dict(self.forced_final), "network_attempts_blocked": list(self.network_blocked),
                "scripted_mark": SCRIPTED_MARK}
