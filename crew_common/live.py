"""Live Anthropic inference for the Flow's crews, with a sanitized record of every provider call.

While ``RecordingTransport.installed()`` is active, every request the Anthropic SDK sends through CrewAI's
native provider is forwarded unchanged to ``api.anthropic.com`` by httpx's own transport; the response is
returned unchanged to the SDK. Each exchange is recorded WITHOUT its headers or body: the UTC start and end,
the agent (identified, as in ``scripted``, by the set of tool names offered), the HTTP status, the provider's
``request-id`` header, the message id, the model that answered, the stop reason, the token usage and the names
of the tools the model asked for. The API key is read only by the Anthropic SDK; it never reaches this record.

Controls (shared by both crews through one ``LiveBudget``): a cap on provider requests (SDK retries included)
and on cumulative input and output tokens. A request over budget is refused before it leaves the process; the
SDK then raises and the crew run fails. There is no fallback to scripted responses: a live run either gets
its answers from the provider or fails.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import httpx

SOURCE = "live Anthropic Messages API (api.anthropic.com), recorded by crew_common.live"
PROVIDER_HOST = "api.anthropic.com"
# Expected use of one Flow is about 45 requests, 350k input and 30k output tokens (docs/live_run.md).
DEFAULT_BUDGET = {"max_requests": 150, "max_input_tokens": 2_000_000, "max_output_tokens": 150_000}
PRICE_CATALOGUE = ("crewai", "llms/litellm_catalog/model_prices_and_context_window.json")
_DROP_HEADERS = {"content-encoding", "content-length", "transfer-encoding"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class LiveCallRefused(RuntimeError):
    """A provider request was refused before leaving the process (budget exhausted or a non-provider host)."""


class LiveBudget:
    def __init__(self, max_requests: int, max_input_tokens: int, max_output_tokens: int):
        self.limits = {"max_requests": max_requests, "max_input_tokens": max_input_tokens,
                       "max_output_tokens": max_output_tokens}
        self.used = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.refused: list[str] = []
        self._lock = threading.Lock()

    @classmethod
    def default(cls) -> "LiveBudget":
        return cls(**DEFAULT_BUDGET)

    def acquire(self) -> None:
        with self._lock:
            over = [k for k, cap in (("requests", "max_requests"), ("input_tokens", "max_input_tokens"),
                                     ("output_tokens", "max_output_tokens")) if self.used[k] >= self.limits[cap]]
            if over:
                self.refused.append(f"{utc_now()} over budget: {over}")
                raise LiveCallRefused(f"live LLM budget exhausted ({', '.join(over)}); the request was not sent")
            self.used["requests"] += 1

    def add_usage(self, usage: dict) -> None:
        with self._lock:
            self.used["input_tokens"] += _input_tokens(usage)
            self.used["output_tokens"] += int(usage.get("output_tokens") or 0)

    def summary(self) -> dict:
        return {"limits": dict(self.limits), "used": dict(self.used), "refused": list(self.refused)}


def _input_tokens(usage: dict) -> int:
    return sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                                "cache_read_input_tokens"))


class RecordingTransport:
    """Forward to Anthropic, record a sanitized summary of each exchange. Same interface as ``ScriptedTransport``."""

    _test_upstream = None        # tests only: a callable (transport, request) -> httpx.Response replacing the network

    def __init__(self, agent_by_tools: dict[frozenset, str], budget: LiveBudget, crew: str, context=None, log=None):
        self.agent_by_tools, self.budget, self.crew, self.context, self.log = agent_by_tools, budget, crew, context, log
        self.exchanges: list[dict] = []
        self.network_blocked: list[str] = []
        self.current: str | None = None
        self._lock = threading.Lock()

    def _agent(self, body: dict) -> str:
        tools = body.get("tools")
        if tools:
            self.current = self.agent_by_tools.get(frozenset(t.get("name") for t in tools), "unknown")
        return self.current or "unknown"

    def _before(self, request: httpx.Request, raw: bytes) -> dict:
        if request.url.host != PROVIDER_HOST:
            self.network_blocked.append(f"host {request.url.host}")
            raise LiveCallRefused(f"only {PROVIDER_HOST} may be called in a live run")
        body = json.loads(raw or b"{}")
        with self._lock:
            rec = {"n": len(self.exchanges) + 1, "crew": self.crew, "agent": self._agent(body), "source": "live",
                   "started_utc": utc_now(), "model_requested": body.get("model"),
                   "max_tokens_requested": body.get("max_tokens"),
                   "request_sha256": hashlib.sha256(raw or b"").hexdigest(),
                   "request_messages": len(body.get("messages") or []),
                   "tools_offered": sorted(t.get("name") for t in body.get("tools") or [])}
            self.exchanges.append(rec)
        try:
            self.budget.acquire()
        except LiveCallRefused as e:
            rec.update(finished_utc=utc_now(), refused=str(e))
            raise
        rec["_t0"] = time.monotonic()
        return rec

    def _failed(self, rec: dict, e: BaseException) -> None:
        rec.pop("_t0", None)
        rec.update(finished_utc=utc_now(), error_type=type(e).__name__)
        if self.log:
            self.log(f"[LLM {rec['n']:>3}] {self.crew}/{rec['agent']}: no response ({type(e).__name__})")

    def _after(self, rec: dict, request: httpx.Request, response: httpx.Response, content: bytes) -> httpx.Response:
        rec["finished_utc"] = utc_now()
        rec["seconds"] = round(time.monotonic() - rec.pop("_t0"), 3)
        rec["status"] = response.status_code
        rec["provider_request_id"] = response.headers.get("request-id")
        try:
            msg = json.loads(content or b"{}")
        except ValueError:
            msg = {}
        if response.status_code == 200 and isinstance(msg, dict):
            usage = {k: v for k, v in (msg.get("usage") or {}).items() if isinstance(v, int)}
            rec.update(message_id=msg.get("id"), model=msg.get("model"), stop_reason=msg.get("stop_reason"),
                       usage=usage, tool_calls=[c.get("name") for c in msg.get("content") or []
                                                if isinstance(c, dict) and c.get("type") == "tool_use"])
            self.budget.add_usage(usage)
        else:
            rec["error_type"] = (msg.get("error") or {}).get("type") if isinstance(msg, dict) else None
        if self.log:
            u = rec.get("usage") or {}
            self.log(f"[LLM {rec['n']:>3}] {self.crew}/{rec['agent']}: HTTP {rec['status']} "
                     f"{rec.get('model') or ''} {rec.get('message_id') or rec.get('error_type') or ''} "
                     f"in={_input_tokens(u)} out={u.get('output_tokens', 0)} {rec.get('stop_reason') or ''} "
                     f"tools={rec.get('tool_calls') or []}")
        headers = [(k, v) for k, v in response.headers.raw if k.decode("latin-1").lower() not in _DROP_HEADERS]
        return httpx.Response(response.status_code, headers=headers, content=content, request=request)

    @contextlib.contextmanager
    def installed(self):
        transport = self
        saved_sync = httpx.HTTPTransport.handle_request
        saved_async = httpx.AsyncHTTPTransport.handle_async_request

        def handle_request(_self, request: httpx.Request) -> httpx.Response:
            rec = transport._before(request, request.read())
            fake = type(transport)._test_upstream
            try:
                response = fake(transport, request) if fake else saved_sync(_self, request)
                content = response.read()
                response.close()
            except Exception as e:
                transport._failed(rec, e)
                raise
            return transport._after(rec, request, response, content)

        async def handle_async_request(_self, request: httpx.Request) -> httpx.Response:
            rec = transport._before(request, await request.aread())
            fake = type(transport)._test_upstream
            try:
                response = fake(transport, request) if fake else await saved_async(_self, request)
                content = await response.aread()
                await response.aclose()
            except Exception as e:
                transport._failed(rec, e)
                raise
            return transport._after(rec, request, response, content)

        httpx.HTTPTransport.handle_request = handle_request
        httpx.AsyncHTTPTransport.handle_async_request = handle_async_request
        try:
            yield self
        finally:
            httpx.HTTPTransport.handle_request = saved_sync
            httpx.AsyncHTTPTransport.handle_async_request = saved_async

    def summary(self) -> dict:
        ok = [e for e in self.exchanges if e.get("status") == 200]
        by_agent: dict[str, int] = {}
        for e in ok:
            by_agent[e["agent"]] = by_agent.get(e["agent"], 0) + 1
        return {"source": SOURCE, "requests": len(self.exchanges), "successful_responses": len(ok),
                "successful_responses_by_agent": by_agent,
                "non_200_responses": sum(1 for e in self.exchanges if e.get("status") not in (None, 200)),
                "refused_before_sending": sum(1 for e in self.exchanges if "refused" in e),
                "usage": usage_totals(self.exchanges), "network_attempts_blocked": list(self.network_blocked)}


def usage_totals(exchanges: list[dict]) -> dict:
    out = {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0}
    for e in exchanges:
        for k in out:
            out[k] += int((e.get("usage") or {}).get(k) or 0)
    return out


def estimate_cost(model: str, usage: dict) -> dict:
    """Estimated USD from the price catalogue bundled with the installed crewai (as ``crew1.run.estimate_cost``)."""
    try:
        dist = metadata.distribution(PRICE_CATALOGUE[0])
        prices = json.loads(Path(dist.locate_file(f"{PRICE_CATALOGUE[0]}/{PRICE_CATALOGUE[1]}")).read_text("utf-8"))
        p = prices[model]
    except (OSError, KeyError, ValueError, metadata.PackageNotFoundError):
        return {"estimated_usd": None, "source": "no price entry for this model in the bundled catalogue"}
    inp = p.get("input_cost_per_token", 0)
    usd = (usage.get("input_tokens", 0) * inp
           + usage.get("cache_creation_input_tokens", 0) * p.get("cache_creation_input_token_cost", inp)
           + usage.get("cache_read_input_tokens", 0) * p.get("cache_read_input_token_cost", inp)
           + usage.get("output_tokens", 0) * p.get("output_cost_per_token", 0))
    return {"estimated_usd": round(usd, 4), "source": "crewai bundled price catalogue (LiteLLM format)",
            "input_cost_per_token": inp, "output_cost_per_token": p.get("output_cost_per_token")}
