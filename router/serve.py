"""The live demo server (TASKS.md T42, T43, T46; lane D).

    python scripts/serve_demo.py                        # Qwen on this Mac + Gemini, router bundle
    python scripts/serve_demo.py --offline              # Wi-Fi off: cached frontier answers only
    python scripts/serve_demo.py --fake                 # no models at all: UI and plumbing check

Endpoints
  GET  /                 the web page (router/static/index.html)
  GET  /health           models, tau*, rule, offline flag
  POST /route            {"query": ...} -> the routing decision only (no generation)
  POST /chat             {"query": ...} -> decision + full answer + cost + running totals
  GET  /chat/stream?q=   server-sent events: `route` (decision), `token` (text pieces), `done` (totals)
  GET  /stats            running totals: queries, local vs frontier, $ spent, $ saved vs always-frontier,
                         router latency p50 / p99

Money. The frontier arm is priced at `frontier_price` (USD per 1M tokens in / out; default Sonnet 4.6,
the Rev A reference) even though the demo calls Gemini's free tier, so "saved" means what the same
traffic would have cost at frontier prices. A query kept local is counted at $0 (cost model M1) and
its always-frontier counterfactual uses the local answer's length as the output-length estimate.

Everything loads once, at start-up. Model calls are serialised with a lock (one Mac, one GPU).
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from router.bundle import RouterBundle
from router.generate import Generation, GenerationCache, GenParams, build_prompt, cache_key

STATIC = Path(__file__).resolve().parent / "static"


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Totals:
    queries: int = 0
    local: int = 0
    frontier: int = 0
    spent_usd: float = 0.0
    all_frontier_usd: float = 0.0
    router_ms: list = field(default_factory=list)
    first_token_ms: list = field(default_factory=list)

    def as_dict(self) -> dict:
        r = np.array(self.router_ms) if self.router_ms else np.array([0.0])
        f = np.array(self.first_token_ms) if self.first_token_ms else np.array([0.0])
        return {"queries": self.queries, "local": self.local, "frontier": self.frontier,
                "spent_usd": self.spent_usd, "all_frontier_usd": self.all_frontier_usd,
                "saved_usd": self.all_frontier_usd - self.spent_usd,
                "saved_share": (1 - self.spent_usd / self.all_frontier_usd) if self.all_frontier_usd > 0 else 0.0,
                "router_ms_p50": float(np.percentile(r, 50)), "router_ms_p99": float(np.percentile(r, 99)),
                "first_token_ms_p50": float(np.percentile(f, 50))}


class Demo:
    """Router + two clients + cache + totals. Framework-free so it can be tested without a server."""

    def __init__(self, bundle: RouterBundle, small, large, cache: GenerationCache, *,
                 params: GenParams = GenParams(max_tokens=512), offline: bool = False,
                 frontier_price: tuple[float, float] = (3.0, 15.0)):
        self.bundle, self.small, self.large, self.cache = bundle, small, large, cache
        self.params, self.offline, self.frontier_price = params, offline, frontier_price
        self.totals = Totals()
        self.lock = threading.Lock()

    def frontier_cost(self, prompt: str, answer: str) -> float:
        return (self.frontier_price[0] * approx_tokens(prompt) + self.frontier_price[1] * approx_tokens(answer)) / 1e6

    def health(self) -> dict:
        return {"small": self.small.name, "large": self.large.name, "tau": self.bundle.tau, "rule": self.bundle.rule,
                "offline": self.offline, "bundle": {k: v for k, v in self.bundle.meta.items() if k != "cal"}}

    def route(self, query: str) -> dict:
        d = self.bundle.decide(query)
        return d.as_dict()

    def stream(self, query: str, force: str | None = None):
        """Yield ("route", dict), then ("token", str)..., then ("done", dict). Errors: ("error", str)."""
        d = self.bundle.decide(query)
        route = force if force in ("local", "frontier") else d.route
        client = self.large if route == "frontier" else self.small
        info = {**d.as_dict(), "route": route, "forced": route != d.route, "model": client.name}
        yield "route", info

        prompt = build_prompt(None, query)
        key = cache_key(client.name, prompt, self.params, 0)
        t0 = time.perf_counter()
        first = None
        pieces: list[str] = []
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                for i in range(0, len(hit.text or ""), 24):            # replay the cached answer in chunks
                    first = first or (time.perf_counter() - t0) * 1000
                    pieces.append(hit.text[i:i + 24])
                    yield "token", hit.text[i:i + 24]
            elif self.offline and route == "frontier":
                yield "error", "offline mode: this frontier answer is not cached"
                return
            else:
                try:
                    for piece in client.stream(prompt, self.params):
                        first = first or (time.perf_counter() - t0) * 1000
                        pieces.append(piece)
                        yield "token", piece
                except Exception as e:                                  # noqa: BLE001 - shown to the user
                    yield "error", f"{type(e).__name__}: {e}"
                    return
                text = "".join(pieces)
                self.cache.put(key, client.name, 0, Generation(text, approx_tokens(prompt), approx_tokens(text),
                                                               latency_ms=(time.perf_counter() - t0) * 1000))
        answer = "".join(pieces)
        cost = self.frontier_cost(prompt, answer) if route == "frontier" else 0.0
        t = self.totals
        t.queries += 1
        t.local += route == "local"
        t.frontier += route == "frontier"
        t.spent_usd += cost
        t.all_frontier_usd += self.frontier_cost(prompt, answer)
        t.router_ms.append(d.router_ms)
        if first is not None:
            t.first_token_ms.append(first)
        yield "done", {"cost_usd": cost, "total_ms": (time.perf_counter() - t0) * 1000 + d.router_ms,
                       "first_token_ms": first, "cached": hit is not None, "totals": t.as_dict()}

    def chat(self, query: str, force: str | None = None) -> dict:
        out: dict = {"answer": ""}
        for kind, payload in self.stream(query, force):
            if kind == "token":
                out["answer"] += payload
            elif kind == "error":
                out["error"] = payload
            else:
                out[kind] = payload
        return out


def create_app(demo: Demo):
    """FastAPI app around a Demo. Imported lazily so the rest of the package does not need FastAPI."""
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel

    app = FastAPI(title="Cost-aware router demo")

    class Query(BaseModel):
        query: str
        force: str | None = None

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return demo.health()

    @app.post("/route")
    def route(q: Query):
        return demo.route(q.query)

    @app.post("/chat")
    def chat(q: Query):
        return demo.chat(q.query, q.force)

    @app.get("/chat/stream")
    def chat_stream(q: str, force: str | None = None):
        def events():
            for kind, payload in demo.stream(q, force):
                yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/stats")
    def stats():
        return demo.totals.as_dict()

    return app
