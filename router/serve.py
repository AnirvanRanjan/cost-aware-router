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
import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from router import graders
from router.bundle import Decision, RouterBundle
from router.generate import Generation, GenerationCache, GenParams, build_prompt, cache_key
from router.signals import _final_answer, mcq_prompt, option_signals, sample_signals, text_agreement_signals

STATIC = Path(__file__).resolve().parent / "static"
DEMO_SET = Path(__file__).resolve().parents[1] / "configs" / "demo_set.yaml"


def _softmax(logprobs: dict[str, float]) -> dict[str, float]:
    """Option letter -> probability (softmax over the options), for the page's option bars."""
    m = max(logprobs.values())
    e = {k: math.exp(v - m) for k, v in logprobs.items()}
    z = sum(e.values())
    return {k: round(v / z, 4) for k, v in e.items()}


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "—"
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _short(text: str, n: int = 40) -> str:
    return (text[: n - 1] + "…") if len(text) > n else (text or "—")


def load_demo_set(path: Path = DEMO_SET) -> list[dict]:
    """[{id, expect, query}] from the demo set, or [] if the file is missing."""
    if not path.exists():
        return []
    import yaml

    items = (yaml.safe_load(path.read_text()) or {}).get("queries", [])
    return [{"id": q["id"], "expect": q.get("expect", "any"), "query": q["query"]} for q in items]


def option_answer(query: str, letter: str) -> str:
    """'The answer is B) Mars.' from the option text in the question (just the letter if not found)."""
    m = re.search(rf"(?:^|\s|\(){letter}[).:]\s*(.+?)(?=\s+\(?[A-J][).:]\s|$)", query, re.S)
    return f"The answer is {letter})" + (f" {m.group(1).strip()}." if m else ".")


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
                 frontier_price: tuple[float, float] = (3.0, 15.0), prompt_style: str | None = "demo"):
        self.bundle, self.small, self.large, self.cache = bundle, small, large, cache
        self.prompt_style = prompt_style          # router.generate.PROMPT_TEMPLATES key; None = raw query
        self.params, self.offline, self.frontier_price = params, offline, frontier_price
        self.totals = Totals()
        self.lock = threading.Lock()

    def frontier_cost(self, prompt: str, answer: str) -> float:
        return (self.frontier_price[0] * approx_tokens(prompt) + self.frontier_price[1] * approx_tokens(answer)) / 1e6

    def health(self) -> dict:
        return {
            "small": self.small.name,
            "large": self.large.name,
            "tau": self.bundle.tau,
            "rule": self.bundle.rule,
            "is_cascade": getattr(self.bundle, "is_cascade", False),
            "offline": self.offline,
            "bundle": {k: v for k, v in self.bundle.meta.items() if k != "cal"},
        }

    def _small_gen(self, prompt: str, params: GenParams, idx: int = 0) -> Generation:
        """One local answer, from the cache when possible (call with self.lock held)."""
        key = cache_key(self.small.name, prompt, params, idx)
        hit = self.cache.get(key)
        if not hasattr(self, "_local_cache_flags"):
            self._local_cache_flags = []
        self._local_cache_flags.append(hit is not None)
        if hit is not None:
            return hit
        g = (self.small.generate_one(prompt, params, idx) if hasattr(self.small, "generate_one")
             else self.small.generate_batch([prompt], params, [idx])[0])
        if g.error is None and g.text is not None and not self.offline:
            self.cache.put(key, self.small.name, idx, g)
        return g

    def _uses_signals(self) -> bool:
        return getattr(self.bundle.featurizer, "name", "").startswith("cascade_t61")

    def _signals_local(self, query: str) -> tuple[Decision, str, dict]:
        """D-15: Qwen answers first and the router reads the confidence signals it was certified on.

        multiple choice  one forward pass: option probabilities, exactly as Track B scored MMLU/ARC
        arithmetic       k sampled answers with the Track B GSM8K prompt and sampling; agreement
        anything else    k sampled brief answers, agreement of their final answers as text --
                         NOT a question type the router was certified on, and labelled so
        """
        labels = graders.parse_options(query)
        if len(labels) >= 2 and hasattr(self.small, "option_logprobs"):
            scored = query if "Print only a single choice" in query else mcq_prompt(query, labels)
            olp = self.small.option_logprobs(scored, labels)
            sig, letter = option_signals(olp)
            d = self.bundle.decide(scored, signals=sig, source="mmlu")   # scored text: same token count as training
            return d, option_answer(query, letter), {
                "kind": "multiple choice", "certified": True, "signal": f"option probability {sig['p_chosen']:.2f}",
                "evidence": {"type": "options", "probs": _softmax(olp), "chosen": letter}}
        k = int(self.bundle.meta.get("samples", 1))
        sp = self.bundle.meta.get("sampling", {"temperature": 0.7, "top_p": 0.8})
        params = replace(self.params, temperature=sp["temperature"], top_p=sp["top_p"])
        arithmetic = re.search(r"\d", query) is not None
        prompt = build_prompt("gsm8k" if arithmetic else self.prompt_style, query)
        gens = [self._small_gen(prompt, params, j) for j in range(k)]
        texts, lps, n_out = [g.text for g in gens], [g.mean_logprob for g in gens], [g.n_out for g in gens]
        if arithmetic:
            sig, _, shown = sample_signals(query, texts, lps, n_out)
            finals = [_fmt_num(graders.extract_number(t or "", query)) for t in texts]
        else:
            sig, shown = text_agreement_signals(texts, lps, n_out)
            finals = [_short(_final_answer(t)) for t in texts]
        d = self.bundle.decide(query, signals=sig, source="gsm8k")
        agree = f"{round(sig['agree'] * k)}/{k} answers agree" if k > 1 else f"mean log-prob {sig['lp_mean']:.2f}"
        return d, texts[shown] or "", {"kind": "arithmetic" if arithmetic else "free-form",
                                       "certified": arithmetic, "signal": agree,
                                       "evidence": {"type": "samples", "answers": finals, "shown": shown}}

    def _answer_local_first(self, query: str) -> tuple[Decision, str, dict]:
        """Cascade bundles: Qwen's answer, then the routing decision on it (call with self.lock held)."""
        self._local_cache_flags = []                 # reset per query: which local generations were cache hits
        if self._uses_signals():
            return self._signals_local(query)
        gen_s = self._small_gen(build_prompt(self.prompt_style, query), self.params, 0)   # D-14 bundle (lane B)
        return self.bundle.decide(query, response=gen_s, mean_logprob=gen_s.mean_logprob), gen_s.text or "", {}

    def route(self, query: str) -> dict:
        if getattr(self.bundle, "is_cascade", False):
            with self.lock:
                d, _, extra = self._answer_local_first(query)
            return {**d.as_dict(), **extra}
        return self.bundle.decide(query).as_dict()

    def stream(self, query: str, force: str | None = None):
        """Yield ("route", dict), then ("token", str)..., then ("done", dict). Errors: ("error", str)."""
        prompt = build_prompt(self.prompt_style, query)
        t0 = time.perf_counter()
        first = None
        pieces: list[str] = []
        hit = None
        local_cached: bool | None = None

        if getattr(self.bundle, "is_cascade", False):
            # D-14 / D-15: the local model answers first (free on the Mac); the router reads that answer
            if force == "frontier":
                d = Decision("frontier", None, None, None, None, float(self.bundle.tau), self.bundle.rule, 0.0)
                route, client = "frontier", self.large
                yield "route", {**d.as_dict(), "route": route, "forced": True, "model": client.name}
            else:
                yield "status", {"message": f"{self.small.name} is answering on this Mac…"}
                with self.lock:
                    d, local_text, extra = self._answer_local_first(query)
                flags = getattr(self, "_local_cache_flags", [])
                local_cached = bool(flags) and all(flags)   # MCQ path: a live forward pass, never cached
                route = force if force in ("local", "frontier") else d.route
                client = self.large if route == "frontier" else self.small
                yield "route", {**d.as_dict(), **extra, "route": route, "forced": route != d.route,
                                "model": client.name, "escalated": route == "frontier"}
                if route == "local":
                    for i in range(0, len(local_text), 24):
                        first = first or (time.perf_counter() - t0) * 1000
                        pieces.append(local_text[i:i + 24])
                        yield "token", local_text[i:i + 24]

        else:
            # Query-only routing: decide on prompt text before generating
            d = self.bundle.decide(query)
            route = force if force in ("local", "frontier") else d.route
            client = self.large if route == "frontier" else self.small
            info = {**d.as_dict(), "route": route, "forced": route != d.route, "model": client.name}
            yield "route", info

        # If frontier was chosen (or query-only local route not yet streamed)
        if not (getattr(self.bundle, "is_cascade", False) and route == "local"):
            key = cache_key(client.name, prompt, self.params, 0)
            with self.lock:
                hit = self.cache.get(key)
                if hit is not None:
                    for i in range(0, len(hit.text or ""), 24):
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
                    except Exception as e:  # noqa: BLE001
                        yield "error", f"{type(e).__name__}: {e}"
                        return
                    text = "".join(pieces)
                    self.cache.put(
                        key,
                        client.name,
                        0,
                        Generation(text, approx_tokens(prompt), approx_tokens(text), latency_ms=(time.perf_counter() - t0) * 1000),
                    )

        answer = "".join(pieces)
        cost = self.frontier_cost(prompt, answer) if route == "frontier" else 0.0
        counterfactual = self.frontier_cost(prompt, answer)   # what this traffic would cost always-frontier
        t = self.totals
        t.queries += 1
        t.local += route == "local"
        t.frontier += route == "frontier"
        t.spent_usd += cost
        t.all_frontier_usd += counterfactual
        t.router_ms.append(d.router_ms)
        if first is not None:
            t.first_token_ms.append(first)
        yield "done", {
            "cost_usd": cost,
            "counterfactual_usd": counterfactual,
            "total_ms": (time.perf_counter() - t0) * 1000 + d.router_ms,
            "first_token_ms": first,
            # Did the answer the user sees come from the cache? Cascade local answers are the
            # k sampled local generations (all of them cached counts as a hit); MCQ runs a live
            # forward pass for the option probabilities and is never a cache hit.
            "cached": (bool(local_cached) if route == "local" and getattr(self.bundle, "is_cascade", False)
                       else hit is not None),
            "totals": t.as_dict(),
        }

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

    @app.get("/demo-set")
    def demo_set():
        """The planned demo questions (configs/demo_set.yaml), for the page's question picker."""
        return load_demo_set()

    return app
