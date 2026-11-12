"""T42/T43/T46: demo server plumbing with stand-in router and models (no MLX, no network)."""
import httpx
import numpy as np
import pytest

from router import generate as gen
from router.bundle import ConstantRouter, LengthFeaturizer, RouterBundle
from router.serve import Demo, create_app


def _demo(tau=0.1, offline=False, cache=None):
    b = RouterBundle(LengthFeaturizer(), ConstantRouter(0.7, 0.9), tau=tau)
    return Demo(b, gen.FakeClient("qwen3.5-4b"), gen.FakeClient("gemini-3-flash"),
                cache if cache is not None else gen.GenerationCache(":memory:"), offline=offline)


def test_routes_by_tau_and_counts_money():
    d = _demo(tau=0.1)                                       # delta_hat 0.2 > 0.1 -> frontier
    r = d.chat("hello")
    assert r["route"]["route"] == "frontier" and r["answer"].startswith("The answer is")
    assert r["done"]["cost_usd"] > 0
    d.bundle.tau = 0.5                                       # now local
    r = d.chat("hello again")
    assert r["route"]["route"] == "local" and r["done"]["cost_usd"] == 0
    t = d.totals.as_dict()
    assert t["queries"] == 2 and t["local"] == 1 and 0 < t["saved_share"] < 1


def test_second_ask_is_served_from_cache():
    d = _demo()
    d.chat("same question")
    assert d.chat("same question")["done"]["cached"]
    assert d.large.calls == 1


def test_offline_mode_uses_cache_and_refuses_uncached_frontier_calls():
    cache = gen.GenerationCache(":memory:")
    _demo(cache=cache).chat("rehearsed question")
    off = _demo(offline=True, cache=cache)
    assert off.chat("rehearsed question")["done"]["cached"]
    assert "offline" in off.chat("brand new question")["error"]


def test_force_overrides_the_router():
    r = _demo(tau=0.1).chat("x", force="local")
    assert r["route"]["route"] == "local" and r["route"]["forced"]


def test_bundle_round_trip(tmp_path):
    b = RouterBundle(LengthFeaturizer(), ConstantRouter(), tau=0.05, meta={"interim": True})
    loaded = RouterBundle.load(b.save(tmp_path / "r.pkl"))
    assert loaded.decide("q").route == b.decide("q").route and loaded.meta["interim"]


def test_http_endpoints():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    c = TestClient(create_app(_demo()))
    assert c.get("/").status_code == 200
    assert c.get("/health").json()["large"] == "gemini-3-flash"
    assert c.post("/route", json={"query": "x"}).json()["route"] == "frontier"
    body = c.get("/chat/stream", params={"q": "hi"}).text
    assert "event: route" in body and "event: token" in body and "event: done" in body
    assert c.get("/stats").json()["queries"] == 1


def test_openai_compat_streaming(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k")
    sse = ('data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
           'data: {"choices":[{"delta":{"content":"lo"}}]}\n\ndata: [DONE]\n\n')
    c = gen.OpenAICompatClient("api", "https://example.test/v1", "m", api_key_env="TEST_KEY",
                               transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse)))
    assert "".join(c.stream("hi", gen.GenParams())) == "Hello"


class MockCascadeRouter:
    """Mock cascade head for tests."""

    def __init__(self, escalate: bool = False):
        self.escalate = escalate

    def predict_gap(self, X):
        return np.full(len(X), 0.5 if self.escalate else -0.2)

    def predict_proba(self, X, calibrated: bool = True):
        delta = self.predict_gap(X)
        return np.clip(0.5 - delta / 2.0, 0.0, 1.0), np.clip(0.5 + delta / 2.0, 0.0, 1.0)


def _cascade_demo(escalate: bool = False, tau: float = 0.0, offline: bool = False, cache=None):
    from router.bundle import CascadeFeaturizer

    b = RouterBundle(CascadeFeaturizer(), MockCascadeRouter(escalate=escalate), tau=tau, _is_cascade=True)
    return Demo(
        b,
        gen.FakeClient("qwen3.5-4b"),
        gen.FakeClient("gemini-3-flash"),
        cache if cache is not None else gen.GenerationCache(":memory:"),
        offline=offline,
    )


def test_cascade_demo_keeps_local_when_confident():
    d = _cascade_demo(escalate=False, tau=0.0)
    r = d.chat("What is 2+2?")
    assert r["route"]["route"] == "local"
    assert r["route"]["model"] == "qwen3.5-4b"
    assert r["done"]["cost_usd"] == 0.0
    assert d.small.calls == 1
    assert d.large.calls == 0
    t = d.totals.as_dict()
    assert t["queries"] == 1 and t["local"] == 1 and t["frontier"] == 0


def test_cascade_demo_escalates_to_frontier_when_uncertain():
    d = _cascade_demo(escalate=True, tau=0.0)
    r = d.chat("Hard math question")
    assert r["route"]["route"] == "frontier"
    assert r["route"]["model"] == "gemini-3-flash"
    assert r["route"]["escalated"] is True
    assert r["done"]["cost_usd"] > 0
    # Small model ran first, then escalated to large:
    assert d.small.calls == 1
    assert d.large.calls == 1
    t = d.totals.as_dict()
    assert t["queries"] == 1 and t["local"] == 0 and t["frontier"] == 1


def test_cascade_demo_route_endpoint_generates_small_first():
    d = _cascade_demo(escalate=False, tau=0.0)
    res = d.route("Test route query")
    assert res["route"] == "local"
    assert d.small.calls == 1

