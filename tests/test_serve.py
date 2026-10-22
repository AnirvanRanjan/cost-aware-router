"""T42/T43/T46: demo server plumbing with stand-in router and models (no MLX, no network)."""
import httpx
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
