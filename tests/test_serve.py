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



# ---- D-15: the T61 signals bundle (option probability for MCQ, k answers for arithmetic) ----------

class SignalRouter:
    """Escalates unless Qwen is sure: all k answers agree, or the chosen option has p >= 0.9."""

    def predict_gap(self, X):
        X = np.asarray(X, dtype=float)
        agree, p = X[:, 0], X[:, 5]
        sure = np.where(np.isnan(p), agree >= 1.0, p >= 0.9)
        return np.where(sure, -0.2, 0.5)


class AgreeingClient(gen.FakeClient):
    """Every sample gives the same answer."""

    def generate_batch(self, prompts, params, sample_idx):
        self.calls += len(prompts)
        return [gen.Generation("3 x 4 = 12. The answer is 12.", 10, 10, mean_logprob=-0.05) for _ in prompts]


def _t61_demo(small=None, samples=3):
    from router.bundle import T61Featurizer

    b = RouterBundle(T61Featurizer(samples), SignalRouter(), tau=0.0, _is_cascade=True,
                     meta={"samples": samples, "sampling": {"temperature": 0.7, "top_p": 0.8}})
    return Demo(b, small or gen.FakeClient("qwen3.5-4b"), gen.FakeClient("gemini-3-flash"), gen.GenerationCache(":memory:"))


def test_t61_multiple_choice_scores_options_once_and_stays_local_when_sure():
    d = _t61_demo()
    r = d.chat("Which planet is known as the Red Planet? A) Venus B) Mars C) Jupiter D) Saturn")
    assert r["route"]["route"] == "local" and r["route"]["kind"] == "multiple choice" and r["route"]["certified"]
    assert r["answer"].startswith("The answer is ")
    assert d.small.calls == 1 and d.large.calls == 0          # one forward pass, no generation
    assert r["status"]["message"].endswith("on this Mac…")


def test_t61_arithmetic_samples_k_answers_and_escalates_when_they_disagree():
    d = _t61_demo()                                          # FakeClient gives 3 different numbers
    r = d.chat("A pencil costs 2 dollars. How much do 3 pencils cost?")
    assert r["route"]["route"] == "frontier" and r["route"]["kind"] == "arithmetic"
    assert r["route"]["signal"] == "1/3 answers agree"
    assert d.small.calls == 3 and d.large.calls == 1


def test_t61_arithmetic_agreeing_answers_stay_local_and_are_cached():
    d = _t61_demo(small=AgreeingClient("qwen3.5-4b"))
    r = d.chat("What is 3 times 4?")
    assert r["route"]["route"] == "local" and r["route"]["signal"] == "3/3 answers agree"
    assert r["answer"].endswith("The answer is 12.")
    d.chat("What is 3 times 4?")
    assert d.small.calls == 3                                # second ask: all 3 answers from the cache


def test_t61_free_form_is_labelled_uncertified():
    d = _t61_demo(small=AgreeingClient("qwen3.5-4b"))
    r = d.route("Who wrote Romeo and Juliet?")
    assert r["kind"] == "free-form" and r["certified"] is False


def test_t61_forced_frontier_skips_the_local_model():
    d = _t61_demo()
    r = d.chat("What is 3 times 4?", force="frontier")
    assert r["route"]["route"] == "frontier" and r["route"]["forced"]
    assert d.small.calls == 0 and d.large.calls == 1


def test_done_reports_counterfactual_and_says_where_the_answer_came_from():
    """T47: the soak needs per-query counterfactual cost and a cache-hit flag it can trust."""
    d = _t61_demo(small=AgreeingClient("qwen3.5-4b"))
    first = d.chat("What is 3 times 4?")["done"]
    second = d.chat("What is 3 times 4?")["done"]
    assert first["cached"] is False and second["cached"] is True      # local: all k samples cached
    assert first["cost_usd"] == 0.0 and first["counterfactual_usd"] > 0

    mcq = d.chat("Which planet is known as the Red Planet? A) Venus B) Mars C) Jupiter D) Saturn")["done"]
    assert mcq["cached"] is False and mcq["counterfactual_usd"] > 0    # option scoring is a live pass

    f = _t61_demo()
    live = f.chat("A pencil costs 2 dollars. How much do 3 pencils cost?")["done"]
    hit = f.chat("A pencil costs 2 dollars. How much do 3 pencils cost?")["done"]
    assert live["cached"] is False and hit["cached"] is True
    assert live["cost_usd"] > 0 and np.isclose(live["cost_usd"], live["counterfactual_usd"])
