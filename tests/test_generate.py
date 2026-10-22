"""T30: one client interface, content-hash cache, resumable runs (router/generate.py)."""
import httpx
import pandas as pd
import pytest

from router import generate as gen
from router import schema


def _corpus(n=10):
    return pd.DataFrame({"qid": [f"q{i:02d}" for i in range(n)], "source": ["gsm8k", "mmlu"] * (n // 2),
                         "prompt": [f"question number {i}" for i in range(n)]})


def test_rerunning_makes_zero_model_calls(tmp_path):
    cache = gen.GenerationCache(tmp_path / "c.sqlite")
    client = gen.FakeClient()
    params = gen.GenParams(temperature=0.7)
    first = gen.run_generation(_corpus(), client, cache, params, k=3)
    assert client.calls == 30 and len(cache) == 30
    second = gen.run_generation(_corpus(), client, gen.GenerationCache(tmp_path / "c.sqlite"), params, k=3)
    assert client.calls == 30                                  # nothing re-asked
    pd.testing.assert_frame_equal(first, second)
    schema.validate(second, schema.RESPONSES)


def test_resumes_after_a_crash(tmp_path):
    cache = gen.GenerationCache(tmp_path / "c.sqlite")

    class Crashy(gen.FakeClient):
        def generate_batch(self, prompts, params, sample_idx):
            if self.calls >= 4:
                raise KeyboardInterrupt
            return super().generate_batch(prompts, params, sample_idx)

    with pytest.raises(KeyboardInterrupt):
        gen.run_generation(_corpus(), Crashy(max_batch=2), cache)
    assert len(cache) == 4                                     # everything finished before the crash is kept
    client = gen.FakeClient(max_batch=2)
    out = gen.run_generation(_corpus(), client, cache)
    assert client.calls == 6 and len(out) == 10


def test_greedy_forces_k_to_one(tmp_path):
    out = gen.run_generation(_corpus(4), gen.FakeClient(), gen.GenerationCache(":memory:"), gen.GenParams(), k=5)
    assert len(out) == 4 and (out["sample_idx"] == 0).all()


def test_any_change_in_prompt_or_params_misses_the_cache():
    p = gen.GenParams()
    k = gen.cache_key("m", "hello", p, 0)
    assert k == gen.cache_key("m", "hello", gen.GenParams(), 0)
    assert k != gen.cache_key("m", "hello!", p, 0)
    assert k != gen.cache_key("m2", "hello", p, 0)
    assert k != gen.cache_key("m", "hello", gen.GenParams(max_tokens=10), 0)
    assert k != gen.cache_key("m", "hello", p, 1)


def test_gsm8k_prompt_asks_for_the_answer_line():
    assert "The answer is" in gen.build_prompt("gsm8k", "2+2?")
    assert gen.build_prompt("mmlu", "Q") == "Q"


def test_errors_are_recorded_but_not_cached(tmp_path):
    cache = gen.GenerationCache(":memory:")
    stats = gen.RunStats()
    out = gen.run_generation(_corpus(4), gen.FakeClient(fail_on=["number 1"]), cache, stats=stats)
    assert stats.errors == 1 and len(cache) == 3
    assert out.loc[out["qid"] == "q01", "text"].isna().all()


def test_offline_mode_refuses_to_call_a_model():
    cache = gen.GenerationCache(":memory:")
    gen.run_generation(_corpus(4), gen.FakeClient(), cache)
    gen.run_generation(_corpus(4), gen.FakeClient(), cache, offline=True)       # all cached: fine
    with pytest.raises(gen.OfflineMiss):
        gen.run_generation(_corpus(6), gen.FakeClient(), cache, offline=True)


def _api(handler, monkeypatch, **kw):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    return gen.OpenAICompatClient("api", "https://example.test/v1", "m-1", api_key_env="TEST_KEY",
                                  transport=httpx.MockTransport(handler), sleep=lambda s: None, **kw)


def test_openai_compat_reads_text_and_usage(monkeypatch):
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.read()
        return httpx.Response(200, json={"choices": [{"message": {"content": "The answer is 4."}}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 6}})

    g = _api(handler, monkeypatch).generate_one("2+2?", gen.GenParams())
    assert g.text == "The answer is 4." and (g.n_in, g.n_out) == (12, 6)
    assert seen["auth"] == "Bearer sk-test" and b'"model":"m-1"' in seen["body"].replace(b" ", b"")


def test_openai_compat_retries_429_then_succeeds(monkeypatch):
    calls = {"n": 0}
    waits = []

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    c = _api(handler, monkeypatch)
    c._sleep = waits.append
    g = c.generate_one("hi", gen.GenParams())
    assert g.text == "ok" and calls["n"] == 3 and waits == [7.0, 7.0]


def test_openai_compat_gives_up_on_client_errors(monkeypatch):
    g = _api(lambda req: httpx.Response(400, text="bad model"), monkeypatch).generate_one("hi", gen.GenParams())
    assert g.text is None and "400" in g.error


def test_missing_key_says_where_to_put_it(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    with pytest.raises(RuntimeError, match=".env"):
        gen.OpenAICompatClient("x", "https://example.test", "m", api_key_env="NOPE_KEY")


def test_presets_load(monkeypatch):
    assert isinstance(gen.load_client("fake"), gen.FakeClient)
    mlx = gen.load_client("qwen3.5-4b")
    assert isinstance(mlx, gen.MLXClient) and mlx.max_batch >= 1        # loading weights is lazy
    with pytest.raises(KeyError, match="clients.yaml"):
        gen.load_client("gpt-9000")
