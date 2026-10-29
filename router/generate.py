"""One way to get answers out of any model, cached and resumable (TASKS.md T30, T31; lane D).

    from router.generate import load_client, GenerationCache, run_generation, GenParams
    client = load_client("qwen3.5-4b")                       # configs/clients.yaml
    cache = GenerationCache("artifacts/cache/generations.sqlite")
    responses = run_generation(corpus, client, cache, GenParams(max_tokens=512), k=3)

Clients
  MLXClient           a local model through mlx-lm on Apple Silicon (Qwen3.5-4B for the demo).
                      Qwen3 "thinking" is switched off in the chat template (enable_thinking=False).
                      Batched decoding (T31) uses mlx_lm.batch_generate when the installed mlx-lm
                      has it, and falls back to one prompt at a time otherwise.
  OpenAICompatClient  any OpenAI-compatible chat endpoint: Gemini (free tier), GitHub Models, Groq,
                      a local Ollama. Rate-limited, retries 429/5xx with backoff, reads token usage.
  FakeClient          deterministic, no model; for tests and for dry runs of the pipeline.

Cache
  SQLite, keyed by a content hash of (model, final prompt text, generation parameters, sample index).
  Every answer is written as soon as it arrives, so a run killed halfway resumes where it stopped,
  and re-running a finished job makes zero model calls (the T30 acceptance test).
  Changing the prompt template or any parameter changes the key, so stale answers are never reused.

Keys come from the environment (.env, never committed). Nothing here runs on import.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import pandas as pd
import yaml

from router import schema

GEN_VERSION = "generate-1.0"
DEFAULT_CLIENTS = Path(__file__).resolve().parent.parent / "configs" / "clients.yaml"

# Per-source prompt templates. The template is part of the cached prompt text, so editing one
# invalidates exactly the answers it affects. GSM8K asks for the "The answer is N." line that
# graders.extract_number looks for first.
PROMPT_TEMPLATES = {
    "gsm8k": "{prompt}\n\nSolve the problem step by step, briefly: short plain-text steps, no LaTeX, "
             "no restating the question. Finish with a final line of the form 'The answer is N.' "
             "where N is a number.",
    "default": "{prompt}",
}


def build_prompt(source: str | None, prompt: str) -> str:
    return PROMPT_TEMPLATES.get(source or "default", PROMPT_TEMPLATES["default"]).format(prompt=prompt)


@dataclass(frozen=True)
class GenParams:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    system: str | None = None
    enable_thinking: bool = False

    def key(self) -> dict:
        return asdict(self)


@dataclass
class Generation:
    text: str | None
    n_in: int
    n_out: int
    latency_ms: float | None = None
    mean_logprob: float | None = None
    cached: bool = False
    error: str | None = None


class Client(Protocol):
    name: str              # the model name written to RESPONSES.model and used in the cache key
    max_batch: int

    def generate_batch(self, prompts: Sequence[str], params: GenParams,
                       sample_idx: Sequence[int]) -> list[Generation]: ...


# ----------------------------------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------------------------------

def cache_key(model: str, prompt: str, params: GenParams, sample_idx: int) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "params": params.key(), "sample_idx": sample_idx},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GenerationCache:
    """Content-addressed store of generations. Safe to share between runs; one writer at a time."""

    def __init__(self, path: str | Path = "artifacts/cache/generations.sqlite"):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the demo server reads and writes from worker threads; callers that
        # share one cache across threads serialise access themselves (serve.py holds a lock).
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS gens (key TEXT PRIMARY KEY, model TEXT, sample_idx INTEGER, "
            "text TEXT, n_in INTEGER, n_out INTEGER, latency_ms REAL, mean_logprob REAL, created REAL)"
        )
        self.db.commit()

    def get(self, key: str) -> Generation | None:
        row = self.db.execute(
            "SELECT text, n_in, n_out, latency_ms, mean_logprob FROM gens WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return Generation(text=row[0], n_in=row[1], n_out=row[2], latency_ms=row[3], mean_logprob=row[4], cached=True)

    def put(self, key: str, model: str, sample_idx: int, g: Generation) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO gens VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, model, sample_idx, g.text, g.n_in, g.n_out, g.latency_ms, g.mean_logprob, time.time()),
        )
        self.db.commit()

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM gens").fetchone()[0]


# ----------------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------------

class OfflineMiss(RuntimeError):
    """offline=True and an answer is not in the cache."""


@dataclass
class RunStats:
    requested: int = 0
    cache_hits: int = 0
    model_calls: int = 0
    errors: int = 0
    seconds: float = 0.0
    errors_sample: list = field(default_factory=list)


def run_generation(
    corpus: pd.DataFrame,
    client: Client,
    cache: GenerationCache,
    params: GenParams = GenParams(),
    *,
    k: int = 1,
    offline: bool = False,
    progress: Callable[[int, int], None] | None = None,
    stats: RunStats | None = None,
) -> pd.DataFrame:
    """Answer every corpus row k times; return a RESPONSES frame. Cached answers are not re-asked.

    With temperature 0, k > 1 just repeats one answer, so k is forced to 1 there.
    Errors are recorded (text None, 0 tokens) but NOT cached, so the next run retries them.
    """
    stats = stats if stats is not None else RunStats()
    t0 = time.time()
    if params.temperature == 0 and k > 1:
        k = 1
    jobs = []
    for qid, source, prompt in zip(corpus["qid"], corpus.get("source", [None] * len(corpus)), corpus["prompt"]):
        text = build_prompt(source, prompt)
        for j in range(k):
            jobs.append((qid, j, text, cache_key(client.name, text, params, j)))
    stats.requested = len(jobs)

    results: dict[tuple[str, int], Generation] = {}
    todo = []
    for qid, j, text, key in jobs:
        hit = cache.get(key)
        if hit is not None:
            results[(qid, j)] = hit
            stats.cache_hits += 1
        else:
            todo.append((qid, j, text, key))
    if todo and offline:
        raise OfflineMiss(f"{len(todo)} of {len(jobs)} answers are not cached (first qid {todo[0][0]!r})")

    done = len(jobs) - len(todo)
    step = max(1, int(getattr(client, "max_batch", 1)))
    for start in range(0, len(todo), step):
        chunk = todo[start:start + step]
        try:
            gens = client.generate_batch([c[2] for c in chunk], params, [c[1] for c in chunk])
        except Exception as e:                                   # whole batch failed: record, retry next run
            gens = [Generation(None, 0, 0, error=f"{type(e).__name__}: {e}") for _ in chunk]
        stats.model_calls += len(chunk)
        for (qid, j, _text, key), g in zip(chunk, gens):
            results[(qid, j)] = g
            if g.error is None and g.text is not None:
                cache.put(key, client.name, j, g)
            else:
                stats.errors += 1
                if len(stats.errors_sample) < 5:
                    stats.errors_sample.append((qid, g.error))
        done += len(chunk)
        if progress:
            progress(done, len(jobs))

    rows = [{"qid": qid, "model": client.name, "sample_idx": j, "text": g.text, "n_in": int(g.n_in),
             "n_out": int(g.n_out), "mean_logprob": g.mean_logprob, "latency_ms": g.latency_ms}
            for (qid, j), g in sorted(results.items())]
    out = pd.DataFrame(rows, columns=schema.RESPONSES.names)
    out["mean_logprob"] = out["mean_logprob"].astype(float)
    out["latency_ms"] = out["latency_ms"].astype(float)
    stats.seconds = time.time() - t0
    return schema.validate(out, schema.RESPONSES)


# ----------------------------------------------------------------------------------------------------
# Clients
# ----------------------------------------------------------------------------------------------------

class FakeClient:
    """Deterministic stand-in: answers 'The answer is <len(prompt) % 97>.' and counts its calls."""

    def __init__(self, name: str = "fake", max_batch: int = 4, fail_on: Iterable[str] = ()):
        self.name = name
        self.max_batch = max_batch
        self.calls = 0
        self.fail_on = set(fail_on)

    def generate_batch(self, prompts, params, sample_idx):
        out = []
        for p, j in zip(prompts, sample_idx):
            self.calls += 1
            if any(f in p for f in self.fail_on):
                out.append(Generation(None, 0, 0, error="planted failure"))
                continue
            text = f"The answer is {(len(p) + j) % 97}."
            out.append(Generation(text, n_in=len(p) // 4, n_out=len(text) // 4, latency_ms=1.0, mean_logprob=-0.1))
        return out

    def stream(self, prompt: str, params: GenParams):
        """Yield the answer a word at a time (for the demo server's streaming path)."""
        g = self.generate_batch([prompt], params, [0])[0]
        for i, w in enumerate((g.text or "").split(" ")):
            yield w if i == 0 else " " + w


class MLXClient:
    """Local generation on Apple Silicon via mlx-lm. Loads the model once, on first use."""

    def __init__(self, name: str, path: str, max_batch: int = 1):
        self.name = name
        self.path = path
        self.max_batch = max_batch
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            try:
                from mlx_lm import load
            except ImportError as e:
                raise ImportError("mlx-lm is needed for local models and runs only on Apple Silicon: "
                                  "pip install mlx-lm") from e
            self._model, self._tok = load(self.path)
        return self._model, self._tok

    def format(self, prompt: str, params: GenParams) -> str:
        _, tok = self._load()
        messages = ([{"role": "system", "content": params.system}] if params.system else []) + \
                   [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=params.enable_thinking)
        except TypeError:                                          # templates without a thinking switch
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _sampler(self, params: GenParams):
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=params.temperature, top_p=params.top_p)

    def generate_one(self, prompt: str, params: GenParams, sample_idx: int = 0) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate

        model, tok = self._load()
        mx.random.seed(params.seed * 1000 + sample_idx)
        text_in = self.format(prompt, params)
        t0 = time.perf_counter()
        pieces, lps, n_in = [], [], 0
        for r in stream_generate(model, tok, text_in, max_tokens=params.max_tokens, sampler=self._sampler(params)):
            pieces.append(r.text)
            n_in = getattr(r, "prompt_tokens", n_in)
            if getattr(r, "logprobs", None) is not None and getattr(r, "token", None) is not None:
                lps.append(float(r.logprobs[r.token]))
        text = "".join(pieces)
        n_out = len(lps) if lps else len(tok.encode(text))
        return Generation(text, n_in=int(n_in or len(tok.encode(text_in))), n_out=int(n_out),
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          mean_logprob=(sum(lps) / len(lps)) if lps else None)

    def stream(self, prompt: str, params: GenParams):
        """Yield text pieces as the local model decodes them."""
        from mlx_lm import stream_generate

        model, tok = self._load()
        for r in stream_generate(model, tok, self.format(prompt, params), max_tokens=params.max_tokens,
                                 sampler=self._sampler(params)):
            if r.text:
                yield r.text

    def generate_batch(self, prompts, params, sample_idx):
        if self.max_batch <= 1 or len(prompts) == 1:
            return [self.generate_one(p, params, j) for p, j in zip(prompts, sample_idx)]
        try:
            from mlx_lm import batch_generate
        except ImportError:
            return [self.generate_one(p, params, j) for p, j in zip(prompts, sample_idx)]
        import mlx.core as mx

        model, tok = self._load()
        mx.random.seed(params.seed * 1000 + min(sample_idx))
        encoded = [tok.encode(self.format(p, params)) for p in prompts]
        t0 = time.perf_counter()
        try:
            res = batch_generate(model, tok, encoded, max_tokens=params.max_tokens,
                                 sampler=self._sampler(params), verbose=False)
            texts = res.texts if hasattr(res, "texts") else list(res)
            if len(texts) != len(prompts) or not all(isinstance(t, str) for t in texts):
                raise ValueError("unexpected batch_generate output")
        except Exception as e:   # batch_generate's signature differs between mlx-lm versions
            import warnings
            warnings.warn(f"mlx_lm.batch_generate failed ({type(e).__name__}: {e}); "
                          "falling back to one prompt at a time for the rest of this run")
            self.max_batch = 1
            return [self.generate_one(p, params, j) for p, j in zip(prompts, sample_idx)]
        ms = (time.perf_counter() - t0) * 1000 / len(prompts)       # amortised per prompt
        return [Generation(t, n_in=len(e), n_out=len(tok.encode(t)), latency_ms=ms) for t, e in zip(texts, encoded)]


class OpenAICompatClient:
    """Chat completions over HTTP for any OpenAI-compatible API (Gemini, GitHub Models, Groq, Ollama)."""

    def __init__(self, name: str, base_url: str, model_id: str, api_key_env: str | None = None,
                 rpm: float | None = None, max_retries: int = 6, timeout: float = 120.0,
                 extra_body: dict | None = None, transport=None, sleep: Callable[[float], None] = time.sleep):
        import httpx

        self.name = name
        self.model_id = model_id
        self.max_batch = 1
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self._sleep = sleep
        self._last = 0.0
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not key:
            raise RuntimeError(f"{api_key_env} is not set. Put it in .env (see .env.example); never commit it.")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout,
                                 transport=transport)

    def _wait_turn(self):
        if self.min_interval:
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                self._sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def list_models(self) -> list[str]:
        """IDs the endpoint serves; T40's check that the key works and the model id is right."""
        r = self.http.get("models")
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", [])]

    def generate_one(self, prompt: str, params: GenParams, sample_idx: int = 0) -> Generation:
        messages = ([{"role": "system", "content": params.system}] if params.system else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"model": self.model_id, "messages": messages, "max_tokens": params.max_tokens,
                "temperature": params.temperature, "top_p": params.top_p, **self.extra_body}
        if params.temperature > 0:
            body["seed"] = params.seed * 1000 + sample_idx
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            t0 = time.perf_counter()
            r = self.http.post("chat/completions", json=body)
            if r.status_code == 200:
                data = r.json()
                choice = data["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                return Generation(text, n_in=int(usage.get("prompt_tokens") or len(prompt) // 4),
                                  n_out=int(usage.get("completion_tokens") or len(text) // 4),
                                  latency_ms=(time.perf_counter() - t0) * 1000)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = r.headers.get("retry-after")
                wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else delay
                self._sleep(min(wait, 120.0))
                delay = min(delay * 2, 120.0)
                continue
            return Generation(None, 0, 0, error=f"HTTP {r.status_code}: {r.text[:200]}")
        return Generation(None, 0, 0, error="retries exhausted")

    def generate_batch(self, prompts, params, sample_idx):
        return [self.generate_one(p, params, j) for p, j in zip(prompts, sample_idx)]

    def stream(self, prompt: str, params: GenParams):
        """Yield text pieces from a streamed chat completion (server-sent events)."""
        messages = ([{"role": "system", "content": params.system}] if params.system else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"model": self.model_id, "messages": messages, "max_tokens": params.max_tokens,
                "temperature": params.temperature, "top_p": params.top_p, "stream": True, **self.extra_body}
        self._wait_turn()
        with self.http.stream("POST", "chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta") or {}
                except (ValueError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    yield delta["content"]


# ----------------------------------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------------------------------

def load_env(path: str | Path = ".env") -> None:
    """Read KEY=value lines from .env into os.environ (without overriding what is already set)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_client(name: str, config: str | Path = DEFAULT_CLIENTS, **overrides) -> Client:
    """Build a client from a preset in configs/clients.yaml."""
    presets = yaml.safe_load(Path(config).read_text())["clients"]
    if name not in presets:
        raise KeyError(f"no client preset {name!r}; known: {sorted(presets)} (configs/clients.yaml)")
    spec = {**presets[name], **overrides}
    kind = spec.pop("kind")
    if kind == "mlx":
        return MLXClient(name=name, **spec)
    if kind == "openai_compat":
        load_env()
        return OpenAICompatClient(name=name, **spec)
    if kind == "fake":
        return FakeClient(name=name, **spec)
    raise ValueError(f"{name}: unknown client kind {kind!r}")
