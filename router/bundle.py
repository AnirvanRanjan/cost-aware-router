"""The router as one file the demo can load: features, heads, calibrators and tau* (T41 interface).

    bundle = RouterBundle.load("artifacts/bundle/router.pkl")
    d = bundle.decide("What is the derivative of x^2 sin x?")
    d.route        # "local" or "frontier"
    d.s_hat_S, d.s_hat_L, d.score, d.tau, d.router_ms

Anything with `transform(list[str]) -> array` can be the featurizer, so lane B can swap the interim
F0 + TF-IDF featurizer for F2 embeddings without touching the server: build a new bundle, same file.

Rule v1: go to the frontier iff  delta_hat = s_hat_L - s_hat_S > tau.
Rule v2: go to the frontier iff  delta_hat / c_hat_L(q) > tau, with c_hat_L from prompt length.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

BUNDLE_VERSION = "bundle-1.0"


class Featurizer(Protocol):
    def transform(self, texts: Sequence[str]) -> np.ndarray: ...


class F0F1Featurizer:
    """Interim featurizer: 23 heuristics (z-scored) + SVD of TF-IDF. No embeddings, CPU only, fast."""

    name = "f0+svd(f1)"

    def __init__(self, mean: np.ndarray, std: np.ndarray, tfidf: Any, svd: Any):
        self.mean, self.std, self.tfidf, self.svd = mean, std, tfidf, svd

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        from router.features import heuristics

        x0, _ = heuristics(list(texts))
        x0 = (np.nan_to_num(np.asarray(x0, dtype=np.float64)) - self.mean) / self.std
        x1 = self.svd.transform(self.tfidf.transform(list(texts)))
        return np.hstack([x0, x1]).astype(np.float32)


class F0F2Featurizer:
    """Production headline featurizer: 768-d BGE dense embeddings + 23 z-scored heuristics (791 dims).

    Sub-15ms compliance: encodes single queries via SentenceEmbeddingExtractor.
    """

    name = "f0+f2"

    def __init__(
        self,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        extractor: Any = None,
    ):
        self.mean = mean
        self.std = std
        self.model_name = model_name
        self.device = device
        self._extractor = extractor

    @property
    def extractor(self):
        if self._extractor is None:
            from router.features import SentenceEmbeddingExtractor

            self._extractor = SentenceEmbeddingExtractor(
                model_name=self.model_name,
                device=self.device,
                normalize_embeddings=True,
            )
        return self._extractor

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_extractor"] = None
        return state

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        from router.features import heuristics

        text_list = list(texts)
        embeddings, _ = self.extractor.encode(text_list)
        x0, _ = heuristics(text_list)
        x0_arr = np.nan_to_num(np.asarray(x0, dtype=np.float32))

        if self.mean is not None and self.std is not None:
            std_safe = np.where(self.std < 1e-8, 1.0, self.std)
            x0_norm = (x0_arr - self.mean) / std_safe
        else:
            x0_norm = x0_arr

        return np.concatenate([embeddings, x0_norm], axis=1).astype(np.float32)


class F2Featurizer:
    """Dense embedding featurizer: 768-d BGE dense embeddings alone."""

    name = "f2_embeddings"

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        extractor: Any = None,
    ):
        self.model_name = model_name
        self.device = device
        self._extractor = extractor

    @property
    def extractor(self):
        if self._extractor is None:
            from router.features import SentenceEmbeddingExtractor

            self._extractor = SentenceEmbeddingExtractor(
                model_name=self.model_name,
                device=self.device,
                normalize_embeddings=True,
            )
        return self._extractor

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_extractor"] = None
        return state

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        embeddings, _ = self.extractor.encode(list(texts))
        return np.asarray(embeddings, dtype=np.float32)


class CascadeFeaturizer:
    """Answer-time and confidence featurizer for the cascade router (D-12, D-14).

    Extracts answer-time signals using router.cascade.extract_single_answer_features:
    arithmetic slips, length, steps, hedge words, letters, option probability, margin, entropy, agreement.
    """

    name = "cascade_1sample"

    def __init__(self, headline_sources: Sequence[str] = ("mmlu", "gsm8k", "arc-challenge", "winogrande")):
        self.headline_sources = tuple(headline_sources)

    def transform_single(
        self,
        prompt: str,
        response: Any = None,
        source: str | None = None,
        option_probs: dict[str, float] | None = None,
        p_chosen: float | None = None,
        mean_logprob: float | None = None,
    ) -> np.ndarray:
        from router.cascade import extract_single_answer_features

        feats = extract_single_answer_features(
            prompt=prompt,
            response=response,
            source=source,
            option_probs=option_probs,
            p_chosen=p_chosen,
            mean_logprob=mean_logprob,
        )
        return np.asarray([feats], dtype=np.float32)

    def transform(
        self,
        texts: Sequence[str],
        responses: Sequence[Any] | None = None,
        sources: Sequence[str] | None = None,
        option_probs_list: Sequence[dict[str, float]] | None = None,
        p_chosen_list: Sequence[float] | None = None,
        mean_logprobs_list: Sequence[float | None] | None = None,
        **kwargs,
    ) -> np.ndarray:
        from router.cascade import extract_cascade_features

        n = len(texts)
        resps = responses if responses is not None else [None] * n
        return extract_cascade_features(
            prompts=texts,
            responses=resps,
            sources=sources,
            option_probs_list=option_probs_list,
            p_chosen_list=p_chosen_list,
            mean_logprobs_list=mean_logprobs_list,
        )


@dataclass
class Decision:
    route: str               # "local" | "frontier"
    s_hat_S: float
    s_hat_L: float
    delta_hat: float
    score: float             # what is compared with tau (delta_hat for v1, gain per $ for v2)
    tau: float
    rule: str
    router_ms: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class RouterBundle:
    featurizer: Any
    router: Any                          # has predict_proba(X) or predict_gap(X)
    tau: float
    rule: str = "v1"
    frontier_price: tuple[float, float] = (3.0, 15.0)   # USD per 1M tokens in / out, for rule v2
    mean_out_tokens: float = 150.0
    meta: dict = field(default_factory=dict)
    version: str = BUNDLE_VERSION
    _is_cascade: bool = False

    @property
    def is_cascade(self) -> bool:
        return bool(
            getattr(self, "_is_cascade", False)
            or self.meta.get("is_cascade", False)
            or getattr(self.featurizer, "name", "").startswith("cascade")
        )

    def predict(
        self,
        texts: Sequence[str],
        responses: Sequence[Any] | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        if hasattr(self.featurizer, "transform"):
            try:
                X = self.featurizer.transform(list(texts), responses=responses, **kwargs)
            except TypeError:
                X = self.featurizer.transform(list(texts))
        else:
            X = np.asarray(texts)

        if hasattr(self.router, "predict_proba"):
            try:
                s_S, s_L = self.router.predict_proba(X, calibrated=True)
            except TypeError:
                s_S, s_L = self.router.predict_proba(X)
        elif hasattr(self.router, "predict_gap"):
            delta = self.router.predict_gap(X)
            s_S = np.clip(0.5 - delta / 2.0, 0.0, 1.0)
            s_L = np.clip(0.5 + delta / 2.0, 0.0, 1.0)
        else:
            raise TypeError("router must implement predict_proba or predict_gap")

        s_S, s_L = np.asarray(s_S, dtype=float), np.asarray(s_L, dtype=float)
        delta = np.clip(s_L - s_S, -1.0, 1.0)
        if self.rule == "v2":
            n_in = np.array([max(1, len(t) // 4) for t in texts], dtype=float)
            c_L = (self.frontier_price[0] * n_in + self.frontier_price[1] * self.mean_out_tokens) / 1e6
            score = delta / np.maximum(c_L, 1e-9)
        else:
            score = delta
        return {
            "s_hat_S": s_S,
            "s_hat_L": s_L,
            "delta_hat": delta,
            "score": score,
            "frontier": score > self.tau,
        }

    def decide(
        self,
        text: str,
        response: Any = None,
        confidence: float | None = None,
        source: str | None = None,
        mean_logprob: float | None = None,
    ) -> Decision:
        t0 = time.perf_counter()
        kwargs = {}
        if confidence is not None:
            kwargs["p_chosen_list"] = [confidence]
        if source is not None:
            kwargs["sources"] = [source]
        if mean_logprob is not None:
            kwargs["mean_logprobs_list"] = [mean_logprob]
        p = self.predict([text], responses=[response] if response is not None else None, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        return Decision(
            route="frontier" if p["frontier"][0] else "local",
            s_hat_S=float(p["s_hat_S"][0]),
            s_hat_L=float(p["s_hat_L"][0]),
            delta_hat=float(p["delta_hat"][0]),
            score=float(p["score"][0]),
            tau=float(self.tau),
            rule=self.rule,
            router_ms=ms,
        )

    def decide_cascade(
        self,
        text: str,
        response: Any,
        confidence: float | None = None,
        source: str | None = None,
        mean_logprob: float | None = None,
    ) -> Decision:
        return self.decide(text, response=response, confidence=confidence, source=source, mean_logprob=mean_logprob)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RouterBundle:
        """Only load bundles you built yourself: pickle runs code from the file."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} is not a RouterBundle")
        return obj


class ConstantRouter:
    """A stand-in head for tests and for running the server before any bundle exists."""

    def __init__(self, s_S: float = 0.7, s_L: float = 0.9):
        self.s_S, self.s_L = s_S, s_L

    def predict_proba(self, X, calibrated: bool = True):
        n = len(X)
        return np.full(n, self.s_S), np.full(n, self.s_L)


class LengthFeaturizer:
    """Stand-in featurizer: one feature, prompt length. Pairs with ConstantRouter / tests."""

    def transform(self, texts):
        return np.array([[len(t)] for t in texts], dtype=np.float32)
