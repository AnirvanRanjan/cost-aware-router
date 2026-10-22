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
    router: Any                          # has predict_proba(X, calibrated=True) -> (s_S, s_L)
    tau: float
    rule: str = "v1"
    frontier_price: tuple[float, float] = (3.0, 15.0)   # USD per 1M tokens in / out, for rule v2
    mean_out_tokens: float = 150.0
    meta: dict = field(default_factory=dict)
    version: str = BUNDLE_VERSION

    def predict(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        X = self.featurizer.transform(list(texts))
        s_S, s_L = self.router.predict_proba(X, calibrated=True)
        s_S, s_L = np.asarray(s_S, dtype=float), np.asarray(s_L, dtype=float)
        delta = np.clip(s_L - s_S, -1.0, 1.0)
        if self.rule == "v2":
            n_in = np.array([max(1, len(t) // 4) for t in texts], dtype=float)
            c_L = (self.frontier_price[0] * n_in + self.frontier_price[1] * self.mean_out_tokens) / 1e6
            score = delta / np.maximum(c_L, 1e-9)
        else:
            score = delta
        return {"s_hat_S": s_S, "s_hat_L": s_L, "delta_hat": delta, "score": score,
                "frontier": score > self.tau}

    def decide(self, text: str) -> Decision:
        t0 = time.perf_counter()
        p = self.predict([text])
        ms = (time.perf_counter() - t0) * 1000
        return Decision(route="frontier" if p["frontier"][0] else "local", s_hat_S=float(p["s_hat_S"][0]),
                        s_hat_L=float(p["s_hat_L"][0]), delta_hat=float(p["delta_hat"][0]),
                        score=float(p["score"][0]), tau=float(self.tau), rule=self.rule, router_ms=ms)

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
