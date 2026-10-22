"""Cost model (DESIGN.md §3.2). Every dollar figure in the project goes through here.

Two cost models, both reported:
  M1 — pure API spend: local models cost $0. The headline number.
  M2 — fully loaded: local generation priced as hardware time (see `local_cost`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DEFAULT_PRICES = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
PER_MTOK = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    name: str
    provider: str
    input: float   # USD per 1M input tokens
    output: float  # USD per 1M output tokens
    free_tier: bool = False
    tokens_per_sec: float | None = None  # local models only
    measured: bool = False               # True once lane D has timed tokens_per_sec on the M2

    @property
    def is_local(self) -> bool:
        return self.provider == "local"


class PriceTable:
    def __init__(
        self,
        models: dict[str, ModelPrice],
        *,
        batch_discount: float,
        cache_read_multiplier: float,
        as_of: str,
        reference_frontier: str,
        reference_local: str,
        hardware: dict[str, "Hardware"] | None = None,
    ):
        self.models = models
        self.hardware = hardware or {}
        self.batch_discount = batch_discount
        self.cache_read_multiplier = cache_read_multiplier
        self.as_of = as_of
        self.reference_frontier = reference_frontier
        self.reference_local = reference_local

    @classmethod
    def load(cls, path: str | Path = DEFAULT_PRICES) -> "PriceTable":
        raw = yaml.safe_load(Path(path).read_text())
        models = {}
        for name, spec in raw["models"].items():
            spec = dict(spec)
            alias = spec.pop("price_as", None)
            if alias is not None:
                if alias not in raw["models"] or "price_as" in raw["models"][alias]:
                    raise KeyError(f"{name}: price_as {alias!r} must name a model with its own prices")
                base = raw["models"][alias]
                spec = {"provider": base["provider"], "input": base["input"], "output": base["output"], **spec}
            models[name] = ModelPrice(name=name, **spec)
        return cls(
            models,
            batch_discount=float(raw["batch_discount"]),
            cache_read_multiplier=float(raw["cache_read_multiplier"]),
            as_of=str(raw["as_of"]),
            reference_frontier=raw["reference_frontier"],
            reference_local=raw["reference_local"],
            hardware={k: Hardware(name=k, **v) for k, v in (raw.get("hardware") or {}).items()},
        )

    def __getitem__(self, name: str) -> ModelPrice:
        try:
            return self.models[name]
        except KeyError:
            raise KeyError(
                f"no price for model {name!r}; known: {sorted(self.models)}. Add it to configs/models.yaml"
            ) from None

    def query_cost(self, model: str, n_in, n_out, *, batch: bool = False, cached_in=0):
        """Dollars for one or many queries (M1). Accepts scalars or arrays of token counts."""
        p = self[model]
        n_in = np.asarray(n_in, dtype=float)
        n_out = np.asarray(n_out, dtype=float)
        cached = np.asarray(cached_in, dtype=float)
        if (n_in < 0).any() or (n_out < 0).any() or (cached < 0).any():
            raise ValueError("token counts must be non-negative")
        if (cached > n_in).any():
            raise ValueError("cached_in cannot exceed n_in")
        usd = (
            p.input * (n_in - cached)
            + p.input * self.cache_read_multiplier * cached
            + p.output * n_out
        ) / PER_MTOK
        if batch:
            usd = usd * (1.0 - self.batch_discount)
        return float(usd) if usd.ndim == 0 else usd


def frame_cost(scores: pd.DataFrame, table: PriceTable, *, batch: bool = False) -> pd.Series:
    """Per-row M1 cost for a scores.parquet frame, from its mean token counts."""
    out = pd.Series(0.0, index=scores.index)
    for model, rows in scores.groupby("model"):
        out.loc[rows.index] = table.query_cost(model, rows["mean_n_in"], rows["mean_n_out"], batch=batch)
    return out


@dataclass(frozen=True)
class Hardware:
    """A local machine priced by the hour for cost model M2. Every field is an assumption."""

    name: str
    price_usd: float
    lifetime_years: float
    hours_per_day: float
    watts: float
    usd_per_kwh: float
    prefill_tokens_per_sec: float | None = None

    @property
    def usd_per_hour(self) -> float:
        amortised = self.price_usd / (self.lifetime_years * 365.0 * self.hours_per_day)
        energy = self.watts / 1000.0 * self.usd_per_kwh
        return amortised + energy


def local_cost(n_out, *, usd_per_hour: float, tokens_per_sec: float):
    """Cost model M2: amortized hardware cost of generating `n_out` tokens locally (decode only)."""
    if tokens_per_sec <= 0:
        raise ValueError("tokens_per_sec must be positive")
    usd = np.asarray(n_out, dtype=float) / tokens_per_sec / 3600.0 * usd_per_hour
    return float(usd) if usd.ndim == 0 else usd


def wall_clock_hours(total_out_tokens: float, tokens_per_sec: float, batch_speedup: float = 1.0) -> float:
    """Hours to decode `total_out_tokens` locally. `batch_speedup` models batched MLX decoding."""
    if tokens_per_sec <= 0 or batch_speedup <= 0:
        raise ValueError("tokens_per_sec and batch_speedup must be positive")
    return total_out_tokens / (tokens_per_sec * batch_speedup) / 3600.0


@dataclass(frozen=True)
class RunEstimate:
    model: str
    n_items: int
    k: int
    tokens_in: float
    tokens_out: float
    usd: float
    hours: float | None


def estimate_run(
    table: PriceTable,
    model: str,
    *,
    n_items: int,
    k: int,
    mean_in: float,
    mean_out: float,
    batch: bool = False,
    batch_speedup: float = 1.0,
) -> RunEstimate:
    """Projected tokens, dollars and (for local models) hours for a generation run. The Phase-0 gate."""
    tokens_in = float(n_items * k * mean_in)
    tokens_out = float(n_items * k * mean_out)
    usd = table.query_cost(model, tokens_in, tokens_out, batch=batch)
    p = table[model]
    hours = (
        wall_clock_hours(tokens_out, p.tokens_per_sec, batch_speedup)
        if p.is_local and p.tokens_per_sec
        else None
    )
    return RunEstimate(model, n_items, k, tokens_in, tokens_out, usd, hours)


def local_query_cost(n_in, n_out, *, hardware: Hardware, tokens_per_sec: float):
    """M2 dollars for one or many local queries: decode time, plus prefill time when it is known."""
    usd = local_cost(n_out, usd_per_hour=hardware.usd_per_hour, tokens_per_sec=tokens_per_sec)
    if hardware.prefill_tokens_per_sec:
        usd = usd + local_cost(n_in, usd_per_hour=hardware.usd_per_hour,
                               tokens_per_sec=hardware.prefill_tokens_per_sec)
    return usd


def crossover_tokens_per_sec(table: PriceTable, api_model: str, *, hardware: Hardware,
                             n_in: float, n_out: float, batch: bool = False) -> float:
    """Decode speed below which local generation (M2) costs MORE than calling `api_model`.

    Solves  hw_cost(n_in, n_out, tps) = api_cost(n_in, n_out)  for tps. Returns inf if prefill alone
    already costs more than the API call (no decode speed can win).
    """
    api = table.query_cost(api_model, n_in, n_out, batch=batch)
    prefill = 0.0
    if hardware.prefill_tokens_per_sec:
        prefill = local_cost(n_in, usd_per_hour=hardware.usd_per_hour, tokens_per_sec=hardware.prefill_tokens_per_sec)
    budget = api - prefill
    if budget <= 0:
        return float("inf")
    return n_out * hardware.usd_per_hour / 3600.0 / budget


def pair_costs(
    table: PriceTable,
    scores: pd.DataFrame,
    *,
    small: str,
    large: str,
    cost_model: str = "M1",
    hardware: str | None = None,
    small_tokens_per_sec: float | None = None,
    frontier_batch: bool = False,
) -> pd.DataFrame:
    """Per-question c_S and c_L for one (small, large) pair, indexed by qid.

    cost_model "M1": c_S = 0 (local is free), c_L at list price (or Batch price with frontier_batch).
    cost_model "M2": c_S is hardware time on `hardware` at `small_tokens_per_sec` (defaults to the
    small model's own speed in models.yaml, else the reference local model's).
    """
    if cost_model not in ("M1", "M2"):
        raise ValueError("cost_model must be 'M1' or 'M2'")
    s = scores[scores["model"] == small].set_index("qid")
    lg = scores[scores["model"] == large].set_index("qid")
    qids = s.index.intersection(lg.index)
    s, lg = s.loc[qids], lg.loc[qids]
    c_L = table.query_cost(large, lg["mean_n_in"], lg["mean_n_out"], batch=frontier_batch)
    if cost_model == "M1":
        c_S = table.query_cost(small, s["mean_n_in"], s["mean_n_out"]) if small in table.models else np.zeros(len(qids))
    else:
        hw = table.hardware[hardware or next(iter(table.hardware))]
        tps = small_tokens_per_sec
        if tps is None:
            own = table.models.get(small)
            tps = (own.tokens_per_sec if own and own.tokens_per_sec else None) or table[table.reference_local].tokens_per_sec
        c_S = local_query_cost(s["mean_n_in"], s["mean_n_out"], hardware=hw, tokens_per_sec=tps)
    return pd.DataFrame({"c_S": np.asarray(c_S, dtype=float), "c_L": np.asarray(c_L, dtype=float)}, index=qids)
