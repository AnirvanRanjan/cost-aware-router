"""Data contracts between pipeline stages (DESIGN.md §4.1).

Every stage reads one parquet and writes one. `write()` and `read()` validate
against the contract, so a malformed file fails loudly at the boundary instead
of silently corrupting a result three stages later.

Contracts freeze at the Week-2 gate. To change one after that: open a PR that
edits this file, bump CONTRACT_VERSION, and get a review from every lane that
reads the contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CONTRACT_VERSION = "1.1"  # 1.1: SOURCES gains the RouterBench datasets (additive)

SOURCES = ("gsm8k", "mmlu", "arena", "mbpp", "hellaswag", "arc-challenge", "winogrande", "mt-bench")
SPLITS = ("train", "cal", "test")


class ContractError(ValueError):
    """A dataframe does not satisfy its stage contract."""


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "str" | "int" | "float" | "vector"
    nullable: bool = False
    lo: float | None = None
    hi: float | None = None
    allowed: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Contract:
    name: str
    columns: tuple[Column, ...]
    key: tuple[str, ...]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]


CORPUS = Contract(
    name="corpus",
    columns=(
        Column("qid", "str"),
        Column("source", "str", allowed=SOURCES),
        Column("subject", "str", nullable=True),
        Column("group_id", "str"),
        Column("split", "str", allowed=SPLITS),
        Column("prompt", "str"),
        Column("reference", "str", nullable=True),
        Column("meta", "str", nullable=True),
    ),
    key=("qid",),
)

RESPONSES = Contract(
    name="responses",
    columns=(
        Column("qid", "str"),
        Column("model", "str"),
        Column("sample_idx", "int", lo=0),
        Column("text", "str", nullable=True),
        Column("n_in", "int", lo=0),
        Column("n_out", "int", lo=0),
        Column("mean_logprob", "float", nullable=True, hi=0.0),
        Column("latency_ms", "float", nullable=True, lo=0.0),
    ),
    key=("qid", "model", "sample_idx"),
)

# Amendment vs DESIGN.md §4.1: scores carries mean token counts, so the cost lane never
# needs responses.parquet. RouterBench ships per-item costs but no per-sample responses.
SCORES = Contract(
    name="scores",
    columns=(
        Column("qid", "str"),
        Column("model", "str"),
        Column("score", "float", lo=0.0, hi=1.0),
        Column("grader", "str"),
        Column("grader_version", "str"),
        Column("mean_n_in", "float", lo=0.0),
        Column("mean_n_out", "float", lo=0.0),
    ),
    key=("qid", "model"),
)

FEATURES = Contract(
    name="features",
    columns=(
        Column("qid", "str"),
        Column("feature_set", "str"),
        Column("vector", "vector"),
    ),
    key=("qid", "feature_set"),
)

PREDICTIONS = Contract(
    name="predictions",
    columns=(
        Column("qid", "str"),
        Column("router", "str"),
        Column("seed", "int", lo=0),
        Column("s_hat_S", "float", lo=0.0, hi=1.0),
        Column("s_hat_L", "float", lo=0.0, hi=1.0),
        Column("delta_hat", "float", lo=-1.0, hi=1.0),
        Column("c_hat_L", "float", lo=0.0),
    ),
    key=("qid", "router", "seed"),
)

CONTRACTS = {c.name: c for c in (CORPUS, RESPONSES, SCORES, FEATURES, PREDICTIONS)}


def _is_sequence(x) -> bool:
    return hasattr(x, "__len__") and not isinstance(x, (str, bytes))


def _check_column(s: pd.Series, col: Column) -> list[str]:
    errs: list[str] = []
    nulls = s.isna() if col.kind != "vector" else s.map(lambda x: x is None)
    if nulls.any() and not col.nullable:
        errs.append(f"{col.name}: {int(nulls.sum())} nulls in a non-nullable column")
    v = s[~nulls]
    if v.empty:
        return errs

    if col.kind == "str":
        non_str = ~v.map(lambda x: isinstance(x, str))
        if non_str.any():
            return errs + [f"{col.name}: {int(non_str.sum())} non-string values"]
        if not col.nullable and (v.str.len() == 0).any():
            errs.append(f"{col.name}: empty strings in a non-nullable column")
        if col.allowed is not None:
            bad = sorted(set(v) - set(col.allowed))
            if bad:
                errs.append(f"{col.name}: values {bad[:5]} not in allowed {list(col.allowed)}")
        return errs

    if col.kind == "vector":
        if (~v.map(_is_sequence)).any() or (v.map(lambda x: len(x) if _is_sequence(x) else 0) == 0).any():
            errs.append(f"{col.name}: every value must be a non-empty numeric sequence")
        return errs

    if col.kind == "int" and not pd.api.types.is_integer_dtype(v):
        return errs + [f"{col.name}: expected an integer dtype, got {s.dtype}"]
    if col.kind == "float":
        if not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
            return errs + [f"{col.name}: expected a numeric dtype, got {s.dtype}"]
        if not np.isfinite(v.to_numpy(dtype=float)).all():
            errs.append(f"{col.name}: non-finite values")
    if col.lo is not None and (v < col.lo).any():
        errs.append(f"{col.name}: {int((v < col.lo).sum())} values below {col.lo}")
    if col.hi is not None and (v > col.hi).any():
        errs.append(f"{col.name}: {int((v > col.hi).sum())} values above {col.hi}")
    return errs


def validate(df: pd.DataFrame, contract: Contract) -> pd.DataFrame:
    """Return `df` unchanged if it satisfies `contract`; raise ContractError listing every problem."""
    errors: list[str] = []
    missing = [n for n in contract.names if n not in df.columns]
    if missing:
        errors.append(f"missing columns {missing}")
    extra = [c for c in df.columns if c not in contract.names and not str(c).startswith("x_")]
    if extra:
        errors.append(f"unexpected columns {extra} (prefix experimental columns with 'x_')")
    for col in contract.columns:
        if col.name in df.columns:
            errors.extend(_check_column(df[col.name], col))
    if all(k in df.columns for k in contract.key):
        n_dup = int(df.duplicated(list(contract.key)).sum())
        if n_dup:
            errors.append(f"{n_dup} duplicate rows on key {contract.key}")
    if contract is FEATURES and not errors:
        lengths = df.groupby("feature_set")["vector"].agg(lambda s: s.map(len).nunique())
        ragged = list(lengths[lengths > 1].index)
        if ragged:
            errors.append(f"vector lengths differ within feature_set {ragged}")
    if errors:
        raise ContractError(f"[{contract.name}] " + "; ".join(errors))
    return df


def write(df: pd.DataFrame, contract: Contract, path: str | Path) -> Path:
    """Validate, then write parquet. The only way stages should write their output."""
    validate(df, contract)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read(path: str | Path, contract: Contract) -> pd.DataFrame:
    """Read parquet, then validate. The only way stages should read their input."""
    return validate(pd.read_parquet(path), contract)


def content_hash(df: pd.DataFrame) -> str:
    """Stable sha256 of a frame's contents (column order and values; ignores the index).

    Used in manifest.json to name artifacts, so "which scores.parquet was this?" always has an answer.
    """
    frame = df.copy()
    for c in frame.columns:
        if frame[c].map(_is_sequence).any():
            frame[c] = frame[c].map(lambda x: tuple(np.asarray(x, dtype=float).round(12)) if _is_sequence(x) else x)
    h = hashlib.sha256()
    h.update(",".join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()[:16]
