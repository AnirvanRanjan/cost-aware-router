"""RouterBench loader (Track A, DESIGN.md Rev A). Turns routerbench_0shot.pkl into the
CORPUS and SCORES contracts.

What it does, each step justified by notes/routerbench_audit.md:
  * keeps the seven datasets from the paper and drops the 21 small custom tasks;
  * parses prompts and responses, which RouterBench stores as stringified Python lists;
  * re-grades GSM8K by numeric exact match against the GSM8K train-split answers, because the
    stored GSM8K scores are a five-level rubric, not correctness. Without the answers file,
    GSM8K is left out rather than loaded with misleading labels;
  * estimates token counts from the text, since RouterBench stores cost but no tokens;
  * groups exact-duplicate prompts under one group_id, then splits 60/20/20 per dataset so no
    group crosses splits. Week 2's dedup.py refines group_id with near-duplicates.

Model columns are kept under their RouterBench names; configs/models.yaml prices them at 2026 rates.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from router import graders, schema
from router.splits import assert_no_leakage, grouped_split

LOADER_VERSION = "routerbench-loader-1.0"
ROUTERBENCH_GRADER_VERSION = "routerbench-0shot"

MAIN_SOURCES = ("mmlu", "hellaswag", "gsm8k", "arc-challenge", "winogrande", "mbpp", "mt-bench")
_GRADER = {
    "mmlu": "routerbench_mcq", "hellaswag": "routerbench_mcq", "arc-challenge": "routerbench_mcq",
    "winogrande": "routerbench_mcq", "mbpp": "routerbench_exec", "mt-bench": "routerbench_judge",
}


def source_of(eval_name: str) -> str | None:
    """Map a RouterBench eval_name to one of MAIN_SOURCES, or None for tasks we drop."""
    e = str(eval_name)
    if e.startswith("mmlu-"):
        return "mmlu"
    if e == "grade-school-math":
        return "gsm8k"
    if e.startswith("mtbench"):
        return "mt-bench"
    return e if e in ("hellaswag", "arc-challenge", "winogrande", "mbpp") else None


def parse_turns(cell) -> list[str]:
    """RouterBench stores "['turn 1', 'turn 2']" as a string. Return the list of turns."""
    if isinstance(cell, (list, tuple, np.ndarray)):
        return [str(x) for x in cell]
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    s = str(cell)
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
    return [s]


def query_text(source: str, turns: list[str]) -> str:
    """The text a router sees. For GSM8K that is the question alone: turn 0 is a fixed few-shot
    preamble identical on every item. For MT-Bench both user turns are kept."""
    if not turns:
        return ""
    if source == "gsm8k":
        return turns[-1].strip()
    if source == "mt-bench" and len(turns) > 1:
        return "\n\n".join(f"[Turn {i + 1}] {t.strip()}" for i, t in enumerate(turns))
    return "\n\n".join(t.strip() for t in turns)


def approx_tokens(text: str) -> int:
    """~4 characters per token, the usual English heuristic. Good enough to re-price at 2026 rates;
    swap in a real tokenizer here if per-item cost precision ever matters."""
    return max(1, math.ceil(len(text) / 4))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def load_gsm8k_gold(train: pd.DataFrame) -> dict[str, float]:
    """Map normalised question -> gold number, from the GSM8K train split (`question`, `answer`)."""
    gold = {}
    for q, a in zip(train["question"], train["answer"]):
        g = graders.gsm8k_gold(a)
        if g is not None:
            gold[_norm(q)] = g
    return gold


@dataclass
class LoadResult:
    corpus: pd.DataFrame
    scores: pd.DataFrame
    report: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def load_routerbench(
    df: pd.DataFrame,
    *,
    large: str,
    small: list[str],
    gsm8k_gold: dict[str, float] | None = None,
    keep_unverified_gsm8k: bool = False,
    fractions: dict | None = None,
    seed: int = 7,
    source_file: str = "routerbench_0shot.pkl",
) -> LoadResult:
    models = [large, *small]
    missing = [m for m in models if m not in df.columns or f"{m}|model_response" not in df.columns]
    if missing:
        available = sorted(c[: -len("|total_cost")] for c in df.columns if str(c).endswith("|total_cost"))
        raise KeyError(f"models not in the file: {missing}. Available: {available}")

    warnings: list[str] = []
    report: dict = {"rows_in": int(len(df)), "models": {"large": large, "small": list(small)}}

    src = df["eval_name"].map(source_of)
    report["dropped_tasks"] = {k: int(v) for k, v in df.loc[src.isna(), "eval_name"].value_counts().items()}
    work = df.loc[src.notna()].copy()
    work["source"] = src[src.notna()]

    if gsm8k_gold is None and not keep_unverified_gsm8k:
        n = int((work["source"] == "gsm8k").sum())
        if n:
            warnings.append(f"GSM8K left out ({n} items): no answers file, and RouterBench's GSM8K scores "
                            "are a rubric, not correctness. Pass the GSM8K train answers to include it.")
        work = work[work["source"] != "gsm8k"]

    turns = work["prompt"].map(parse_turns)
    work["query"] = [query_text(s, t) for s, t in zip(work["source"], turns)]
    work["n_in"] = [approx_tokens("\n".join(t)) for t in turns]

    reference = pd.Series(None, index=work.index, dtype=object)
    is_gsm = work["source"] == "gsm8k"
    if gsm8k_gold is not None and is_gsm.any():
        gold = work.loc[is_gsm, "query"].map(lambda q: gsm8k_gold.get(_norm(q)))
        unmatched = gold.isna()
        report["gsm8k"] = {"matched": int((~unmatched).sum()), "unmatched_dropped": int(unmatched.sum())}
        if unmatched.any():
            warnings.append(f"GSM8K: {int(unmatched.sum())} questions had no match in the answers file and were dropped.")
        work = work.drop(index=gold[unmatched].index)
        reference.loc[gold[~unmatched].index] = gold[~unmatched].map(lambda g: f"{g:g}")
        is_gsm = work["source"] == "gsm8k"

    score_rows = []
    bad = set()
    for m in models:
        responses = work[f"{m}|model_response"].map(lambda c: "\n".join(parse_turns(c)))
        raw = pd.to_numeric(work[m], errors="coerce")
        for i, (qid, source, resp, rb_score, n_in) in enumerate(
            zip(work["sample_id"], work["source"], responses, raw, work["n_in"])
        ):
            if source == "gsm8k" and gsm8k_gold is not None:
                score = graders.gsm8k_exact_match(resp, float(reference.loc[work.index[i]]))
                grader, version = "gsm8k_exact_match", graders.VERSION
            else:
                if pd.isna(rb_score):
                    bad.add(qid)
                    continue
                score = float(np.clip(rb_score, 0.0, 1.0))
                grader = "routerbench_gsm8k_rubric_UNVERIFIED" if source == "gsm8k" else _GRADER[source]
                version = ROUTERBENCH_GRADER_VERSION
            score_rows.append({"qid": str(qid), "model": m, "score": score, "grader": grader,
                               "grader_version": version, "mean_n_in": float(n_in),
                               "mean_n_out": float(approx_tokens(resp))})
    scores = pd.DataFrame(score_rows, columns=schema.SCORES.names)
    if bad:
        warnings.append(f"{len(bad)} items had a missing score for at least one model and were dropped.")
        scores = scores[~scores["qid"].isin(bad)]
        work = work[~work["sample_id"].astype(str).isin(bad)]

    work["group_id"] = [
        "rb-" + hashlib.sha1(f"{s}|{_norm(q)}".encode()).hexdigest()[:12]
        for s, q in zip(work["source"], work["query"])
    ]
    work["split"] = ""
    for s, idx in work.groupby("source").groups.items():
        work.loc[idx, "split"] = grouped_split(work.loc[idx, "group_id"], fractions, seed=seed)

    oracle = work["oracle_model_to_route_to"] if "oracle_model_to_route_to" in work.columns else None
    corpus = pd.DataFrame({
        "qid": work["sample_id"].astype(str).to_numpy(),
        "source": work["source"].to_numpy(),
        "subject": [e[len("mmlu-"):] if s == "mmlu" else (e if s == "mt-bench" else None)
                    for s, e in zip(work["source"], work["eval_name"])],
        "group_id": work["group_id"].to_numpy(),
        "split": work["split"].to_numpy(),
        "prompt": work["query"].to_numpy(),
        "reference": reference.loc[work.index].to_numpy(),
        "meta": [json.dumps({"eval_name": e, "rb_oracle": None if oracle is None else str(o),
                             "rb_file": source_file, "loader": LOADER_VERSION})
                 for e, o in zip(work["eval_name"], oracle if oracle is not None else [None] * len(work))],
    })
    empty = corpus["prompt"].str.len() == 0
    if empty.any():
        warnings.append(f"{int(empty.sum())} items with an empty prompt were dropped.")
        scores = scores[~scores["qid"].isin(corpus.loc[empty, "qid"])]
        corpus = corpus[~empty]

    schema.validate(corpus, schema.CORPUS)
    schema.validate(scores.reset_index(drop=True), schema.SCORES)
    assert_no_leakage(corpus)

    report["kept"] = {s: int(n) for s, n in corpus["source"].value_counts().items()}
    report["splits"] = {k: int(v) for k, v in corpus["split"].value_counts().items()}
    report["exact_duplicate_groups"] = int((corpus["group_id"].value_counts() > 1).sum())
    return LoadResult(corpus.reset_index(drop=True), scores.reset_index(drop=True), report, warnings)


def headroom_table(corpus: pd.DataFrame, scores: pd.DataFrame, large: str, small: list[str]) -> pd.DataFrame:
    """Mean score per dataset for each model, and the large-minus-small headroom for each small model."""
    wide = scores.pivot(index="qid", columns="model", values="score")
    src = corpus.set_index("qid")["source"].reindex(wide.index)
    means = wide.groupby(src).mean()
    out = pd.DataFrame({"n": src.value_counts()}).join(means)
    for m in small:
        out[f"headroom vs {m.split('/')[-1]}"] = means[large] - means[m]
    return out.sort_values("n", ascending=False)
