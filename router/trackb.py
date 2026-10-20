"""Track B: line Qwen's answers up against RouterBench's GPT-4 scores on the same questions (T35).

Qwen3.5-4B is the headline small model and the one the demo runs (DECISIONS.md D-08). Its answers
come from lane D's runs on the M2, in two forms:

  * GSM8K: free-text responses (`responses.parquet`, k samples per question), graded here by
    numeric exact match against the gold answer already stored in the corpus;
  * MMLU, ARC, Winogrande: a SCORES frame from the `mcq_logprob` grader (lane B, T32), one row
    per question.

The frontier side reuses RouterBench's GPT-4 scores, so Track B needs no API calls. Questions,
groups and splits are taken from the Track A corpus unchanged, so both tracks are evaluated on
identical test questions. Works on partial results: whatever Qwen has answered so far is aligned,
and the report says what is still missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from router import graders, schema

TRACKB_VERSION = "trackb-1.0"
TRACKB_SOURCES = ("gsm8k", "mmlu", "arc-challenge", "winogrande")
KILL_CHECK = 0.05


@dataclass
class TrackB:
    corpus: pd.DataFrame
    scores: pd.DataFrame
    report: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def grade_gsm8k_responses(responses: pd.DataFrame, corpus: pd.DataFrame, model: str) -> pd.DataFrame:
    """SCORES rows for free-text GSM8K responses: mean exact match over the k samples per question."""
    c = corpus.set_index("qid")
    r = responses[(responses["model"] == model) & responses["qid"].isin(c.index[c["source"] == "gsm8k"])].copy()
    if r.empty:
        return pd.DataFrame(columns=schema.SCORES.names)
    gold = c.loc[r["qid"], "reference"].astype(float).to_numpy()
    question = c.loc[r["qid"], "prompt"].to_numpy()
    r["correct"] = [graders.gsm8k_exact_match(t, g, q) for t, g, q in zip(r["text"], gold, question)]
    g = r.groupby("qid").agg(score=("correct", "mean"), mean_n_in=("n_in", "mean"), mean_n_out=("n_out", "mean"))
    out = g.reset_index().assign(model=model, grader="gsm8k_exact_match", grader_version=graders.VERSION)
    return out[schema.SCORES.names]


def build_trackb(
    corpus: pd.DataFrame,
    rb_scores: pd.DataFrame,
    *,
    small: str = "qwen3.5-4b",
    large: str = "gpt-4-1106-preview",
    responses: pd.DataFrame | None = None,
    mcq_scores: pd.DataFrame | None = None,
) -> TrackB:
    """Align the small model's scores with the large model's on shared qids, one row per (qid, model)."""
    warnings: list[str] = []
    parts = []
    if responses is not None:
        schema.validate(responses, schema.RESPONSES)
        parts.append(grade_gsm8k_responses(responses, corpus, small))
    if mcq_scores is not None:
        schema.validate(mcq_scores, schema.SCORES)
        m = mcq_scores[mcq_scores["model"] == small]
        src = corpus.set_index("qid")["source"]
        wrong = m[~m["qid"].map(src).isin([s for s in TRACKB_SOURCES if s != "gsm8k"])]
        if len(wrong):
            warnings.append(f"{len(wrong)} multiple-choice scores are for questions outside MMLU/ARC/Winogrande "
                            "or not in the corpus; ignored.")
        parts.append(m.drop(index=wrong.index))
    if not parts:
        raise ValueError("Nothing to align: pass `responses` (GSM8K) and/or `mcq_scores` (MMLU, ARC, Winogrande).")

    small_scores = pd.concat(parts, ignore_index=True)
    if small_scores["qid"].duplicated().any():
        raise ValueError("More than one score per question for the small model; check the inputs.")
    large_scores = rb_scores[(rb_scores["model"] == large) & rb_scores["qid"].isin(small_scores["qid"])]
    missing_large = set(small_scores["qid"]) - set(large_scores["qid"])
    if missing_large:
        warnings.append(f"{len(missing_large)} questions have no {large} score and were dropped.")
        small_scores = small_scores[~small_scores["qid"].isin(missing_large)]

    keep = set(small_scores["qid"])
    tb_corpus = corpus[corpus["qid"].isin(keep)].reset_index(drop=True)
    tb_scores = pd.concat([small_scores, large_scores], ignore_index=True)[schema.SCORES.names]
    schema.validate(tb_corpus, schema.CORPUS)
    schema.validate(tb_scores, schema.SCORES)

    eligible = corpus[corpus["source"].isin(TRACKB_SOURCES)]
    coverage, headroom = {}, {}
    wide = tb_scores.pivot(index="qid", columns="model", values="score")
    src = tb_corpus.set_index("qid")["source"]
    for s in TRACKB_SOURCES:
        total = int((eligible["source"] == s).sum())
        done = int((tb_corpus["source"] == s).sum())
        coverage[s] = {"aligned": done, "eligible": total, "share": round(done / total, 4) if total else 0.0}
        if done:
            q = wide.loc[src[src == s].index]
            headroom[s] = {"Q_S": float(q[small].mean()), "Q_L": float(q[large].mean()),
                           "gap": float(q[large].mean() - q[small].mean())}
    failing = [s for s, h in headroom.items() if h["gap"] < KILL_CHECK]
    if failing:
        warnings.append(f"Kill-check: gap below {KILL_CHECK} on {failing}. No routing headroom there for this pair.")
    report = {
        "version": TRACKB_VERSION, "small": small, "large": large,
        "coverage": coverage, "headroom": headroom,
        "splits": {k: int(v) for k, v in tb_corpus["split"].value_counts().items()},
    }
    return TrackB(tb_corpus, tb_scores, report, warnings)
