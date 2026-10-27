#!/usr/bin/env python3
"""EXPLORATORY: does the small model's own answer predict its failures better than the question does?

    python scripts/12_cascade_explore.py          # Mixtral vs GPT-4, frozen splits, headline sources

A query-only router decides before the small model S runs. A cascade lets S answer first, then
decides whether to keep that answer or escalate to L. Under cost model M1 that costs nothing extra
(S is local and free) but adds S's latency to every escalated query. This script measures how much
routing gets better when the router may look at S's answer, using RouterBench's stored Mixtral
responses (one sample per question, no log-probabilities):

  answer   features of S's answer only: chosen option letter, answer length, unparseable answer,
           and for GSM8K the reasoning itself (length, steps, arithmetic slips found by re-checking
           every "a op b = c" in the text, final number missing or copied from the question).
  answer+2nd   the above plus agreement with a second local model (Yi-34B): same letter / same
           number. A stand-in for Qwen's self-consistency (k samples) or its option probability,
           which RouterBench does not store but our own runs will.
  query+answer+2nd   the above plus the 23 query heuristics (F0).

Everything is fitted on TRAIN (gradient-boosted trees on the gap s_L - s_S), tau* is picked on CAL
with the WSR fixed-sequence rule, and test is touched once, through scripts/08_threshold_report.py.
Writes artifacts/predictions/cascade_explore.parquet (PREDICTIONS contract, all splits).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router import schema  # noqa: E402
from router.data import answer_letter, parse_turns  # noqa: E402
from router.features import heuristics  # noqa: E402
from router.graders import extract_number  # noqa: E402

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")
_EQ = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([+\-*/x×÷])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)")
_HEDGE = re.compile(r"\b(?:assume|assuming|approximately|not sure|unclear|cannot|can't|however|wait|actually)\b", re.I)


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def arithmetic_slips(text: str) -> tuple[int, int]:
    """(checked, wrong): every 'a op b = c' in the text, re-computed."""
    checked = wrong = 0
    for a, op, b, c in _EQ.findall(text):
        a, b, c = _num(a), _num(b), _num(c)
        if None in (a, b, c):
            continue
        if op in "*x×":
            v = a * b
        elif op in "/÷":
            if b == 0:
                continue
            v = a / b
        elif op == "+":
            v = a + b
        else:
            v = a - b
        checked += 1
        wrong += abs(v - c) > max(1e-6, 0.01 * abs(v))
    return checked, wrong


def answer_features(source: str, question: str, resp, resp2) -> list[float]:
    text = " ".join(parse_turns(resp)) if resp is not None else ""
    text2 = " ".join(parse_turns(resp2)) if resp2 is not None else ""
    f = {"len": np.log1p(len(text))}
    if source == "gsm8k":
        n1, n2 = extract_number(text, question), extract_number(text2, question)
        checked, wrong = arithmetic_slips(text)
        qnums = {_num(x) for x in re.findall(r"\d[\d,]*\.?\d*", question)}
        f.update({"no_answer": float(n1 is None), "eq_checked": checked, "eq_wrong": wrong,
                  "eq_wrong_any": float(wrong > 0), "steps": text.count("\n"),
                  "answer_in_question": float(n1 in qnums) if n1 is not None else 0.0,
                  "non_integer": float(n1 is not None and abs(n1 - round(n1)) > 1e-9),
                  "hedge": len(_HEDGE.findall(text)),
                  "agree2": float(n1 is not None and n2 is not None and abs(n1 - n2) < 1e-6)})
        letters = [0.0] * 5
    else:
        l1, l2 = answer_letter(resp), answer_letter(resp2)
        letters = [float(l1 == x) for x in "ABCDE"]
        f.update({"no_answer": float(l1 is None), "eq_checked": 0, "eq_wrong": 0, "eq_wrong_any": 0.0,
                  "steps": 0, "answer_in_question": 0.0, "non_integer": 0.0, "hedge": 0,
                  "agree2": float(l1 is not None and l1 == l2)})
    src = [float(source == s) for s in HEADLINE]   # dataset identity is visible from the prompt format
    return [f[k] for k in ("len", "no_answer", "eq_checked", "eq_wrong", "eq_wrong_any", "steps",
                           "answer_in_question", "non_integer", "hedge")] + letters + src, f["agree2"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("artifacts/raw/routerbench/routerbench_0shot.pkl"))
    ap.add_argument("--corpus", type=Path, default=Path("artifacts/routerbench/corpus.parquet"))
    ap.add_argument("--scores", type=Path, default=Path("artifacts/routerbench/scores.parquet"))
    ap.add_argument("--small", default="mistralai/mixtral-8x7b-chat")
    ap.add_argument("--second", default="zero-one-ai/Yi-34B-Chat")
    ap.add_argument("--large", default="gpt-4-1106-preview")
    ap.add_argument("--out", type=Path, default=Path("artifacts/predictions/cascade_explore.parquet"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    from router.costs import PriceTable
    from router.data import approx_tokens

    raw = pd.read_pickle(args.raw)
    raw["qid"] = raw["sample_id"].astype(str)
    raw = raw.drop_duplicates("qid").set_index("qid")
    corpus = schema.read(args.corpus, schema.CORPUS)
    corpus = corpus[corpus["source"].isin(HEADLINE)].set_index("qid")
    wide = schema.read(args.scores, schema.SCORES).pivot(index="qid", columns="model", values="score")
    qids = corpus.index[corpus.index.isin(wide.index) & corpus.index.isin(raw.index)]
    c, w, r = corpus.loc[qids], wide.loc[qids], raw.loc[qids]
    print(f"{len(qids):,} headline questions with stored responses")

    rows, agree = [], []
    for q, src, prompt in zip(qids, c["source"], c["prompt"]):
        feats, ag = answer_features(src, prompt, r.at[q, f"{args.small}|model_response"],
                                    r.at[q, f"{args.second}|model_response"])
        rows.append(feats)
        agree.append(ag)
    A = np.asarray(rows, dtype=float)
    A2 = np.column_stack([A, agree])
    X0, _ = heuristics(c["prompt"].tolist())
    QA2 = np.column_stack([np.nan_to_num(np.asarray(X0, dtype=float)), A2])

    s_S, s_L = w[args.small].to_numpy(float), w[args.large].to_numpy(float)
    gap = s_L - s_S
    tr = (c["split"] == "train").to_numpy()
    prices = PriceTable.load()
    n_in = np.array([approx_tokens(p) for p in c["prompt"]], dtype=float)
    c_hat_L = np.asarray(prices.query_cost(args.large, n_in, np.full(len(c), 43.5)), dtype=float)

    frames, info = [], {"n": int(len(qids)), "interim": True, "exploratory": True,
                        "pair": {"small": args.small, "large": args.large, "second_opinion": args.second}}
    for name, X in (("cascade_answer", A), ("cascade_answer+2nd", A2), ("cascade_query+answer+2nd", QA2)):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=args.seed).fit(X[tr], gap[tr])
        d = np.clip(m.predict(X), -1, 1)
        frames.append(pd.DataFrame({"qid": qids.astype(str), "router": name, "seed": args.seed,
                                    "s_hat_S": np.clip(0.5 - d / 2, 0, 1), "s_hat_L": np.clip(0.5 + d / 2, 0, 1),
                                    "delta_hat": d, "c_hat_L": c_hat_L}))
    # how informative is agreement on its own?
    te = (c["split"] == "test").to_numpy()
    ag = np.asarray(agree, dtype=bool)
    info["test_S_accuracy_when_agree"] = float(s_S[te & ag].mean())
    info["test_S_accuracy_when_disagree"] = float(s_S[te & ~ag].mean())
    info["test_share_agree"] = float(ag[te].mean())
    preds = pd.concat(frames, ignore_index=True)
    schema.write(preds, schema.PREDICTIONS, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
