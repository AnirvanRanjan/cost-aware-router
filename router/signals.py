"""Qwen's confidence signals, computed the same way offline (Track B) and live (demo server).

These are the features of lane B's official cascade routers (T61, `scripts/16_trackb_preview.py`,
`cascade_1sample` / `cascade_3sample`): the router sees only how sure Qwen is of its own answer,
plus which kind of question it is. Keeping one implementation for training and serving is what makes
the demo's threshold the certified one (D-15).

  GSM8K-style questions (a number as the answer): k sampled answers ->
      agree       share of the k answers equal to the most common one (1/3, 2/3, 1)
      n_distinct  number of different answers
      lp_mean     mean of the answers' mean token log-probabilities
      lp_min      lowest of them
      len         log(1 + mean answer length in tokens)
  Multiple choice: next-token log-probabilities of the option letters ->
      p_chosen    probability of the most likely option (softmax over the options)
      margin      its lead over the runner-up
Fields that do not apply are NaN (gradient-boosted trees handle missing values natively).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from router import graders

HEADLINE = ("mmlu", "gsm8k", "arc-challenge", "winogrande")
CONF_COLS = ("agree", "n_distinct", "lp_mean", "lp_min", "len", "p_chosen", "margin")
NAN = float("nan")


def sample_signals(question: str, texts: Sequence[str | None], mean_logprobs: Sequence[float | None],
                   n_outs: Sequence[int]) -> tuple[dict, float | None, int]:
    """Signals for k sampled answers to a numeric question.

    Returns (signals, the answer the user sees, index of a sample that gives it). With k > 1 the
    answer is the majority (ties -> first sample), as in `scripts/16_trackb_preview.py`.
    """
    texts = [t or "" for t in texts]
    k = len(texts)
    nums = [graders.extract_number(t, question) for t in texts]
    lps = [float(x) for x in mean_logprobs if x is not None and not math.isnan(float(x))]
    if k == 1:
        answer, agree, n_distinct = nums[0], 1.0, 1.0
    else:
        counts = Counter(round(n, 6) for n in nums if n is not None)
        top, top_n = counts.most_common(1)[0] if counts else (None, 0)
        answer = top if top_n > 1 else nums[0]
        agree, n_distinct = top_n / k, float(len(counts))
    shown = next((i for i, n in enumerate(nums)
                  if n is not None and answer is not None and round(n, 6) == round(answer, 6)), 0)
    sig = {"agree": agree, "n_distinct": n_distinct,
           "lp_mean": float(np.mean(lps)) if lps else NAN, "lp_min": float(np.min(lps)) if lps else NAN,
           "len": float(np.log1p(np.mean(n_outs))), "p_chosen": NAN, "margin": NAN}
    return sig, answer, shown


def text_agreement_signals(answers: Sequence[str], mean_logprobs: Sequence[float | None],
                           n_outs: Sequence[int]) -> tuple[dict, int]:
    """Like `sample_signals` for free-form answers: agreement of the final answers as text.

    NOT what the router was trained or certified on; the server labels these decisions uncertified.
    """
    norm = [_final_answer(a) for a in answers]
    counts = Counter(n for n in norm if n)
    top, top_n = counts.most_common(1)[0] if counts else (None, 0)
    k = len(answers)
    lps = [float(x) for x in mean_logprobs if x is not None and not math.isnan(float(x))]
    sig = {"agree": top_n / k if k else 0.0, "n_distinct": float(len(counts)),
           "lp_mean": float(np.mean(lps)) if lps else NAN, "lp_min": float(np.min(lps)) if lps else NAN,
           "len": float(np.log1p(np.mean(n_outs))), "p_chosen": NAN, "margin": NAN}
    shown = next((i for i, n in enumerate(norm) if n == top), 0) if top_n > 1 else 0
    return sig, shown


def _final_answer(text: str | None) -> str:
    t = (text or "").strip()
    low = t.lower()
    i = low.rfind("the answer is")
    tail = t[i + len("the answer is"):] if i >= 0 else (t.splitlines()[-1] if t else "")
    return " ".join("".join(ch for ch in tail.lower() if ch.isalnum() or ch.isspace()).split())


def option_signals(option_logprobs: dict[str, float]) -> tuple[dict, str]:
    """Signals from the option letters' log-probabilities, as in `gpu_trackb_vllm.mcq_record`."""
    labels = list(option_logprobs)
    lp = np.array([option_logprobs[k] for k in labels], dtype=float)
    pr = np.exp(lp - lp.max())
    pr = pr / pr.sum()
    order = np.argsort(-pr)
    margin = float(pr[order[0]] - pr[order[1]]) if len(pr) > 1 else 1.0
    sig = {"agree": NAN, "n_distinct": NAN, "lp_mean": NAN, "lp_min": NAN, "len": NAN,
           "p_chosen": float(pr[order[0]]), "margin": margin}
    return sig, labels[int(order[0])]


def feature_matrix(signals: Sequence[dict], sources: Sequence[str | None]) -> np.ndarray:
    """Rows of [CONF_COLS..., one-hot over HEADLINE] -- the column order of the T61 routers."""
    conf = np.array([[float(s.get(c, NAN)) for c in CONF_COLS] for s in signals], dtype=float)
    src = np.array([[float(src == h) for h in HEADLINE] for src in sources], dtype=float)
    return np.column_stack([conf, src])


def mcq_prompt(query: str, labels: Sequence[str]) -> str:
    """Put a typed multiple-choice question into RouterBench's MMLU format, the one Track B scored.

    "Which planet ...? A) Venus B) Mars ..." -> instruction, stem, one option per line, and the
    "Print only a single choice ... Answer:" tail, so the option probabilities mean what they meant
    in training.
    """
    import re

    parts = re.split(r"(?:^|\s)\(?([A-J])[).:]\s*", query.strip())
    stem, opts = parts[0].strip(), {}
    for i in range(1, len(parts) - 1, 2):
        opts.setdefault(parts[i], parts[i + 1].strip())
    lines = "\n".join(f"{lab}) {opts.get(lab, '')}".rstrip() for lab in labels)
    choices = " or ".join(f'"{lab}"' for lab in labels)
    return (f"Please answer with the letter of the correct answer.\n\n{stem}\n{lines}\n"
            f"Print only a single choice  from {choices} without explanation. Answer:")
