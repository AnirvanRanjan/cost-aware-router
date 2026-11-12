"""Cascade Router (DESIGN.md §1.2, Decision D-12).

Routes queries after observing the small model S's generated answer (and confidence/consensus signals),
deciding whether to accept S's answer or escalate to frontier model L.

Under cost model M1 (local model S is free on local hardware), running S first adds zero monetary cost,
only paying S's execution latency on escalated queries.

Features observed:
  1. Question features (F0 heuristics, F2 embeddings).
  2. Answer-time features: length, steps, hedge words, arithmetic slips in GSM8K working.
  3. Confidence & consensus signals:
     - Agreement across k samples (self-consistency) or agreement with a 2nd local model (e.g. Yi-34B).
     - Multiple-choice option probability / margin / entropy from next-token log-probs.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from router import schema
from router.data import answer_letter, parse_turns
from router.graders import extract_number

_EQ = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([+\-*/x×÷])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)")
_HEDGE = re.compile(r"\b(?:assume|assuming|approximately|not sure|unclear|cannot|can't|however|wait|actually)\b", re.I)

HEADLINE_SOURCES = ("mmlu", "gsm8k", "arc-challenge", "winogrande")

ANSWER_FEATURE_NAMES = [
    "len",
    "no_answer",
    "eq_checked",
    "eq_wrong",
    "eq_wrong_any",
    "steps",
    "answer_in_question",
    "non_integer",
    "hedge",
    "letter_A",
    "letter_B",
    "letter_C",
    "letter_D",
    "letter_E",
    "src_mmlu",
    "src_gsm8k",
    "src_arc_challenge",
    "src_winogrande",
    "agreement",
    "p_chosen",
    "margin",
    "entropy",
]


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def arithmetic_slips(text: str) -> tuple[int, int]:
    """Check mathematical expressions 'a op b = c' in text and return (checked, wrong)."""
    checked = wrong = 0
    for a, op, b, c in _EQ.findall(text):
        a_n, b_n, c_n = _num(a), _num(b), _num(c)
        if None in (a_n, b_n, c_n):
            continue
        if op in "*x×":
            v = a_n * b_n
        elif op in "/÷":
            if b_n == 0:
                continue
            v = a_n / b_n
        elif op == "+":
            v = a_n + b_n
        else:
            v = a_n - b_n
        checked += 1
        wrong += int(abs(v - c_n) > max(1e-6, 0.01 * abs(v)))
    return checked, wrong


def extract_letter_flexible(response: Any) -> str | None:
    """Extract MCQ option letter, checking leading format first, then common answer patterns."""
    l = answer_letter(response)
    if l:
        return l
    turns = parse_turns(response)
    if not turns:
        return None
    text = " ".join(turns).strip()
    m = re.search(r"(?:answer is|answer:|\bchoice\b|\boption\b)\s*\(?([A-E])\b", text, re.I)
    if m:
        return m.group(1).upper()
    return None


def extract_single_answer_features(
    prompt: str,
    response: Any,
    source: str | None = None,
    second_response: Any = None,
    k_samples: Sequence[str] | None = None,
    option_probs: dict[str, float] | None = None,
    p_chosen: float | None = None,
    mean_logprob: float | None = None,
) -> list[float]:
    """Extract answer-time features for a single query and response."""
    if hasattr(response, "text"):
        resp_text = response.text
        if mean_logprob is None and getattr(response, "mean_logprob", None) is not None:
            mean_logprob = response.mean_logprob
    elif isinstance(response, dict):
        resp_text = response.get("text", "")
        if mean_logprob is None and "mean_logprob" in response:
            mean_logprob = response.get("mean_logprob")
    else:
        resp_text = response

    text = " ".join(parse_turns(resp_text)) if resp_text is not None else ""
    text2 = " ".join(parse_turns(second_response)) if second_response is not None else ""

    length_feat = float(np.log1p(len(text)))
    hedge_count = float(len(_HEDGE.findall(text)))

    # Detect task type: math / numeric vs multiple choice
    is_gsm8k = source == "gsm8k" or (source is None and re.search(r"\b(?:calculate|compute|\d+\s*[+\-*/=])\b", prompt, re.I) and not re.search(r"[A-D]\)", prompt))

    letters = [0.0] * 5
    agreement_val = 0.5  # Neutral default

    if is_gsm8k:
        n1 = extract_number(text, prompt)
        n2 = extract_number(text2, prompt) if text2 else None
        checked, wrong = arithmetic_slips(text)
        qnums = {_num(x) for x in re.findall(r"\d[\d,]*\.?\d*", prompt)}

        no_answer = float(n1 is None)
        eq_checked = float(checked)
        eq_wrong = float(wrong)
        eq_wrong_any = float(wrong > 0)
        steps = float(text.count("\n"))
        ans_in_q = float(n1 in qnums) if n1 is not None else 0.0
        non_int = float(n1 is not None and abs(n1 - round(n1)) > 1e-9)

        if k_samples:
            parsed_k = [extract_number(s, prompt) for s in k_samples]
            valid_k = [k for k in parsed_k if k is not None]
            if valid_k:
                # Share of samples matching the first sample / modal answer
                modal_count = max(valid_k.count(x) for x in set(valid_k))
                agreement_val = float(modal_count / len(valid_k))
            else:
                agreement_val = 0.0
        elif text2:
            agreement_val = float(n1 is not None and n2 is not None and abs(n1 - n2) < 1e-6)
    else:
        l1 = extract_letter_flexible(response)
        l2 = extract_letter_flexible(second_response) if second_response is not None else None
        letters = [float(l1 == x) for x in "ABCDE"]
        no_answer = float(l1 is None)
        eq_checked = 0.0
        eq_wrong = 0.0
        eq_wrong_any = 0.0
        steps = float(text.count("\n"))
        ans_in_q = 0.0
        non_int = 0.0

        if k_samples:
            parsed_k = [extract_letter_flexible(s) for s in k_samples]
            valid_k = [k for k in parsed_k if k is not None]
            if valid_k:
                modal_count = max(valid_k.count(x) for x in set(valid_k))
                agreement_val = float(modal_count / len(valid_k))
            else:
                agreement_val = 0.0
        elif second_response is not None:
            agreement_val = float(l1 is not None and l1 == l2)

    # Source dataset one-hot
    src_onehot = [float(source == s) for s in HEADLINE_SOURCES]

    # Confidence signal: MCQ probability, explicit p_chosen, or mean token logprob
    has_p = p_chosen is not None and not np.isnan(p_chosen)
    has_mlp = mean_logprob is not None and not np.isnan(mean_logprob)

    if has_p:
        p_c = float(p_chosen)
    elif option_probs:
        p_c = float(max(option_probs.values()))
    elif has_mlp:
        p_c = float(np.clip(np.exp(float(mean_logprob)), 0.0, 1.0))
    else:
        p_c = 0.5  # Neutral default

    if option_probs and len(option_probs) >= 2:
        sorted_probs = sorted(option_probs.values(), reverse=True)
        margin_val = float(sorted_probs[0] - sorted_probs[1])
        probs_arr = np.array(sorted_probs, dtype=float)
        probs_arr = np.clip(probs_arr, 1e-12, 1.0)
        probs_arr = probs_arr / probs_arr.sum()
        entropy_val = float(-np.sum(probs_arr * np.log(probs_arr)))
    elif has_mlp:
        margin_val = float(mean_logprob)
        entropy_val = float(-mean_logprob)
    else:
        margin_val = float(p_c - (1.0 - p_c) / 3.0) if p_c > 0.25 else 0.0
        entropy_val = 1.0

    return [
        length_feat,
        no_answer,
        eq_checked,
        eq_wrong,
        eq_wrong_any,
        steps,
        ans_in_q,
        non_int,
        hedge_count,
        *letters,
        *src_onehot,
        agreement_val,
        p_c,
        margin_val,
        entropy_val,
    ]


def extract_cascade_features(
    prompts: Sequence[str],
    responses: Sequence[Any],
    sources: Sequence[str] | None = None,
    second_responses: Sequence[Any] | None = None,
    k_samples_list: Sequence[Sequence[str]] | None = None,
    option_probs_list: Sequence[dict[str, float]] | None = None,
    p_chosen_list: Sequence[float] | None = None,
    mean_logprobs_list: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Extract matrix of answer-time and confidence features across multiple queries."""
    n = len(prompts)
    rows = []
    for i in range(n):
        prompt = prompts[i]
        resp = responses[i]
        src = sources[i] if sources is not None else None
        resp2 = second_responses[i] if second_responses is not None else None
        k_samps = k_samples_list[i] if k_samples_list is not None else None
        opt_probs = option_probs_list[i] if option_probs_list is not None else None
        p_ch = p_chosen_list[i] if p_chosen_list is not None else None
        mlp = mean_logprobs_list[i] if mean_logprobs_list is not None else None

        feats = extract_single_answer_features(
            prompt=prompt,
            response=resp,
            source=src,
            second_response=resp2,
            k_samples=k_samps,
            option_probs=opt_probs,
            p_chosen=p_ch,
            mean_logprob=mlp,
        )
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


class CascadeRouter:
    """Cascade router model (Decision D-12).

    Predicts expected capability delta delta_hat = s_L - s_S using
    question features combined with S's answer features and confidence.
    """

    def __init__(
        self,
        name: str = "cascade_router",
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 1.0,
        seed: int = 42,
    ):
        self.name = name
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.l2_regularization = l2_regularization
        self.seed = seed
        self.model: HistGradientBoostingRegressor | None = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y_gap: np.ndarray) -> CascadeRouter:
        """Fit gradient boosted trees on capability gap y_gap = s_L - s_S."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y_gap, dtype=np.float32)
        self.model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            random_state=self.seed,
        )
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict_gap(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("CascadeRouter must be fitted before predicting.")
        preds = self.model.predict(np.asarray(X, dtype=np.float32))
        return np.clip(np.asarray(preds, dtype=float), -1.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return marginal estimates s_hat_S, s_hat_L consistent with predicted delta_hat."""
        delta = self.predict_gap(X)
        s_hat_S = np.clip(0.5 - delta / 2.0, 0.0, 1.0)
        s_hat_L = np.clip(0.5 + delta / 2.0, 0.0, 1.0)
        return s_hat_S, s_hat_L

    def to_predictions_df(
        self,
        X: np.ndarray,
        qids: Sequence[str],
        c_hat_L: np.ndarray | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Produce schema.PREDICTIONS contract DataFrame."""
        delta_hat = self.predict_gap(X)
        s_hat_S, s_hat_L = self.predict_proba(X)
        n = len(qids)
        df = pd.DataFrame({
            "qid": [str(q) for q in qids],
            "router": [self.name] * n,
            "seed": [int(seed if seed is not None else self.seed)] * n,
            "s_hat_S": s_hat_S,
            "s_hat_L": s_hat_L,
            "delta_hat": delta_hat,
            "c_hat_L": np.asarray(c_hat_L if c_hat_L is not None else np.zeros(n), dtype=float),
        })
        schema.validate(df, schema.PREDICTIONS)
        return df
