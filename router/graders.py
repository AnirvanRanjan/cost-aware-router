"""Graders (DESIGN.md §3.1). Lane B owns this module; it starts with what the loader needs.

`gsm8k_exact_match` re-grades GSM8K from stored responses, because RouterBench's GSM8K scores are
a five-level rubric rather than correctness (see notes/routerbench_audit.md).
"""
from __future__ import annotations

import math
import re

VERSION = "graders-1.1"

_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+"
_PRIORITY = [
    re.compile(r"####\s*\$?\s*(" + _NUM + ")"),
    re.compile(r"answer is\s*:?\s*\$?\s*(" + _NUM + ")", re.IGNORECASE),
    re.compile(r"answer\s*:\s*\$?\s*(" + _NUM + ")", re.IGNORECASE),
]
_ANY = re.compile(_NUM)


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


_EQ = re.compile(r"=\s*\$?\s*(" + _NUM + ")")
# Split into sentences on newlines and on full stops not followed by a digit, so "15.4" stays whole
# but a list marker like "11. Therefore ..." is cut off from the sentence that follows it.
_SENT = re.compile(r"\.(?!\d)|\n")
_ENUMERATOR = re.compile(r"^\s*\(?\d+\)?\s*$")


def _sentences_with_numbers(text: str) -> list[str]:
    return [s for s in _SENT.split(text) if _ANY.search(s) and not _ENUMERATOR.match(s)]


def extract_number(text: str | None, question: str | None = None) -> float | None:
    """The model's final numeric answer, or None if the text has no number.

    In order: an explicit marker ("#### 72", "The answer is 72", "Answer: 72"); otherwise the
    result of the last "= X", unless a later sentence states a different number, in which case
    the first number of that last sentence (the model's stated conclusion); otherwise the last
    number. If `question` is given, numbers copied from it ("in 6 minutes", "$100 bills") never
    override the computed result, because a closing sentence restates them. Chosen by checking GPT-4's GSM8K responses against the gold answers: "last number"
    misgrades restatements like "480 peanuts in 6 minutes", and "last = X" credits intermediate
    values the model later abandons.
    """
    if not text:
        return None
    t = str(text).replace("\u2212", "-").replace("$", "")
    for pat in _PRIORITY:
        hits = pat.findall(t)
        if hits:
            return _to_float(hits[-1])
    eqs = _EQ.findall(t)
    if eqs:
        value = _to_float(eqs[-1])
        later = _sentences_with_numbers(t[t.rfind("="):])[1:]
        if later:
            given = {_to_float(x) for x in _ANY.findall(str(question))} if question else set()
            stated = [v for v in (_to_float(x) for x in _ANY.findall(later[-1])) if v not in given]
            if stated and value not in stated:
                return stated[0]
        return value
    sents = _sentences_with_numbers(t)
    if sents:
        return _to_float(_ANY.findall(sents[-1])[-1])
    return None


def gsm8k_gold(answer_field: str) -> float | None:
    """Gold answer from a GSM8K `answer` field, which ends with '#### <number>'."""
    m = re.search(r"####\s*(" + _NUM + ")", str(answer_field))
    return _to_float(m.group(1)) if m else None


def numeric_match(pred: float | None, gold: float | None, tol: float = 1e-6) -> bool:
    if pred is None or gold is None or math.isnan(pred) or math.isnan(gold):
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


def gsm8k_exact_match(response: str | None, gold: float | None, question: str | None = None) -> float:
    """1.0 if the response's final number equals the gold answer, else 0.0."""
    return 1.0 if numeric_match(extract_number(response, question), gold) else 0.0
