"""Graders (DESIGN.md §3.1). Lane B owns this module; it starts with what the loader needs.

`gsm8k_exact_match` re-grades GSM8K from stored responses, because RouterBench's GSM8K scores are
a five-level rubric rather than correctness (see notes/routerbench_audit.md).
"""
from __future__ import annotations

import math
import re

VERSION = "graders-1.0"

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


def extract_number(text: str | None) -> float | None:
    """The model's final numeric answer: an explicit marker ("#### 72", "The answer is 72",
    "Answer: 72") if present, otherwise the last number in the text. None if there is no number."""
    if not text:
        return None
    t = str(text).replace("−", "-").replace("$", "")
    for pat in _PRIORITY:
        hits = pat.findall(t)
        if hits:
            return _to_float(hits[-1])
    hits = _ANY.findall(t)
    return _to_float(hits[-1]) if hits else None


def gsm8k_gold(answer_field: str) -> float | None:
    """Gold answer from a GSM8K `answer` field, which ends with '#### <number>'."""
    m = re.search(r"####\s*(" + _NUM + ")", str(answer_field))
    return _to_float(m.group(1)) if m else None


def numeric_match(pred: float | None, gold: float | None, tol: float = 1e-6) -> bool:
    if pred is None or gold is None or math.isnan(pred) or math.isnan(gold):
        return False
    return abs(pred - gold) <= tol * max(1.0, abs(gold))


def gsm8k_exact_match(response: str | None, gold: float | None) -> float:
    """1.0 if the response's final number equals the gold answer, else 0.0."""
    return 1.0 if numeric_match(extract_number(response), gold) else 0.0
