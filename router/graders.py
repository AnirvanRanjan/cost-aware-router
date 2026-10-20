"""Graders (DESIGN.md §3.1). Lane B owns this module; it starts with what the loader needs.

`gsm8k_exact_match` re-grades GSM8K from stored responses, because RouterBench's GSM8K scores are
a five-level rubric rather than correctness (see notes/routerbench_audit.md).

`parse_options` and `mcq_score` grade multiple-choice items by next-token option probability,
avoiding the formatting collapse ("always A") seen in small models (briefs/B.md T32).
"""
from __future__ import annotations

import math
import re

VERSION = "graders-1.2"

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


# Regex to identify candidate multiple-choice option markers like '(A)', 'A.', 'A)', or '[A]'
_OPTION_LABEL_PAT = re.compile(
    r"(?:^|[\n\s])"
    r"(?:"
    r"\(([A-Z])\)"
    r"|\[([A-Z])\]"
    r"|\b([A-Z])[\.\:\)](?=\s|$)"
    r")"
)


def parse_options(prompt: str | None) -> list[str]:
    """Extract multiple-choice option labels from prompt text (briefs/B.md T32).

    Finds the longest contiguous alphabetical sequence of option labels starting with 'A'
    (e.g., ['A', 'B'] for Winogrande, ['A', 'B', 'C', 'D'] for MMLU, or ['A', 'B', 'C', 'D', 'E']
    for 5-choice ARC questions). Does not hard-code four options.
    """
    if not prompt or not isinstance(prompt, str):
        return []
    matches: list[str] = []
    for m in _OPTION_LABEL_PAT.finditer(prompt):
        letter = m.group(1) or m.group(2) or m.group(3)
        if letter:
            matches.append(letter)

    best_seq: list[str] = []
    for i in range(len(matches)):
        if matches[i] == "A":
            curr = ["A"]
            expected_ord = ord("B")
            for j in range(i + 1, len(matches)):
                if ord(matches[j]) == expected_ord:
                    curr.append(matches[j])
                    expected_ord += 1
                elif ord(matches[j]) > expected_ord:
                    break
            if len(curr) > len(best_seq):
                best_seq = curr
    return best_seq


def mcq_score(option_logprobs: dict[str, float] | None, gold: str | None) -> float:
    """Softmax over option logprobs, returning P(gold) in [0.0, 1.0] (DESIGN.md §1.2).

    Pure Python, model-free, numerically stable.
    Returns 0.0 if gold is None, not present among options, or if option_logprobs is empty.
    """
    if not option_logprobs or gold is None:
        return 0.0
    norm_gold = str(gold).strip().upper()
    norm_lps = {str(k).strip().upper(): float(v) for k, v in option_logprobs.items()}
    if norm_gold not in norm_lps:
        return 0.0

    max_lp = max(norm_lps.values())
    if math.isinf(max_lp) and max_lp < 0:
        return 1.0 / len(norm_lps)

    exp_sum = sum(math.exp(v - max_lp) for v in norm_lps.values())
    if exp_sum <= 0.0 or math.isnan(exp_sum):
        return 0.0

    p_gold = math.exp(norm_lps[norm_gold] - max_lp) / exp_sum
    return float(max(0.0, min(1.0, p_gold)))
