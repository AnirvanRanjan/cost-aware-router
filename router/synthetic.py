"""Synthetic corpus with a planted, controllable routing signal (DESIGN.md §6).

Why this exists: from week 3, the router, statistics and systems lanes build
against this data before any real scores exist. Because the signal is planted,
you know the right answer, so it doubles as a test oracle:

  * a latent difficulty d drives both models' probability of success;
  * the small model is more sensitive to d than the large one, so the expected
    gap E[s_L - s_S] grows with difficulty. That gap is what the router learns;
  * prompt text is generated so that length, digit count and cue words track d,
    which gives the F0/F1/F2 features something real to find;
  * MMLU subjects carry a subject-level difficulty offset, for LOSO experiments;
  * a few percent of queries get a near-duplicate paraphrase that shares their
    group_id, so dedup and grouped splitting can be tested.

It is NOT a model of real LLM behaviour. Never report a number computed on it.
The latent truth lives in a separate frame; it never appears in the corpus.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from router import schema
from router.splits import grouped_split

GENERATOR_VERSION = "synthetic-1.0"

SOURCE_OFFSET = {"gsm8k": 0.0, "mmlu": 0.2, "arena": 0.5}

SUBJECTS = (
    "anatomy", "astronomy", "econometrics", "formal_logic", "high_school_physics",
    "jurisprudence", "machine_learning", "moral_disputes", "nutrition",
    "philosophy", "virology", "world_religions",
)

_OBJECTS = ("apples", "tickets", "books", "bus seats", "bottles", "tiles", "coins", "boxes")
_CUES = (
    "Explain your reasoning step by step.",
    "Prove that your answer is correct.",
    "Derive the general case before answering.",
    "Justify every step rigorously.",
)
_ACADEMIC = (
    "theory", "evidence", "function", "process", "structure", "principle", "model",
    "argument", "system", "factor", "measure", "effect", "claim", "method", "variable", "context",
)
_ARENA_EASY = (
    "Give me a name for a cat.", "What is the capital of France?",
    "Say hello in Spanish.", "Suggest a quick breakfast.",
)
_ARENA_VERBS = ("Compare", "Design", "Critique", "Analyse", "Write", "Refactor", "Summarise")
_ARENA_TOPICS = (
    "a caching layer for a web API", "two approaches to climate policy",
    "a short story with an unreliable narrator", "the trade-offs of microservices",
    "a SQL query that finds duplicate orders", "the causes of the 2008 financial crisis",
)
_FILLER = (
    "State any assumptions you make.", "Keep the answer concise.", "Consider the edge cases.",
    "Cite the key trade-offs.", "Use a worked example.", "Structure it with headings.",
)
_PREFIXES = ("Quick question: ", "Please help: ", "Can you answer this? ", "Hi! ")

# (base, slope) of mean output tokens as a function of difficulty: (small model, large model)
_OUT_TOKENS = {"gsm8k": ((160, 50), (190, 40)), "mmlu": ((1, 0), (5, 0)), "arena": ((220, 90), (300, 110))}


@dataclass(frozen=True)
class SyntheticConfig:
    n: int = 2000
    seed: int = 7
    mix: dict = field(default_factory=lambda: {"gsm8k": 0.4, "mmlu": 0.4, "arena": 0.2})
    dup_rate: float = 0.05
    small_model: str = "qwen3.5-4b"
    large_model: str = "claude-sonnet-4.6"
    # success probability p_M(d) = sigmoid(a_M - b_M * d); small is steeper -> gap grows with d
    a_small: float = 0.6
    b_small: float = 1.6
    a_large: float = 2.4
    b_large: float = 1.0
    subject_spread: float = 0.6
    fractions: dict = field(default_factory=lambda: {"train": 0.60, "cal": 0.20, "test": 0.20})


@dataclass
class SyntheticData:
    corpus: pd.DataFrame  # satisfies schema.CORPUS
    scores: pd.DataFrame  # satisfies schema.SCORES
    truth: pd.DataFrame   # qid, difficulty, p_S, p_L. Tests and oracle only, never features.
    config: SyntheticConfig


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _gsm8k_prompt(rng, d):
    n_facts = int(np.clip(round(2 + 1.5 * d + rng.normal(0, 0.7)), 1, 9))
    obj = str(rng.choice(_OBJECTS))
    values = [int(rng.integers(2, 60 + int(40 * max(d, 0.0)))) for _ in range(n_facts)]
    facts = [f"Batch {i + 1} has {v} {obj}." for i, v in enumerate(values)]
    sold = int(rng.integers(1, 20))
    text = " ".join(facts) + f" Then {sold} {obj} are sold. How many {obj} are left?"
    if rng.random() < _sigmoid(1.5 * d - 0.5):
        text += " " + str(rng.choice(_CUES))
    return text, str(sum(values) - sold)


def _mmlu_prompt(rng, d, subject):
    n_words = int(np.clip(round(12 + 7 * d + rng.normal(0, 3)), 5, 45))
    stem = " ".join(rng.choice(_ACADEMIC, size=n_words))
    options = " ".join(f"{letter}) {' '.join(rng.choice(_ACADEMIC, size=3))}" for letter in "ABCD")
    text = f"[{subject.replace('_', ' ')}] Which statement about the {stem} is correct? {options}"
    return text, str(rng.choice(list("ABCD")))


def _arena_prompt(rng, d):
    if d < -0.4:
        return str(rng.choice(_ARENA_EASY)), None
    parts = [f"{rng.choice(_ARENA_VERBS)} {rng.choice(_ARENA_TOPICS)}."]
    n_extra = int(np.clip(round(1 + 1.5 * d + rng.normal(0, 0.8)), 0, 7))
    parts += [str(x) for x in rng.choice(_FILLER, size=n_extra)]
    if d > 1.0 and rng.random() < 0.5:
        parts.append("```python\ndef solve(items):\n    ...\n```")
    return " ".join(parts), None


def _paraphrase(rng, text):
    return str(rng.choice(_PREFIXES)) + text


def _soft_mcq(rng, p, concentration=8.0):
    """Soft MMLU score: P(correct option) under logprob scoring, with mean p."""
    return float(np.clip(rng.beta(p * concentration + 1e-3, (1 - p) * concentration + 1e-3), 0.0, 1.0))


def _pairwise(rng, p_small, p_large, tie_rate=0.2):
    """Small model's pairwise-judge score: 1 win, 0.5 tie, 0 loss."""
    if rng.random() < tie_rate:
        return 0.5
    return 0.0 if rng.random() < _sigmoid(4.0 * (p_large - p_small)) else 1.0


def _out_tokens(rng, source, d):
    (b_s, m_s), (b_l, m_l) = _OUT_TOKENS[source]
    noise = np.ones(2) if source == "mmlu" else rng.lognormal(0.0, 0.25, size=2)
    return float(max(1.0, (b_s + m_s * d) * noise[0])), float(max(1.0, (b_l + m_l * d) * noise[1]))


def make_synthetic(cfg: SyntheticConfig | None = None) -> SyntheticData:
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    n_dup = int(round(cfg.dup_rate * cfg.n))
    n_base = cfg.n - n_dup
    names = list(cfg.mix)
    probs = np.array([cfg.mix[s] for s in names], dtype=float)
    sources = rng.choice(names, size=n_base, p=probs / probs.sum())
    subject_offset = {s: rng.normal(0.0, cfg.subject_spread) for s in SUBJECTS}

    rows = []
    for i in range(n_base):
        src = str(sources[i])
        subject = str(rng.choice(SUBJECTS)) if src == "mmlu" else None
        d = rng.normal(SOURCE_OFFSET[src] + (subject_offset[subject] if subject else 0.0), 1.0)
        if src == "gsm8k":
            prompt, ref = _gsm8k_prompt(rng, d)
        elif src == "mmlu":
            prompt, ref = _mmlu_prompt(rng, d, subject)
        else:
            prompt, ref = _arena_prompt(rng, d)
        rows.append({"qid": f"syn-{src}-{i:05d}", "source": src, "subject": subject,
                     "group_id": f"g{i:05d}", "prompt": prompt, "reference": ref, "d": float(d)})

    for j in rng.choice(n_base, size=n_dup, replace=False):
        base = rows[int(j)]
        rows.append({**base, "qid": base["qid"] + "-dup", "prompt": _paraphrase(rng, base["prompt"])})

    d = np.array([r["d"] for r in rows])
    p_small = _sigmoid(cfg.a_small - cfg.b_small * d)
    p_large = _sigmoid(cfg.a_large - cfg.b_large * d)

    score_rows, truth_rows = [], []
    for r, ps, pl in zip(rows, p_small, p_large):
        src = r["source"]
        if src == "gsm8k":
            s_small, s_large, grader = rng.binomial(5, ps) / 5, rng.binomial(3, pl) / 3, "exact_match@k"
        elif src == "mmlu":
            s_small, s_large, grader = _soft_mcq(rng, ps), _soft_mcq(rng, pl), "mcq_logprob"
        else:
            s_small = _pairwise(rng, ps, pl)
            s_large, grader = 1.0 - s_small, "pairwise_judge"
        n_in = float(len(r["prompt"]) // 4 + 20)
        out_small, out_large = _out_tokens(rng, src, r["d"])
        for model, score, n_out in ((cfg.small_model, s_small, out_small), (cfg.large_model, s_large, out_large)):
            score_rows.append({"qid": r["qid"], "model": model, "score": float(score), "grader": grader,
                               "grader_version": GENERATOR_VERSION, "mean_n_in": n_in, "mean_n_out": n_out})
        truth_rows.append({"qid": r["qid"], "difficulty": r["d"], "p_S": float(ps), "p_L": float(pl)})

    corpus = pd.DataFrame(rows).drop(columns="d")
    corpus["meta"] = json.dumps({"generator": GENERATOR_VERSION})
    corpus["split"] = grouped_split(corpus["group_id"], cfg.fractions, seed=cfg.seed)
    corpus = corpus[schema.CORPUS.names]
    scores = pd.DataFrame(score_rows)[schema.SCORES.names]

    schema.validate(corpus, schema.CORPUS)
    schema.validate(scores, schema.SCORES)
    return SyntheticData(corpus, scores, pd.DataFrame(truth_rows), cfg)
