"""Feature extraction for the cost-aware router (DESIGN.md §1.3, briefs/B.md T10 & T11).

Features must come from the text alone. Never use `source`, `subject`, or `qid`:
in the live demo a query arrives with no label saying which benchmark it is from.
"""
from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from router.graders import parse_options

VERSION = "features-1.1"

_PUNCT_CHARS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
_SENT_SPLIT = re.compile(r"[.!?]+(?:\s+|\Z)")
_NUM_LITERAL = re.compile(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b")
_URL_PAT = re.compile(r"(?:https?://|www\.)\S+")
_LATEX_PAT = re.compile(r"(\$|\\frac|\\sqrt|\\sum|\\int|\\begin|\\alpha|\\beta|\\partial|\\times|\\div)")
_CODE_FENCE = re.compile(r"(```|~~~)")
_TABLE_ROW = re.compile(r"(?:^|\n)\s*\|.*?\|\s*(?:\n|\Z)")

_PASSAGE_CUES = re.compile(
    r"(refers to the following (?:information|passage|text)|"
    r"read the following (?:passage|text)|"
    r"based on the (?:passage|text|information)|"
    r"according to the (?:passage|text))",
    re.IGNORECASE,
)

_CUE_PROVE = re.compile(r"\b(?:prove|proof|proven)\b", re.IGNORECASE)
_CUE_DERIVE = re.compile(r"\b(?:derive|derivation|derived)\b", re.IGNORECASE)
_CUE_STEP = re.compile(r"\bstep[\s\-]by[\s\-]step\b", re.IGNORECASE)
_CUE_EXPLAIN = re.compile(r"\b(?:explain|explanation)\b", re.IGNORECASE)
_IMPERATIVE_VERBS = re.compile(
    r"\b(?:calculate|compute|solve|determine|find|identify|evaluate|describe|compare|analyse|analyze|classify|show)\b",
    re.IGNORECASE,
)

HEURISTIC_NAMES: list[str] = [
    "token_count",
    "char_count",
    "sentence_count",
    "mean_word_length",
    "digit_density",
    "punctuation_density",
    "uppercase_density",
    "question_mark_count",
    "code_fence_count",
    "latex_count",
    "url_count",
    "table_row_count",
    "num_options",
    "has_reading_passage",
    "cue_word_prove",
    "cue_word_derive",
    "cue_word_step_by_step",
    "cue_word_explain",
    "numeric_literal_count",
    "max_bracket_depth",
    "type_token_ratio",
    "flesch_kincaid_grade",
    "imperative_verb_count",
]


def _count_syllables(word: str) -> int:
    w = word.lower().strip()
    if not w:
        return 0
    clean = re.sub(r"[^a-z]", "", w)
    if not clean:
        return 0
    if len(clean) <= 3:
        return 1
    vowel_groups = len(re.findall(r"[aeiouy]+", clean))
    if clean.endswith("e") and not clean.endswith("le") and vowel_groups > 1:
        vowel_groups -= 1
    return max(1, vowel_groups)


def _max_bracket_depth(text: str) -> int:
    d = 0
    max_d = 0
    for ch in text:
        if ch in "([{":
            d += 1
            if d > max_d:
                max_d = d
        elif ch in ")]}":
            d = max(0, d - 1)
    return max_d


def _extract_single_prompt(text: str | None) -> list[float]:
    if not text or not isinstance(text, str):
        return [0.0] * len(HEURISTIC_NAMES)

    char_count = len(text)
    words = text.split()
    token_count = len(words)

    # Sentences
    raw_sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    sentence_count = max(1, len(raw_sents)) if token_count > 0 else 0

    # Word stats
    total_word_len = sum(len(w) for w in words)
    mean_word_length = total_word_len / max(token_count, 1)

    # Densities
    n_digits = sum(1 for c in text if c.isdigit())
    n_punct = sum(1 for c in text if c in _PUNCT_CHARS)
    n_upper = sum(1 for c in text if c.isupper())

    digit_density = n_digits / max(char_count, 1)
    punct_density = n_punct / max(char_count, 1)
    upper_density = n_upper / max(char_count, 1)

    question_mark_count = float(text.count("?"))
    code_fence_count = float(len(_CODE_FENCE.findall(text)))
    latex_count = float(len(_LATEX_PAT.findall(text)))
    url_count = float(len(_URL_PAT.findall(text)))
    table_row_count = float(len(_TABLE_ROW.findall(text)))

    # Multiple choice options
    opts = parse_options(text)
    num_options = float(len(opts))

    # Reading passage cue
    has_reading_passage = 1.0 if _PASSAGE_CUES.search(text) else 0.0

    # Cue words
    cue_prove = float(len(_CUE_PROVE.findall(text)))
    cue_derive = float(len(_CUE_DERIVE.findall(text)))
    cue_step = float(len(_CUE_STEP.findall(text)))
    cue_explain = float(len(_CUE_EXPLAIN.findall(text)))
    imperatives = float(len(_IMPERATIVE_VERBS.findall(text)))

    # Numeric literals & bracket depth
    num_literals = float(len(_NUM_LITERAL.findall(text)))
    max_b_depth = float(_max_bracket_depth(text))

    # Vocabulary diversity: Type-Token Ratio
    words_lower = [w.lower() for w in words]
    unique_words = len(set(words_lower))
    type_token_ratio = unique_words / max(token_count, 1)

    # Flesch-Kincaid Grade Level:
    # 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    if token_count > 0 and sentence_count > 0:
        total_syllables = sum(_count_syllables(w) for w in words)
        words_per_sent = token_count / sentence_count
        syllables_per_word = total_syllables / token_count
        fk_grade = 0.39 * words_per_sent + 11.8 * syllables_per_word - 15.59
        fk_grade = max(0.0, fk_grade)
    else:
        fk_grade = 0.0

    return [
        float(token_count),
        float(char_count),
        float(sentence_count),
        float(mean_word_length),
        float(digit_density),
        float(punct_density),
        float(upper_density),
        question_mark_count,
        code_fence_count,
        latex_count,
        url_count,
        table_row_count,
        num_options,
        has_reading_passage,
        cue_prove,
        cue_derive,
        cue_step,
        cue_explain,
        num_literals,
        max_b_depth,
        float(type_token_ratio),
        float(fk_grade),
        imperatives,
    ]


def heuristics(prompts: Sequence[str] | str) -> tuple[np.ndarray, list[str]]:
    """Extract F0 heuristic features from raw prompt text (DESIGN.md §1.3).

    Pure function operating solely on prompt text.
    Returns:
        matrix: np.ndarray of shape (N, len(HEURISTIC_NAMES))
        names: list of feature names matching columns
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    if not prompts:
        return np.empty((0, len(HEURISTIC_NAMES)), dtype=np.float32), list(HEURISTIC_NAMES)

    rows = [_extract_single_prompt(p) for p in prompts]
    mat = np.array(rows, dtype=np.float32)
    return mat, list(HEURISTIC_NAMES)


class TfidfFeatureExtractor:
    """F1 TF-IDF feature extractor (DESIGN.md §1.3, briefs/B.md T11).

    Combines word 1-2 grams with character 3-5 grams.
    STRICT ANTI-LEAKAGE RULE: Must be fitted on the `train` split only.
    """

    def __init__(
        self,
        word_ngram_range: tuple[int, int] = (1, 2),
        char_ngram_range: tuple[int, int] = (3, 5),
        max_word_features: int = 50_000,
        max_char_features: int = 50_000,
        min_df: int | float = 2,
        sublinear_tf: bool = True,
    ):
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

        self.word_vec: TfidfVectorizer | None = None
        self.char_vec: TfidfVectorizer | None = None
        self.union: FeatureUnion | None = None
        self.is_fitted: bool = False

    def fit(self, train_prompts: Sequence[str]) -> TfidfFeatureExtractor:
        """Fit the TF-IDF feature extractor ONLY on the training split prompts."""
        prompts = [str(p) if p is not None else "" for p in train_prompts]

        self.word_vec = TfidfVectorizer(
            ngram_range=self.word_ngram_range,
            max_features=self.max_word_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )
        self.char_vec = TfidfVectorizer(
            ngram_range=self.char_ngram_range,
            analyzer="char",
            max_features=self.max_char_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
        )
        self.union = FeatureUnion([
            ("word", self.word_vec),
            ("char", self.char_vec),
        ])
        self.union.fit(prompts)
        self.is_fitted = True
        return self

    def transform(self, prompts: Sequence[str] | str) -> scipy.sparse.csr_matrix:
        """Transform prompts using the fitted TF-IDF vocabulary."""
        if not self.is_fitted or self.union is None:
            raise RuntimeError("TfidfFeatureExtractor must be fitted before transforming.")
        if isinstance(prompts, str):
            prompts = [prompts]
        clean_prompts = [str(p) if p is not None else "" for p in prompts]
        return self.union.transform(clean_prompts)

    def fit_transform(self, train_prompts: Sequence[str]) -> scipy.sparse.csr_matrix:
        self.fit(train_prompts)
        return self.transform(train_prompts)


def extract_f1_by_split(
    corpus: Any,
    extractor: TfidfFeatureExtractor | None = None,
    train_split: str = "train",
) -> tuple[dict[str, scipy.sparse.csr_matrix], TfidfFeatureExtractor]:
    """Fit TF-IDF on the train split only, then transform every split (briefs/B.md T11).

    Enforces that fitting strictly never sees non-train splits ('cal', 'test').
    Returns a dictionary mapping split name to its sparse feature matrix, and the fitted extractor.
    """
    if "split" not in corpus.columns or "prompt" not in corpus.columns:
        raise ValueError("corpus dataframe must contain 'split' and 'prompt' columns.")

    train_mask = corpus["split"] == train_split
    if not train_mask.any():
        raise ValueError(f"corpus contains no rows for train split '{train_split}'.")

    if extractor is None:
        extractor = TfidfFeatureExtractor()

    train_prompts = corpus.loc[train_mask, "prompt"].tolist()
    extractor.fit(train_prompts)

    features_by_split: dict[str, scipy.sparse.csr_matrix] = {}
    for split_name in corpus["split"].unique():
        split_prompts = corpus.loc[corpus["split"] == split_name, "prompt"].tolist()
        features_by_split[split_name] = extractor.transform(split_prompts)

    return features_by_split, extractor
