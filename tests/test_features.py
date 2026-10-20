"""Unit tests for F0 text heuristics (DESIGN.md §1.3, briefs/B.md T10)."""
import numpy as np
import pytest

from router.features import HEURISTIC_NAMES, heuristics


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def test_hand_computed_values():
    prompt = "Is this 2 + 2?"
    mat, names = heuristics(prompt)
    assert mat.shape == (1, len(HEURISTIC_NAMES))
    assert names == HEURISTIC_NAMES

    feat = dict(zip(names, mat[0]))
    assert feat["token_count"] == 5.0
    assert feat["char_count"] == 14.0
    assert feat["question_mark_count"] == 1.0
    assert feat["max_bracket_depth"] == 0.0
    assert feat["num_options"] == 0.0
    assert feat["digit_density"] == pytest.approx(2.0 / 14.0)


def test_code_fences_brackets_and_latex():
    prompt = "Solve $\\frac{a}{b}$ with code:\n```python\nx = (1 + [2])\n```"
    mat, names = heuristics([prompt])
    feat = dict(zip(names, mat[0]))

    assert feat["code_fence_count"] == 2.0
    assert feat["latex_count"] >= 1.0
    assert feat["max_bracket_depth"] == 2.0  # ( ... [ ... ] )


def test_cue_words_and_options():
    prompt = "Prove step by step. Which is right?\nA) yes\nB) no\nAnswer:"
    mat, names = heuristics([prompt])
    feat = dict(zip(names, mat[0]))

    assert feat["cue_word_prove"] == 1.0
    assert feat["cue_word_step_by_step"] == 1.0
    assert feat["num_options"] == 2.0


def test_empty_and_edge_cases():
    mat, names = heuristics([])
    assert mat.shape == (0, len(HEURISTIC_NAMES))

    mat_none, _ = heuristics(["", None])
    assert mat_none.shape == (2, len(HEURISTIC_NAMES))
    assert np.all(mat_none == 0.0)


def test_token_count_spearman_rho_on_synthetic_gsm8k(syn):
    gsm = syn.corpus[syn.corpus["source"] == "gsm8k"].set_index("qid")
    truth = syn.truth.set_index("qid").loc[gsm.index]

    mat, names = heuristics(gsm["prompt"].tolist())
    idx = names.index("token_count")
    token_counts = mat[:, idx]
    difficulty = truth["difficulty"].to_numpy(dtype=float)

    rho = _spearman(token_counts, difficulty)
    # briefs/B.md: "token count has Spearman ρ > 0.3 with truth.difficulty on synthetic GSM8K"
    assert rho > 0.3, f"Expected Spearman rho > 0.3, got {rho:.3f}"
