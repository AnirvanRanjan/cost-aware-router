"""Unit tests for F0 heuristics, F1 TF-IDF, and F2 embeddings (DESIGN.md §1.3, briefs/B.md T10-T12)."""
import numpy as np
import pandas as pd
import pytest

from router.features import (
    HEURISTIC_NAMES,
    SentenceEmbeddingExtractor,
    TfidfFeatureExtractor,
    build_f2_features,
    extract_f1_by_split,
    heuristics,
)


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


def test_f1_tfidf_extractor_word_and_char_ngrams():
    train_prompts = [
        "Calculate the derivative of x^2 + 3x.",
        "Solve the differential equation with python code.",
        "Prove the theorem step by step.",
    ]
    extractor = TfidfFeatureExtractor(min_df=1, max_word_features=1000, max_char_features=1000)
    X = extractor.fit_transform(train_prompts)

    assert X.shape[0] == len(train_prompts)
    assert X.shape[1] > 10  # Has both word and char n-grams
    assert extractor.is_fitted is True

    # Transform new prompts
    X_new = extractor.transform(["Calculate something else."])
    assert X_new.shape[0] == 1
    assert X_new.shape[1] == X.shape[1]


def test_f1_tfidf_fitting_never_sees_cal_or_test():
    """Anti-leakage proof: briefs/B.md T11 - Done when a test proves fitting never sees cal or test."""
    corpus = pd.DataFrame({
        "qid": ["q1", "q2", "q3", "q4", "q5", "q6"],
        "split": ["train", "train", "cal", "cal", "test", "test"],
        "prompt": [
            "Normal training question about calculus and math.",
            "Another standard train question with code and algorithms.",
            "Canary calibration sentence: cal_canary_super_unique_token.",
            "Second calibration item with common math words.",
            "Canary test evaluation sentence: test_canary_super_unique_token.",
            "Second test evaluation query.",
        ],
    })

    extractor = TfidfFeatureExtractor(min_df=1)
    features_by_split, fitted_ext = extract_f1_by_split(corpus, extractor=extractor)

    # 1. Check all splits were transformed
    assert set(features_by_split.keys()) == {"train", "cal", "test"}

    # 2. Check feature dimensionalities match across all splits
    n_features = features_by_split["train"].shape[1]
    assert features_by_split["cal"].shape[1] == n_features
    assert features_by_split["test"].shape[1] == n_features
    assert features_by_split["train"].shape[0] == 2
    assert features_by_split["cal"].shape[0] == 2
    assert features_by_split["test"].shape[0] == 2

    # 3. PROVE THAT FITTING NEVER SAW CAL OR TEST:
    # The canary words must NOT appear in the fitted word vocabulary!
    word_vocab = fitted_ext.word_vec.vocabulary_
    assert "cal_canary_super_unique_token" not in word_vocab, "LEAKAGE: cal token found in fitted vocabulary!"
    assert "test_canary_super_unique_token" not in word_vocab, "LEAKAGE: test token found in fitted vocabulary!"


class FakeSentenceTransformer:
    """Mock SentenceTransformer model for fast deterministic unit testing."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.max_seq_length = 512

        class FakeTokenizer:
            def encode(self, text, add_special_tokens=True, truncation=False):
                return list(range(len(text.split())))

        self.tokenizer = FakeTokenizer()

    def encode(
        self,
        prompts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        rng = np.random.default_rng(42)
        n = len(prompts)
        vecs = rng.normal(size=(n, self.dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        return vecs


def test_sentence_embedding_extractor_with_mock():
    mock_model = FakeSentenceTransformer(dim=32)
    extractor = SentenceEmbeddingExtractor(
        max_length=5,
        model=mock_model,
    )

    prompts = [
        "Short text.",  # 2 tokens <= 5 -> not truncated
        "This is a longer prompt that exceeds the five token maximum limit.",  # 11 tokens > 5 -> truncated
    ]
    embs, n_trunc = extractor.encode(prompts)

    assert embs.shape == (2, 32)
    assert n_trunc == 1
    # Check L2 normalized
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0)


def test_build_f2_features_satisfies_schema_contract(tmp_path):
    corpus = pd.DataFrame({
        "qid": ["q1", "q2", "q3"],
        "prompt": ["First prompt", "Second prompt", "Third prompt"],
    })
    mock_model = FakeSentenceTransformer(dim=16)
    extractor = SentenceEmbeddingExtractor(model=mock_model)

    out_file = tmp_path / "features.parquet"
    df, report = build_f2_features(corpus, extractor=extractor, output_path=out_file)

    assert len(df) == 3
    assert set(df.columns) == {"qid", "feature_set", "vector"}
    assert (df["feature_set"] == "F2").all()
    assert report["dim"] == 16
    assert report["n_prompts"] == 3

    # Check that written file can be read and validated with schema
    from router import schema
    read_df = schema.read(out_file, schema.FEATURES)
    assert len(read_df) == 3
    assert len(read_df.iloc[0]["vector"]) == 16
