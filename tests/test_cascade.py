"""Unit tests for CascadeRouter and answer-time feature extraction (Decision D-12)."""
import numpy as np
import pytest

from router import schema
from router.cascade import (
    ANSWER_FEATURE_NAMES,
    CascadeRouter,
    arithmetic_slips,
    extract_cascade_features,
    extract_single_answer_features,
)


def test_arithmetic_slips_detection():
    # Correct equations
    text_ok = "First, 48 / 2 = 24. Then, 24 + 48 = 72. Total is 72."
    checked, wrong = arithmetic_slips(text_ok)
    assert checked == 2
    assert wrong == 0

    # Incorrect equation
    text_bad = "We multiply 12 * 50 = 700. Wait, 600 / 60 = 10."
    checked, wrong = arithmetic_slips(text_bad)
    assert checked == 2
    assert wrong == 1  # 12 * 50 = 600, not 700


def test_extract_single_answer_features_mcq():
    prompt = "Which organ pumps blood? A) Brain B) Heart C) Lungs D) Liver\nAnswer:"
    resp1 = "The correct answer is B"
    resp2 = "B) Heart"

    feats = extract_single_answer_features(
        prompt=prompt,
        response=resp1,
        source="mmlu",
        second_response=resp2,
        option_probs={"A": 0.05, "B": 0.85, "C": 0.05, "D": 0.05},
        p_chosen=0.85,
    )
    assert len(feats) == len(ANSWER_FEATURE_NAMES)

    feat_dict = dict(zip(ANSWER_FEATURE_NAMES, feats))
    # Option B selected
    assert feat_dict["letter_B"] == 1.0
    assert feat_dict["letter_A"] == 0.0
    assert feat_dict["no_answer"] == 0.0
    # Both agree on B
    assert feat_dict["agreement"] == 1.0
    assert feat_dict["p_chosen"] == pytest.approx(0.85)
    assert feat_dict["margin"] == pytest.approx(0.80)  # 0.85 - 0.05
    assert feat_dict["src_mmlu"] == 1.0


def test_extract_single_answer_features_gsm8k_k_samples():
    prompt = "Natalia sold 48 clips, then half as many. How many in total?"
    resp = "48 / 2 = 24. 48 + 24 = 72."
    k_samples = [
        "48 / 2 = 24. Total 72.",
        "She sold 72 clips.",
        "Total is 50.",  # outlier
    ]

    feats = extract_single_answer_features(
        prompt=prompt,
        response=resp,
        source="gsm8k",
        k_samples=k_samples,
    )
    feat_dict = dict(zip(ANSWER_FEATURE_NAMES, feats))
    assert feat_dict["eq_checked"] == 2.0
    assert feat_dict["eq_wrong"] == 0.0
    assert feat_dict["src_gsm8k"] == 1.0
    # 2 out of 3 samples agree on 72
    assert feat_dict["agreement"] == pytest.approx(2 / 3, abs=1e-3)


def test_extract_cascade_features_matrix():
    prompts = ["Question 1 A) x B) y", "What is 2 + 2?"]
    responses = ["Answer is A", "2 + 2 = 4"]
    sources = ["mmlu", "gsm8k"]

    X_ans = extract_cascade_features(prompts, responses, sources=sources)
    assert X_ans.shape == (2, len(ANSWER_FEATURE_NAMES))
    assert not np.isnan(X_ans).any()


def test_cascade_router_fit_and_predict():
    np.random.seed(42)
    N = 80
    D = len(ANSWER_FEATURE_NAMES)
    X = np.random.randn(N, D).astype(np.float32)
    # Target gap: s_L - s_S
    y_gap = np.random.uniform(-0.5, 0.8, size=N).astype(np.float32)

    router = CascadeRouter(max_iter=20, seed=42)
    router.fit(X, y_gap)

    pred_gap = router.predict_gap(X)
    assert pred_gap.shape == (N,)
    assert np.all(pred_gap >= -1.0) and np.all(pred_gap <= 1.0)

    s_S, s_L = router.predict_proba(X)
    assert np.all(s_S >= 0.0) and np.all(s_S <= 1.0)
    assert np.all(s_L >= 0.0) and np.all(s_L <= 1.0)
    # Check consistency: s_L - s_S == delta_hat
    assert np.allclose(s_L - s_S, pred_gap)

    # Test schema.PREDICTIONS contract
    qids = [f"q_{i}" for i in range(N)]
    pred_df = router.to_predictions_df(X, qids=qids)
    schema.validate(pred_df, schema.PREDICTIONS)
    assert len(pred_df) == N
