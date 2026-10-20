"""Track B alignment on a small hand-built corpus, with Qwen results in both input forms."""
import pandas as pd
import pytest

from router import schema
from router.trackb import build_trackb

L, S = "gpt-4-1106-preview", "qwen3.5-4b"


@pytest.fixture
def corpus():
    rows = [("g1", "gsm8k", "Natalia sold 48 clips, then half as many. Total?", "72"),
            ("g2", "gsm8k", "Weng earns 12 an hour for 50 minutes. Pay?", "10"),
            ("g3", "gsm8k", "Unanswered by Qwen so far?", "5"),
            ("m1", "mmlu", "Which bone? A) x B) y", None),
            ("m2", "mmlu", "Which organ? A) x B) y", None),
            ("w1", "winogrande", "The trophy did not fit because _ was big. A) trophy B) case", None),
            ("h1", "hellaswag", "A man is on a roof. he A) ...", None)]
    return pd.DataFrame({
        "qid": [r[0] for r in rows], "source": [r[1] for r in rows], "subject": [None] * len(rows),
        "group_id": [f"grp-{r[0]}" for r in rows], "split": ["train", "cal", "test", "train", "test", "cal", "train"],
        "prompt": [r[2] for r in rows], "reference": [r[3] for r in rows], "meta": [None] * len(rows)})


@pytest.fixture
def rb_scores(corpus):
    return pd.DataFrame([{"qid": q, "model": L, "score": 1.0, "grader": "x", "grader_version": "1",
                          "mean_n_in": 50.0, "mean_n_out": 100.0} for q in corpus["qid"]])


@pytest.fixture
def responses():
    rows = [("g1", 0, "48/2 = 24, 48 + 24 = 72 clips."), ("g1", 1, "She sold 70."), ("g1", 2, "Total = 72"),
            ("g2", 0, "12 * 50 / 60 = 10"), ("g2", 1, "She earned 10."), ("g2", 2, "She earned 10.")]
    return pd.DataFrame({"qid": [r[0] for r in rows], "model": S, "sample_idx": [r[1] for r in rows],
                         "text": [r[2] for r in rows], "n_in": 60, "n_out": 40,
                         "mean_logprob": -0.5, "latency_ms": 900.0})


@pytest.fixture
def mcq():
    return pd.DataFrame({"qid": ["m1", "m2", "w1"], "model": S, "score": [0.9, 0.2, 0.6],
                         "grader": "mcq_logprob", "grader_version": "1", "mean_n_in": 80.0, "mean_n_out": 1.0})


def test_aligns_both_input_forms(corpus, rb_scores, responses, mcq):
    tb = build_trackb(corpus, rb_scores, responses=responses, mcq_scores=mcq)
    schema.validate(tb.corpus, schema.CORPUS)
    schema.validate(tb.scores, schema.SCORES)
    assert set(tb.corpus["qid"]) == {"g1", "g2", "m1", "m2", "w1"}
    sc = tb.scores.set_index(["qid", "model"])["score"]
    assert sc[("g1", S)] == pytest.approx(2 / 3)       # 2 of 3 samples correct
    assert sc[("g2", S)] == 1.0
    assert sc[("m2", S)] == 0.2 and sc[("m1", L)] == 1.0


def test_splits_come_from_track_a_unchanged(corpus, rb_scores, responses, mcq):
    tb = build_trackb(corpus, rb_scores, responses=responses, mcq_scores=mcq)
    merged = tb.corpus.merge(corpus, on="qid", suffixes=("", "_a"))
    assert (merged["split"] == merged["split_a"]).all() and (merged["group_id"] == merged["group_id_a"]).all()


def test_partial_runs_report_coverage(corpus, rb_scores, responses):
    tb = build_trackb(corpus, rb_scores, responses=responses)       # GSM8K only, MCQ not run yet
    cov = tb.report["coverage"]
    assert cov["gsm8k"] == {"aligned": 2, "eligible": 3, "share": pytest.approx(2 / 3, abs=1e-4)}
    assert cov["mmlu"]["aligned"] == 0
    assert "mmlu" not in tb.report["headroom"]


def test_kill_check_and_out_of_scope_scores(corpus, rb_scores, mcq):
    bad = pd.concat([mcq, mcq.iloc[[0]].assign(qid="h1")])          # HellaSwag isn't in Track B
    rb = rb_scores.assign(score=0.2)                                  # GPT-4 no better than Qwen
    tb = build_trackb(corpus, rb, mcq_scores=bad)
    assert any("outside MMLU/ARC/Winogrande" in w for w in tb.warnings)
    assert any("Kill-check" in w for w in tb.warnings)
    assert "h1" not in set(tb.corpus["qid"])


def test_needs_some_input(corpus, rb_scores):
    with pytest.raises(ValueError, match="Nothing to align"):
        build_trackb(corpus, rb_scores)
