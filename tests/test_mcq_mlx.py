"""Unit tests for MLXOptionScorer with a fake model and tokenizer (briefs/B.md T32)."""
import numpy as np
import pytest

from router.graders import mcq_score
from router.mcq_mlx import MLXOptionScorer


class FakeTokenizer:
    """Mock tokenizer simulating Qwen's chat template and tokenization."""

    def __init__(self):
        # Specific token IDs for clean letters and space-prefixed letters
        self.vocab = {
            "A": 100,
            " A": 101,
            "B": 200,
            " B": 201,
            "C": 300,
            " C": 301,
            "D": 400,
            " D": 401,
            "E": 500,
            " E": 501,
        }
        self.chat_template_called = False

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.chat_template_called = True
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, text, add_special_tokens=True):
        if text in self.vocab:
            return [self.vocab[text]]
        # Arbitrary token sequence for prompt text
        return [1, 2, 3, 4]


class FakeModel:
    """Mock model returning pre-determined next-token logits over vocabulary."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Default all logits to 0.0
        self.logits_vector = np.zeros(vocab_size, dtype=np.float64)

    def set_logit(self, token_id: int, val: float):
        self.logits_vector[token_id] = val

    def __call__(self, input_ids):
        # Shape: (1, seq_len, vocab_size)
        seq_len = len(input_ids[0]) if hasattr(input_ids, "__len__") else 4
        arr = np.zeros((1, seq_len, self.vocab_size), dtype=np.float64)
        arr[0, -1, :] = self.logits_vector
        return arr


def test_scorer_formats_prompt_and_reads_logits():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    # Set logits such that option B has the highest probability
    # Token 100 is "A", 101 is " A"
    model.set_logit(100, 2.0)
    model.set_logit(101, 1.5)  # max for A is 2.0

    # Token 200 is "B", 201 is " B"
    model.set_logit(200, 3.0)
    model.set_logit(201, 5.0)  # max for B is 5.0

    # Token 300 is "C", 301 is " C"
    model.set_logit(300, 1.0)
    model.set_logit(301, 1.0)  # max for C is 1.0

    # Token 400 is "D", 401 is " D"
    model.set_logit(400, 0.0)
    model.set_logit(401, 0.5)  # max for D is 0.5

    scorer = MLXOptionScorer(model=model, tokenizer=tokenizer)
    prompt = "Which bone protects the brain? A) Femur B) Cranium C) Tibia D) Fibula\nAnswer:"
    lps = scorer.option_logprobs(prompt, ["A", "B", "C", "D"])

    assert tokenizer.chat_template_called is True
    assert set(lps.keys()) == {"A", "B", "C", "D"}

    # B has highest logprob
    assert lps["B"] > lps["A"] > lps["C"] > lps["D"]

    # When B is gold, score should be dominant
    score_b = mcq_score(lps, gold="B")
    assert score_b > 0.85

    score_a = mcq_score(lps, gold="A")
    assert score_a < 0.15


def test_scorer_takes_max_over_leading_space():
    tokenizer = FakeTokenizer()
    model = FakeModel()

    # Case 1: Model prefers "A" without space (100)
    model.set_logit(100, 8.0)
    model.set_logit(101, 2.0)

    # For B, model prefers " B" with space (201)
    model.set_logit(200, 1.0)
    model.set_logit(201, 4.0)

    scorer = MLXOptionScorer(model=model, tokenizer=tokenizer)
    lps = scorer.option_logprobs("Question? A) opt1 B) opt2", ["A", "B"])

    # Max for A is logit 8.0, max for B is logit 4.0
    assert lps["A"] > lps["B"]
    score_a = mcq_score(lps, gold="A")
    assert score_a > 0.95


def test_scorer_handles_variable_labels_and_empty():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    scorer = MLXOptionScorer(model=model, tokenizer=tokenizer)

    # Empty labels returns empty dict
    assert scorer.option_logprobs("Prompt", []) == {}

    # 2 options (Winogrande)
    lps_2 = scorer.option_logprobs("Prompt", ["A", "B"])
    assert set(lps_2.keys()) == {"A", "B"}

    # 5 options (ARC)
    lps_5 = scorer.option_logprobs("Prompt", ["A", "B", "C", "D", "E"])
    assert set(lps_5.keys()) == {"A", "B", "C", "D", "E"}
