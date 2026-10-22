import math

import pytest

from router.graders import (
    extract_number,
    gsm8k_exact_match,
    gsm8k_gold,
    mcq_score,
    numeric_match,
    parse_options,
)


@pytest.mark.parametrize("text, want", [
    ("Natalia sold 72 clips altogether in April and May.", 72),
    ("48 / 2 = 24 clips in May.\n48 + 24 = 72 clips.", 72),
    ("So she earned $10.", 10),
    ("The answer is 1,234.", 1234),
    ("Total = 5 + 7 = 12.\n#### 12", 12),
    ("Answer: -3.5 degrees", -3.5),
    ("It drops by −4 points", -4),
    ("The answer is 18. (I checked 3 times.)", 18),    # explicit marker beats the last number
    # regressions from real GPT-4 / Mixtral GSM8K responses:
    ("(4/3)*360 = 480 peanuts\nSo, Darma could eat 480 peanuts in 6 minutes.", 480),
    ("16 intervals x 30 = 480 cans\nSo one machine makes 480 cans of soda in 8 hours.", 480),
    ("77 / 5 = 15.4 minutes\nRounded to the nearest minute, it will take 15 minutes.", 15),
    ("240 / 80 = 3. 6. However, he can put 4 shelves in the room.", 4),
    ("she has 2 blue and 4 yellow. 11. Therefore, Martha has 3 blue butterflies.", 3),
])
def test_extract_number(text, want):
    assert extract_number(text) == pytest.approx(want)


@pytest.mark.parametrize("text", ["", None, "no digits here"])
def test_extract_number_none(text):
    assert extract_number(text) is None


def test_gsm8k_gold_reads_the_hash_line():
    assert gsm8k_gold("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72") == 72
    assert gsm8k_gold("#### 1,000") == 1000
    assert gsm8k_gold("no answer line") is None


def test_numeric_match_and_exact_match():
    assert numeric_match(72.0, 72)
    assert not numeric_match(71.999, 72)
    assert not numeric_match(None, 72)
    assert not numeric_match(math.nan, 72)
    assert gsm8k_exact_match("so 72 clips.", 72) == 1.0
    assert gsm8k_exact_match("so 71 clips.", 72) == 0.0


def test_numbers_copied_from_the_question_do_not_override_the_result():
    q = "Marly has $300 in total. How many $100 bills can she get?"
    r = "$300 / $100 = 3\nMarly can get three $100 bills."
    assert extract_number(r, question=q) == 3
    assert extract_number("6 minutes = 360 s; (4/3)*360 = 480\nShe eats 480 peanuts in 6 minutes.",
                          question="How many peanuts in 6 minutes?") == 480


def test_parse_options_standard_formats():
    # MMLU style A–D
    p1 = "Which of the following is a bone?\nA. Femur\nB. Humerus\nC. Tibia\nD. Fibula\nAnswer:"
    assert parse_options(p1) == ["A", "B", "C", "D"]

    p2 = "Which statement?\n(A) First\n(B) Second\n(C) Third\n(D) Fourth"
    assert parse_options(p2) == ["A", "B", "C", "D"]

    p3 = "Which bone? A) x B) y C) z D) w"
    assert parse_options(p3) == ["A", "B", "C", "D"]


def test_parse_options_variable_counts():
    # Winogrande 2 options (A–B)
    w = "The trophy did not fit because _ was big. A) trophy B) case"
    assert parse_options(w) == ["A", "B"]

    # ARC 5 options (A–E)
    arc5 = "Question?\n(A) 1\n(B) 2\n(C) 3\n(D) 4\n(E) 5"
    assert parse_options(arc5) == ["A", "B", "C", "D", "E"]

    # ARC 3 options (A–C)
    arc3 = "Question?\nA: apple\nB: banana\nC: cherry"
    assert parse_options(arc3) == ["A", "B", "C"]


def test_parse_options_non_mcq_and_edge_cases():
    assert parse_options("") == []
    assert parse_options(None) == []
    assert parse_options("Natalia sold 48 clips and 24 in May. Total?") == []

    # Prompt with Vitamin A and Dr. A. Smith before the real options
    tricky = "Dr. A. Smith studied Vitamin A. Which is true?\nA. Fact 1\nB. Fact 2\nC. Fact 3\nAnswer:"
    assert parse_options(tricky) == ["A", "B", "C"]


def test_mcq_score_probabilities():
    # 4 equal options -> 0.25
    equal_lps = {"A": -1.0, "B": -1.0, "C": -1.0, "D": -1.0}
    assert mcq_score(equal_lps, "A") == pytest.approx(0.25)
    assert mcq_score(equal_lps, "C") == pytest.approx(0.25)

    # Dominant A -> ~1.0
    dom_a = {"A": 10.0, "B": -10.0, "C": -10.0, "D": -10.0}
    assert mcq_score(dom_a, "A") > 0.9999
    assert mcq_score(dom_a, "B") < 0.0001

    # Sum of probabilities across options equals 1.0
    lps = {"A": -0.5, "B": -1.2, "C": -2.0, "D": -0.1}
    total_p = sum(mcq_score(lps, opt) for opt in ["A", "B", "C", "D"])
    assert total_p == pytest.approx(1.0)


def test_mcq_score_edge_cases():
    assert mcq_score({}, "A") == 0.0
    assert mcq_score({"A": -1.0}, None) == 0.0
    assert mcq_score({"A": -1.0, "B": -1.0}, "C") == 0.0
    # Case insensitivity
    assert mcq_score({"a": 0.0, "b": -10.0}, "A") > 0.99
    assert mcq_score({"A": 0.0, "B": -10.0}, "a") > 0.99
