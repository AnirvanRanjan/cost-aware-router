import math

import pytest

from router.graders import extract_number, gsm8k_exact_match, gsm8k_gold, numeric_match


@pytest.mark.parametrize("text, want", [
    ("Natalia sold 72 clips altogether in April and May.", 72),
    ("48 / 2 = 24 clips in May.\n48 + 24 = 72 clips.", 72),
    ("So she earned $10.", 10),
    ("The answer is 1,234.", 1234),
    ("Total = 5 + 7 = 12.\n#### 12", 12),
    ("Answer: -3.5 degrees", -3.5),
    ("It drops by −4 points", -4),
    ("The answer is 18. (I checked 3 times.)", 18),    # explicit marker beats the last number
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
