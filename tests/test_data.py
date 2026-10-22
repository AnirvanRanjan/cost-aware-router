"""The loader on a small frame laid out exactly like routerbench_0shot.pkl (checked on the real file):
list-string prompts and responses, object-dtype score columns, a few-shot preamble on GSM8K."""
import json

import pandas as pd
import pytest

from router import schema
from router.costs import PriceTable, frame_cost
from router.data import headroom_table, load_gsm8k_gold, load_routerbench, parse_turns, source_of
from router.splits import assert_no_leakage

L, S = "gpt-4-1106-preview", "mistralai/mixtral-8x7b-chat"
PREAMBLE = "TASK: Solve the following grade school math problem.\nQuestion: 2+3?\nAnswer: 5\n"


def _row(sid, ev, turns, l_resp, s_resp, l_score, s_score):
    return {"sample_id": sid, "eval_name": ev, "prompt": repr(turns),
            L: l_score, f"{L}|model_response": repr([l_resp]), f"{L}|total_cost": 1e-3,
            S: s_score, f"{S}|model_response": repr([s_resp]), f"{S}|total_cost": 1e-5,
            "oracle_model_to_route_to": S}


@pytest.fixture
def rb():
    rows = [
        _row("grade-school-math.dev.0", "grade-school-math", [PREAMBLE, "Natalia sold 48 clips, then half as many. Total?"],
             "48 + 24 = 72 clips.", "She sold 70 clips.", 0.75, 0.25),
        _row("grade-school-math.dev.1", "grade-school-math", [PREAMBLE, "Weng earns $12 an hour for 50 minutes. Pay?"],
             "She earned $10.", "She earned $10.", 0.75, 0.75),
        _row("grade-school-math.dev.2", "grade-school-math", [PREAMBLE, "A question missing from the answers file?"],
             "7", "7", 0.5, 0.5),
        _row("mmlu-anatomy.val.0", "mmlu-anatomy", ["Which bone? A) x B) y"], "A", "B", 1, 0),
        _row("mmlu-anatomy.val.1", "mmlu-anatomy", ["Which  bone? A) x B) y"], "A", "A", 1, 1),   # exact dup of .0
        _row("mmlu-virology.val.0", "mmlu-virology", ["Which virus? A) x B) y"], "B", "A", 1, 0),
        _row("hellaswag.0", "hellaswag", ["A man is on a roof. he\nA) ..."], "A)", "A)", 1, 0),
        _row("mtbench.0", "mtbench", ["Write a blog post.", "Rewrite it starting each sentence with A."],
             "Post...", "Post...", 0.9, 0.6),
        _row("chinese_zodiac.0", "chinese_zodiac", ["生肖?"], "x", "y", 1, 0),
    ]
    df = pd.DataFrame(rows)
    df[L] = df[L].astype(object)      # RouterBench stores scores as object dtype
    df[S] = df[S].astype(object)
    return df


@pytest.fixture
def gold():
    train = pd.DataFrame({
        "question": ["Natalia sold 48 clips, then half as many.  Total?", "Weng earns $12 an hour for 50 minutes. Pay?"],
        "answer": ["48/2 = 24\n#### 72", "12/60*50 = 10\n#### 10"],
    })
    return load_gsm8k_gold(train)


def test_source_mapping_and_turn_parsing():
    assert source_of("mmlu-anatomy") == "mmlu" and source_of("grade-school-math") == "gsm8k"
    assert source_of("mtbench-math") == "mt-bench" and source_of("chinese_zodiac") is None
    assert parse_turns("['a', 'b']") == ["a", "b"]
    assert parse_turns("not a list") == ["not a list"]


def test_loads_the_seven_datasets_and_drops_custom_tasks(rb, gold):
    res = load_routerbench(rb, large=L, small=[S], gsm8k_gold=gold)
    assert set(res.corpus["source"]) == {"gsm8k", "mmlu", "hellaswag", "mt-bench"}
    assert res.report["dropped_tasks"] == {"chinese_zodiac": 1}
    schema.validate(res.corpus, schema.CORPUS)
    schema.validate(res.scores, schema.SCORES)
    assert_no_leakage(res.corpus)


def test_gsm8k_is_regraded_against_gold(rb, gold):
    res = load_routerbench(rb, large=L, small=[S], gsm8k_gold=gold)
    sc = res.scores.set_index(["qid", "model"])["score"]
    assert sc[("grade-school-math.dev.0", L)] == 1.0    # 72 == 72, though RouterBench said 0.75
    assert sc[("grade-school-math.dev.0", S)] == 0.0    # 70 != 72
    assert sc[("grade-school-math.dev.1", S)] == 1.0
    assert "grade-school-math.dev.2" not in set(res.corpus["qid"])   # no gold -> dropped
    assert res.report["gsm8k"] == {"matched": 2, "unmatched_dropped": 1}
    gsm = res.scores[res.scores["qid"].str.startswith("grade")]
    assert set(gsm["grader"]) == {"gsm8k_exact_match"}


def test_without_answers_gsm8k_is_left_out_loudly(rb):
    res = load_routerbench(rb, large=L, small=[S])
    assert "gsm8k" not in set(res.corpus["source"])
    assert any("GSM8K left out" in w for w in res.warnings)
    res2 = load_routerbench(rb, large=L, small=[S], keep_unverified_gsm8k=True)
    graders = set(res2.scores.loc[res2.scores["qid"].str.startswith("grade"), "grader"])
    assert graders == {"routerbench_gsm8k_rubric_UNVERIFIED"}


def test_prompts_are_clean_text(rb, gold):
    c = load_routerbench(rb, large=L, small=[S], gsm8k_gold=gold).corpus.set_index("qid")
    assert c.loc["grade-school-math.dev.0", "prompt"].startswith("Natalia")      # preamble removed
    assert "[Turn 2]" in c.loc["mtbench.0", "prompt"]
    assert not c["prompt"].str.startswith("['").any()
    assert c.loc["mmlu-anatomy.val.0", "subject"] == "anatomy"
    assert json.loads(c.loc["mmlu-anatomy.val.0", "meta"])["eval_name"] == "mmlu-anatomy"


def test_exact_duplicates_share_a_group_and_a_split(rb, gold):
    c = load_routerbench(rb, large=L, small=[S], gsm8k_gold=gold).corpus.set_index("qid")
    a, b = c.loc["mmlu-anatomy.val.0"], c.loc["mmlu-anatomy.val.1"]
    assert a["group_id"] == b["group_id"] and a["split"] == b["split"]


def test_token_counts_and_2026_repricing(rb, gold):
    res = load_routerbench(rb, large=L, small=[S], gsm8k_gold=gold)
    assert (res.scores["mean_n_in"] > 0).all() and (res.scores["mean_n_out"] > 0).all()
    cost = frame_cost(res.scores, PriceTable.load())
    is_large = res.scores["model"] == L
    assert (cost[is_large] > 0).all() and (cost[~is_large] == 0).all()


def test_unknown_model_names_list_what_exists(rb):
    with pytest.raises(KeyError, match="Available"):
        load_routerbench(rb, large="gpt-5", small=[S])


def test_headroom_table(rb, gold):
    res = load_routerbench(rb, large=L, small=[S], gsm8k_gold=gold)
    t = headroom_table(res.corpus, res.scores, L, [S])
    assert t.loc["mmlu", "headroom vs mixtral-8x7b-chat"] == pytest.approx(2 / 3)


# ---- T07: gold letters for multiple choice -------------------------------------------------

def _mcq_frame():
    rows = [
        # qid, eval, L response, S response, L score, S score
        ("mmlu-a.val.0", "mmlu-anatomy", "A", "B", 1, 0),        # only GPT-4 right -> A
        ("mmlu-a.val.1", "mmlu-anatomy", "(C)", "C) yes", 1, 1),  # both right, agree -> C
        ("mmlu-a.val.2", "mmlu-anatomy", "D", "B", 0, 0),        # nobody right -> no gold
        ("mmlu-a.val.3", "mmlu-anatomy", "A", "B", 1, 1),        # both "right", disagree -> no gold
        ("mmlu-a.val.4", "mmlu-anatomy", "E", "A", 1, 0),        # E isn't an offered option -> no gold
        ("winogrande.dev.0", "winogrande", "B", "A", 1, 0),
        ("hellaswag.0", "hellaswag", "A)", "B)", 0, 1),
    ]
    out = []
    for i, (qid, ev, lr, sr, ls, ss) in enumerate(rows):
        opts = "A) x\nB) y" if ev == "winogrande" else "A) w\nB) x\nC) y\nD) z"
        out.append(_row(qid, ev, [f"Question {i}?\n{opts}\nAnswer:"], lr, sr, ls, ss))
    df = pd.DataFrame(out)
    df[L] = df[L].astype(object)
    df[S] = df[S].astype(object)
    return df


def test_gold_letters_are_voted_by_correct_models():
    res = load_routerbench(_mcq_frame(), large=L, small=[S])
    ref = res.corpus.set_index("qid")["reference"]
    assert ref["mmlu-a.val.0"] == "A"
    assert ref["mmlu-a.val.1"] == "C"
    assert ref["winogrande.dev.0"] == "B"
    assert ref["hellaswag.0"] == "B"
    for qid in ("mmlu-a.val.2", "mmlu-a.val.3", "mmlu-a.val.4"):
        assert pd.isna(ref[qid])


def test_gold_letter_report_and_warnings():
    res = load_routerbench(_mcq_frame(), large=L, small=[S])
    g = res.report["gold_letters"]
    assert g["mmlu"] == {"items": 5, "with_gold": 2, "share": 0.4, "conflicts": 1}
    assert g["winogrande"]["share"] == 1.0
    assert any("disagreed" in w for w in res.warnings)
    meta = json.loads(res.corpus.set_index("qid").loc["mmlu-a.val.1", "meta"])
    assert meta["gold_votes"] == 2


def test_answer_letter_parsing():
    from router.data import answer_letter, option_labels
    assert answer_letter("['A']") == "A"
    assert answer_letter("['(B) because']") == "B"
    assert answer_letter("['Comment']") is None
    assert option_labels("Q?\nA) x\nB) y\nC) z") == {"A", "B", "C"}
