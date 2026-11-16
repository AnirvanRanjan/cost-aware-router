"""D-15: the live server computes Qwen's confidence signals exactly as the T61 research router did."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from router import threshold as th
from router.signals import CONF_COLS, mcq_prompt, option_signals, sample_signals

ROOT = Path(__file__).resolve().parents[1]


def _preview():
    spec = importlib.util.spec_from_file_location("preview", ROOT / "scripts" / "16_trackb_preview.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy():
    corpus = pd.DataFrame({"qid": ["q1", "q2", "q3"], "prompt": ["What is 5 + 7?", "What is 2 * 9?", "What is 1 + 1?"],
                           "reference": ["12", "18", "2"]})
    texts = {"q1": ["The answer is 12.", "The answer is 12.", "The answer is 13."],
             "q2": ["The answer is 18.", "The answer is 17.", "The answer is 16."],
             "q3": ["no idea", "The answer is 2.", "The answer is 2."]}
    rows = [{"qid": q, "sample_idx": j, "text": t, "mean_logprob": -0.05 * (j + 1), "n_out": 40 + 10 * j}
            for q, ts in texts.items() for j, t in enumerate(ts)]
    return corpus, pd.DataFrame(rows)


def test_sample_signals_match_the_t61_training_code():
    corpus, resp = _toy()
    pv = _preview()
    for k in (1, 3):
        want = pv.gsm8k_signals(resp, corpus, n_samples=k).set_index("qid")
        for qid, g in resp.groupby("qid"):
            g = g.sort_values("sample_idx").head(k)
            q = corpus.set_index("qid").at[qid, "prompt"]
            got, ans, _ = sample_signals(q, g["text"].tolist(), g["mean_logprob"].tolist(), g["n_out"].tolist())
            gold = float(corpus.set_index("qid").at[qid, "reference"])
            assert float(ans is not None and abs(ans - gold) < 1e-6) == want.at[qid, "s_hard"]
            for c in ("agree", "n_distinct", "lp_mean", "lp_min", "len"):
                assert np.isclose(got[c], want.at[qid, c]), (k, qid, c)


def test_option_signals_match_the_t61_training_code():
    lps = {"A": -2.3, "B": -0.1, "C": -4.0, "D": -3.1}
    pr = np.exp(np.array(list(lps.values())) - max(lps.values()))
    opts = pd.DataFrame({"qid": ["m1"], "correct": [1.0], "p_chosen": [float((pr / pr.sum()).max())],
                         "logprobs": [json.dumps(lps)]})
    want = _preview().mcq_signals(opts).iloc[0]
    got, letter = option_signals(lps)
    assert letter == "B"
    for c in CONF_COLS:
        assert np.isclose(got[c], want[c], equal_nan=True), c


def test_mcq_prompt_is_routerbench_format():
    p = mcq_prompt("Which planet is red? A) Venus B) Mars", ["A", "B"])
    assert p.startswith("Please answer with the letter") and "\nA) Venus\nB) Mars\n" in p and p.endswith("Answer:")


def test_seed_free_wsr_certifies_a_good_router_and_has_no_seed():
    rng = np.random.default_rng(0)
    n = 1500
    s_S = rng.integers(0, 2, n).astype(float)
    s_L = np.maximum(s_S, rng.integers(0, 2, n)).astype(float)
    scores = (s_L - s_S) + 0.3 * rng.normal(size=n)          # informative router
    sel = th.select_tau(scores, s_S, s_L, method="wsr_avg")
    assert sel.certified and sel.cal["pgr"] >= 0.95
    a = th.wsr_avg_pvalues(scores, s_S, s_L, [0.0, 0.5])
    assert np.array_equal(a, th.wsr_avg_pvalues(scores, s_S, s_L, [0.0, 0.5]))
