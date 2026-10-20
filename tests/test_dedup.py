import numpy as np
import pandas as pd
import pytest

from router.dedup import (dedup_corpus, drop_boilerplate, embedding_pairs, jaccard, lexical_pairs,
                          shingles, similarity)
from router.splits import assert_no_leakage
from router.synthetic import SyntheticConfig, make_synthetic

TEMPLATE = ("Please answer with the letter of the correct answer. {q} A) one B) two C) three D) four "
            "Print only a single choice from A or B or C or D without explanation. Answer:")


def test_shingles_and_jaccard():
    a, b = shingles("the quick brown fox jumps over the lazy dog"), shingles("the quick brown fox jumps over the lazy cat")
    assert 0.5 < jaccard(a, b) < 1.0
    assert shingles("tiny text") == shingles("Tiny, text!")       # shorter than k: one exact-match shingle
    assert jaccard(set(), a) == 0.0


def test_prefixed_short_prompt_counts_as_duplicate():
    a, b = shingles("Say hello in Spanish."), shingles("Please help: Say hello in Spanish.")
    assert similarity(a, b) == 1.0                         # containment
    long = shingles("Say hello in Spanish and then explain the long history of the Spanish language in Latin America and in Europe, in detail, with examples please")
    assert similarity(a, long) < 0.8                       # too different in size to count


def test_boilerplate_is_ignored():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(500)]
    texts = [TEMPLATE.format(q=" ".join(rng.choice(vocab, 12))) for _ in range(300)]
    # different questions in the same long template must not look like duplicates
    assert lexical_pairs(texts) == []
    # but a real near-copy inside the template still does
    texts.append(TEMPLATE.format(q=texts[0].split("answer. ")[1].split(" A)")[0] + " please"))
    hits = lexical_pairs(texts)
    assert [(i, j) for i, j, _ in hits] == [(0, 300)]
    assert drop_boilerplate([shingles(t) for t in texts[:50]], min_count=5)[0] != shingles(texts[0])


def test_embedding_pairs():
    v = np.array([[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert [(i, j) for i, j, _ in embedding_pairs(v, threshold=0.95)] == [(0, 1)]


@pytest.fixture(scope="module")
def syn_ungrouped():
    syn = make_synthetic(SyntheticConfig(n=2000, seed=3))
    c = syn.corpus.copy()
    truth = c.groupby("group_id")["qid"].apply(list)
    planted = [tuple(sorted(q)) for q in truth if len(q) == 2]
    c["group_id"] = c["qid"]                              # forget the planted grouping
    return c, planted


def test_recovers_planted_near_duplicates(syn_ungrouped):
    c, planted = syn_ungrouped
    res = dedup_corpus(c)
    gid = res.corpus.set_index("qid")["group_id"]
    found = sum(gid[a] == gid[b] for a, b in planted)
    assert found / len(planted) >= 0.9                    # KICKOFF.md T04 "done when"
    assert_no_leakage(res.corpus)


def test_keeps_existing_groups_and_reports(syn_ungrouped):
    c, _ = syn_ungrouped
    c = c.copy()
    c.loc[c.index[:2], "group_id"] = "manual-group"       # an exact-duplicate group from the loader
    res = dedup_corpus(c)
    g = res.corpus.set_index("qid").loc[c["qid"][:2], "group_id"]
    assert g.nunique() == 1
    for key in ("pairs_by_source_and_method", "groups_merged_away", "rows_whose_split_changed", "splits"):
        assert key in res.report


def test_embedding_pass_merges_paraphrases():
    c = pd.DataFrame({
        "qid": ["a", "b", "c"], "source": ["mmlu"] * 3, "subject": [None] * 3,
        "group_id": ["a", "b", "c"], "split": ["train", "cal", "test"],
        "prompt": ["What is the capital of France?", "Which city is France's capital?", "Name a prime number."],
        "reference": [None] * 3, "meta": [None] * 3,
    })
    emb = {"a": np.array([1.0, 0.0]), "b": np.array([0.99, 0.05]), "c": np.array([0.0, 1.0])}
    res = dedup_corpus(c, embeddings=emb)
    g = res.corpus.set_index("qid")["group_id"]
    assert g["a"] == g["b"] != g["c"]
    assert res.report["embedding_pass"] is True


def test_short_prompts_sharing_only_content_words_are_not_duplicates():
    # real false positive from MBPP before the raw-containment check
    texts = ["Write a function to find the length of the longest palindromic subsequence in the given string.",
             "Write a python function to find the length of the longest word."]
    texts += [f"Write a function to find the {w} of the given list." for w in
              ("sum", "max", "min", "mean", "median", "mode", "range", "product", "count", "sorted copy") * 3]
    assert (0, 1) not in [(i, j) for i, j, _ in lexical_pairs(texts)]
