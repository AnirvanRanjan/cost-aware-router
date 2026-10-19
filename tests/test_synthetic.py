"""The synthetic data is a test oracle, so these tests check that the planted signal is really there."""
import numpy as np
import pytest

from router import schema
from router.splits import assert_no_leakage
from router.synthetic import SyntheticConfig, make_synthetic


def _spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def _wide(syn):
    w = syn.scores.pivot(index="qid", columns="model", values="score")
    return w[syn.config.small_model], w[syn.config.large_model]


def test_shapes(syn):
    assert len(syn.corpus) == 1500
    assert len(syn.scores) == 2 * 1500
    assert set(syn.scores["model"]) == {syn.config.small_model, syn.config.large_model}


@pytest.mark.parametrize("source", ["gsm8k", "mmlu", "arena"])
def test_every_source_has_routing_headroom(syn, source):
    s_small, s_large = _wide(syn)
    qids = syn.corpus.loc[syn.corpus["source"] == source, "qid"]
    assert s_large[qids].mean() - s_small[qids].mean() >= 0.05


def test_quality_gap_tracks_difficulty(syn):
    s_small, s_large = _wide(syn)
    truth = syn.truth.set_index("qid").loc[s_small.index]
    assert _spearman(truth["difficulty"], s_large - s_small) > 0.2


@pytest.mark.parametrize("source", ["gsm8k", "mmlu"])
def test_prompt_length_carries_signal(syn, source):
    c = syn.corpus[syn.corpus["source"] == source].set_index("qid")
    d = syn.truth.set_index("qid").loc[c.index, "difficulty"]
    assert _spearman(c["prompt"].str.len(), d) > 0.4


def test_oracle_router_recovers_most_of_the_gap_at_half_the_calls(syn):
    s_small, s_large = _wide(syn)
    gap = (s_large - s_small).to_numpy()
    to_large = np.zeros(len(gap), dtype=bool)
    to_large[np.argsort(-gap)[: len(gap) // 2]] = True
    q = np.where(to_large, s_large, s_small).mean()
    pgr = (q - s_small.mean()) / (s_large.mean() - s_small.mean())
    assert pgr > 0.85          # a random router recovers ~0.50 at the same call rate


def test_near_duplicates_share_a_group_and_a_split(syn):
    dups = syn.corpus[syn.corpus["qid"].str.endswith("-dup")]
    assert len(dups) > 0
    sizes = syn.corpus["group_id"].value_counts()
    assert (sizes[dups["group_id"]] == 2).all()
    assert_no_leakage(syn.corpus)


def test_latent_truth_never_leaks_into_the_corpus(syn):
    assert "difficulty" not in syn.corpus.columns
    assert not syn.corpus["meta"].str.contains("difficulty").any()


def test_generation_is_deterministic():
    a = make_synthetic(SyntheticConfig(n=300, seed=1))
    b = make_synthetic(SyntheticConfig(n=300, seed=1))
    c = make_synthetic(SyntheticConfig(n=300, seed=2))
    assert schema.content_hash(a.corpus) == schema.content_hash(b.corpus)
    assert schema.content_hash(a.scores) == schema.content_hash(b.scores)
    assert schema.content_hash(a.corpus) != schema.content_hash(c.corpus)
