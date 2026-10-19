import numpy as np
import pandas as pd
import pytest

from router import schema
from router.schema import ContractError


def test_synthetic_frames_satisfy_their_contracts(syn):
    schema.validate(syn.corpus, schema.CORPUS)
    schema.validate(syn.scores, schema.SCORES)


def test_missing_column_is_rejected(syn):
    with pytest.raises(ContractError, match="missing columns"):
        schema.validate(syn.corpus.drop(columns="prompt"), schema.CORPUS)


def test_out_of_range_score_is_rejected(syn):
    bad = syn.scores.copy()
    bad.loc[bad.index[0], "score"] = 1.2
    with pytest.raises(ContractError, match="above 1.0"):
        schema.validate(bad, schema.SCORES)


def test_duplicate_key_is_rejected(syn):
    bad = pd.concat([syn.corpus, syn.corpus.iloc[[0]]])
    with pytest.raises(ContractError, match="duplicate"):
        schema.validate(bad, schema.CORPUS)


def test_unknown_split_label_is_rejected(syn):
    bad = syn.corpus.copy()
    bad.loc[bad.index[0], "split"] = "validation"
    with pytest.raises(ContractError, match="not in allowed"):
        schema.validate(bad, schema.CORPUS)


def test_null_in_non_nullable_column_is_rejected(syn):
    bad = syn.corpus.copy()
    bad.loc[bad.index[0], "prompt"] = None
    with pytest.raises(ContractError, match="nulls"):
        schema.validate(bad, schema.CORPUS)


def test_experimental_columns_need_the_x_prefix(syn):
    schema.validate(syn.corpus.assign(x_note="ok"), schema.CORPUS)
    with pytest.raises(ContractError, match="unexpected columns"):
        schema.validate(syn.corpus.assign(note="not ok"), schema.CORPUS)


def test_parquet_roundtrip_preserves_contract(tmp_path, syn):
    path = schema.write(syn.scores, schema.SCORES, tmp_path / "scores.parquet")
    back = schema.read(path, schema.SCORES)
    pd.testing.assert_frame_equal(back, syn.scores.reset_index(drop=True))


def test_feature_vectors_must_share_a_length():
    good = pd.DataFrame({"qid": ["a", "b"], "feature_set": ["F0", "F0"], "vector": [np.ones(3), np.zeros(3)]})
    schema.validate(good, schema.FEATURES)
    ragged = good.assign(vector=[np.ones(3), np.zeros(4)])
    with pytest.raises(ContractError, match="lengths differ"):
        schema.validate(ragged, schema.FEATURES)


def test_content_hash_is_stable_and_sensitive(syn):
    h = schema.content_hash(syn.scores)
    assert h == schema.content_hash(syn.scores.copy())
    changed = syn.scores.copy()
    changed.loc[changed.index[0], "score"] = 0.123
    assert schema.content_hash(changed) != h
