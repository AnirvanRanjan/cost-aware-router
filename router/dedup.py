"""Near-duplicate detection and grouping (DESIGN.md §2.5, leakage vector 1). Task T04.

Two passes, both within one dataset at a time (splits are made per dataset):

1. **Lexical, MinHash.** Word 3-gram shingles; boilerplate shingles shared by many prompts of the
   same dataset (MMLU's "Please answer with the letter…", for example) are removed first, so two
   different questions don't look alike just because of the template. MinHash + LSH proposes
   candidate pairs; every candidate is then checked exactly (Jaccard, or containment for "same text plus a
   prefix", ≥ 0.8 by default), so the result has no MinHash noise.
2. **Semantic, optional.** Cosine similarity of sentence embeddings (≥ 0.95). Run it by passing
   the F2 vectors from features.parquet once lane B has computed them (T12).

Pairs are joined into connected components; each component becomes one group_id, and the corpus is
re-split so that no group crosses train / cal / test.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from router.splits import assert_no_leakage, grouped_split

DEDUP_VERSION = "dedup-1.0"
_MERSENNE = np.uint64((1 << 61) - 1)
_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")


def shingles(text: str, k: int = 3) -> set[int]:
    """Hashed word k-grams. The same k must be used for every text, or sets can't be compared.
    3-grams rather than DESIGN's 5-grams: short prompts with a few added words still overlap, and
    the boilerplate filter keeps templates from inflating similarity. Texts shorter than k become
    one shingle (exact match only)."""
    toks = _tokens(text)
    if not toks:
        return set()
    if len(toks) < k:
        return {_hash32(" ".join(toks))}
    return {_hash32(" ".join(toks[i:i + k])) for i in range(len(toks) - k + 1)}


def drop_boilerplate(sets: list[set[int]], max_df: float = 0.02, min_count: int = 20) -> list[set[int]]:
    """Remove shingles that appear in more than `max_df` of the documents (and at least
    `min_count` of them). These are templates and instructions, not content."""
    counts: dict[int, int] = {}
    for s in sets:
        for h in s:
            counts[h] = counts.get(h, 0) + 1
    limit = max(min_count, int(max_df * len(sets)))
    common = {h for h, c in counts.items() if c > limit}
    return [s - common for s in sets]


def minhash(sets: list[set[int]], num_perm: int = 128, seed: int = 0) -> np.ndarray:
    """(n, num_perm) MinHash signatures. Empty sets get an all-max row and never match."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 31, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, size=num_perm, dtype=np.uint64)
    sig = np.full((len(sets), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, s in enumerate(sets):
        if s:
            x = np.fromiter(s, dtype=np.uint64, count=len(s))[:, None]
            sig[i] = ((x * a + b) % _MERSENNE).min(axis=0)
    return sig


def lsh_candidates(sig: np.ndarray, bands: int = 32, max_bucket: int = 200) -> set[tuple[int, int]]:
    """Index pairs that agree on every row of at least one band. Oversized buckets are skipped:
    they come from residual boilerplate, and exact Jaccard would reject them anyway."""
    n, p = sig.shape
    rows = p // bands
    empty = (sig == np.iinfo(np.uint64).max).all(axis=1)
    pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = {}
        block = sig[:, band * rows:(band + 1) * rows]
        for i in range(n):
            if not empty[i]:
                buckets.setdefault(block[i].tobytes(), []).append(i)
        for members in buckets.values():
            if 1 < len(members) <= max_bucket:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        pairs.add((members[x], members[y]))
    return pairs


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a: set, b: set, *, raw_a: set | None = None, raw_b: set | None = None,
               min_ratio: float = 0.3) -> float:
    """Jaccard, or containment when one text is the other plus a little (a prefix like "Quick
    question:"). Containment only counts when the smaller set is at least `min_ratio` of the larger,
    and, if the unfiltered sets are given, when it also holds before boilerplate removal: removing
    boilerplate can shrink two different short prompts to a few shared content words."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    small, large = sorted((len(a), len(b)))
    jac = inter / (small + large - inter)
    if small / large < min_ratio:
        return jac
    contain = inter / small
    if raw_a is not None and raw_b is not None and raw_a and raw_b:
        contain = min(contain, len(raw_a & raw_b) / min(len(raw_a), len(raw_b)))
    return max(jac, contain)


def lexical_pairs(texts: list[str], *, k: int = 3, threshold: float = 0.8, seed: int = 0) -> list[tuple[int, int, float]]:
    """(i, j, similarity) for every pair of texts at or above `threshold`, after boilerplate removal."""
    raw = [shingles(t, k) for t in texts]
    sets = drop_boilerplate(raw)
    # a text that was all boilerplate falls back to its raw shingles, so exact copies still match
    sets = [s if s else r for s, r in zip(sets, raw)]
    cands = lsh_candidates(minhash(sets, seed=seed))
    out = []
    for i, j in cands:
        sim = similarity(sets[i], sets[j], raw_a=raw[i], raw_b=raw[j])
        if sim >= threshold:
            out.append((i, j, sim))
    return sorted(out)


def embedding_pairs(vectors: np.ndarray, *, threshold: float = 0.95, block: int = 2048) -> list[tuple[int, int, float]]:
    """(i, j, cosine) for every pair of rows at or above `threshold`. Blockwise, so memory stays small."""
    v = np.asarray(vectors, dtype=np.float32)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    out = []
    for start in range(0, len(v), block):
        sims = v[start:start + block] @ v.T
        ii, jj = np.nonzero(sims >= threshold)
        for i, j in zip(ii + start, jj):
            if i < j:
                out.append((int(i), int(j), float(sims[i - start, j])))
    return sorted(out)


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class DedupResult:
    corpus: pd.DataFrame
    pairs: pd.DataFrame            # qid_a, qid_b, source, method, similarity
    report: dict = field(default_factory=dict)


def dedup_corpus(
    corpus: pd.DataFrame,
    *,
    embeddings: dict[str, np.ndarray] | None = None,
    jaccard_threshold: float = 0.8,
    cosine_threshold: float = 0.95,
    fractions: dict | None = None,
    seed: int = 7,
) -> DedupResult:
    """Merge near-duplicates into shared group_ids and re-split each dataset with no group crossing."""
    corpus = corpus.reset_index(drop=True)
    rows = []
    for source, idx in corpus.groupby("source").groups.items():
        idx = list(idx)
        texts = corpus.loc[idx, "prompt"].tolist()
        for i, j, sim in lexical_pairs(texts, threshold=jaccard_threshold, seed=seed):
            rows.append((corpus.at[idx[i], "qid"], corpus.at[idx[j], "qid"], source, "minhash", sim))
        if embeddings is not None:
            have = [q for q in corpus.loc[idx, "qid"] if q in embeddings]
            if have:
                mat = np.stack([embeddings[q] for q in have])
                for i, j, sim in embedding_pairs(mat, threshold=cosine_threshold):
                    rows.append((have[i], have[j], source, "embedding", sim))
    pairs = pd.DataFrame(rows, columns=["qid_a", "qid_b", "source", "method", "similarity"])

    uf = _UnionFind(corpus["qid"])
    for _, grp in corpus.groupby("group_id"):          # keep the loader's exact-duplicate groups
        q = grp["qid"].tolist()
        for other in q[1:]:
            uf.union(q[0], other)
    for a, b in zip(pairs["qid_a"], pairs["qid_b"]):
        uf.union(a, b)

    old_group = corpus["group_id"].copy()
    old_split = corpus["split"].copy()
    root = corpus["qid"].map(uf.find)
    first_gid = dict(zip(corpus["qid"], corpus["group_id"]))
    corpus["group_id"] = root.map(lambda r: first_gid[r])
    for source, idx in corpus.groupby("source").groups.items():
        corpus.loc[idx, "split"] = grouped_split(corpus.loc[idx, "group_id"], fractions, seed=seed)
    assert_no_leakage(corpus)

    sizes = corpus["group_id"].value_counts()
    report = {
        "version": DEDUP_VERSION,
        "thresholds": {"jaccard": jaccard_threshold, "cosine": cosine_threshold if embeddings is not None else None},
        "embedding_pass": embeddings is not None,
        "pairs_by_source_and_method": {f"{s}/{m}": int(n) for (s, m), n in pairs.groupby(["source", "method"]).size().items()},
        "groups_merged_away": int(old_group.nunique() - corpus["group_id"].nunique()),
        "rows_in_multi_item_groups": int(corpus["group_id"].map(sizes).gt(1).sum()),
        "largest_group": int(sizes.max()),
        "rows_whose_split_changed": int((old_split != corpus["split"]).sum()),
        "splits": {k: int(v) for k, v in corpus["split"].value_counts().items()},
    }
    return DedupResult(corpus, pairs, report)
