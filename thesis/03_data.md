# Chapter 3 — Data: Corpus, Labels and Splits

> **DRAFT for Anirvan Ranjan (lane A, task T50).** Generated from `notes/chapter_A_facts.md` and the frozen
> release `data-910a584c8279baa7`. Every number here can be reproduced with the scripts named beside it.
> Rewrite it in your own voice and check UPES's rules on declaring AI assistance before submission.

## 3.1 Why the data decides the result

A router is a classifier trained to predict where a small model fails. It learns only what its labels
tell it, so three properties of the data bound everything later chapters can claim:

1. **Labels must measure correctness.** If a benchmark's score reflects a formatting habit or a broken
   grader, the router learns the habit, not the difficulty of the question.
2. **The small model must be capable of the task at all.** A model that answers at chance level gives
   the router nothing to separate: every question looks equally hard.
3. **No near-copy of a test question may appear in training.** Benchmarks contain rewrites, shared
   passages and twin items. If one half of a pair is in training and the other in test, the router's
   test score measures memory, not generalisation.

This chapter describes how the corpus was built to meet these three conditions. Section 3.2 introduces
the source, 3.3 reports an audit of it, 3.4 and 3.5 repair the labels, 3.6 removes near-duplicates
and freezes the splits, and 3.7 explains how the model pairs were chosen. Section 3.8 lists what the
data still cannot support.

## 3.2 Source: RouterBench

We build on RouterBench [@hu2024routerbench], a public collection of pre-computed answers from eleven
language models to the same set of prompts, released with per-answer quality scores and costs. Its
value to a student project is that the expensive part of a routing study — getting a frontier model's
answer to tens of thousands of questions — has already been paid for: we use RouterBench's stored
GPT-4 answers as the frontier arm instead of buying them (Rev A of the design).

We use the zero-shot file `routerbench_0shot.pkl` (36,497 rows × 37 columns). Each row holds a prompt,
its dataset label, and for each of the eleven models its response, score and cost. Two storage details
matter for every later step: prompts and responses are stored as stringified Python lists and are
parsed with `ast.literal_eval` before any feature sees them; and the dataset label (`eval_name`) is
fine-grained (one label per MMLU subject, for example), so it is mapped to a source dataset by
`router.data.source_of`.

We keep seven datasets and drop 21 small custom tasks (1,719 rows): Chinese-language tasks,
accounting_audit, bias_detection, consensus_summary, abstract2title and test-match. They are not part of
the RouterBench paper's evaluation, most have fewer than 400 items, and several have negative headroom.
Table 3.1 shows the resulting corpus.

**Table 3.1 — The corpus after loading.**

| Dataset | Questions | Task type |
|---|--:|---|
| MMLU | 14,042 | 4-option multiple choice, 57 subjects |
| HellaSwag | 10,042 | 4-option sentence completion |
| GSM8K | 7,450 | Grade-school maths word problems, numeric answer |
| ARC-Challenge | 1,470 | Science multiple choice |
| Winogrande | 1,267 | 2-option pronoun resolution |
| MBPP | 427 | Python programming |
| MT-Bench | 80 | Open-ended multi-turn chat |
| **Total** | **34,778** | |

## 3.3 Audit: reading answers, not scores

Before using any score, we read the models' responses directly (`scripts/audit_routerbench.py`,
`notes/routerbench_audit.md`). Score columns alone would have hidden four problems.

**A small model that answers at chance.** Mistral-7B answers "A" on 94% of MMLU questions, 100% of
HellaSwag questions and 86% of ARC questions. Its multiple-choice scores (0.25, 0.26 and 0.39) are
therefore a formatting failure, not a measure of how hard the questions are for it, and a router trained
on that pair would learn only that multiple choice should go to GPT-4. We dropped Mistral-7B as a small
model (decision D-05). Mixtral-8x7B shows a milder version of the same habit on HellaSwag, answering
"A" on 64% of items, so HellaSwag is reported separately from the headline results (D-06).

**GSM8K scores that are not correctness.** RouterBench's GSM8K scores take only five values (0, 0.25,
0.5, 0.75 and 1), GPT-4 never reaches 1.0 on any item, and GPT-4's mean is 0.66 — far below its
published accuracy on the task. The responses themselves are complete, so we re-grade them (§3.4).

**Mislabelled splits.** The 7,450 GSM8K items are drawn from GSM8K's *training* split (7,473 items),
not its test split. Conversely, the 14,042 MMLU items labelled `val` are MMLU's full *test* set: the
per-subject counts match it exactly (for example professional_law 1,534, abstract_algebra 100). Neither
affects our method, but both matter for the limitations in §3.8.

**A dataset too small for a guarantee.** MT-Bench has 80 items. The threshold-selection procedure of
Chapter 5 needs on the order of 1,000 calibration questions per dataset to give a useful bound, so
MT-Bench is reported as a point estimate with a confidence interval only.

## 3.4 Re-grading GSM8K

We re-score every stored GSM8K response by numeric exact match against the gold answer from the GSM8K
training split. All 7,450 items matched a gold answer by question text.

The difficult part is choosing *which* number in a free-text answer is the model's final answer. Table
3.2 compares three extraction rules on all 7,450 items.

**Table 3.2 — Effect of the answer-extraction rule on graded accuracy (all 7,450 items).**

| Extraction rule | GPT-4 | Mixtral-8x7B | Mistral-7B |
|---|--:|--:|--:|
| Last number in the response | 0.844 | 0.540 | 0.343 |
| Result of the last "= X" | 0.942 | 0.596 | 0.384 |
| **Chosen:** last "= X", unless the closing sentence states a different number (numbers copied from the question excluded) | **0.949** | **0.593** | **0.374** |

"Last number" fails when the closing sentence restates part of the question ("…480 peanuts *in 6
minutes*"). "Last = X" credits intermediate values the model later abandons. The chosen rule
(`router.graders.extract_number`) handles both; each failure case is a regression test in
`tests/test_graders.py`.

The design required a grader to agree with human judgement on at least 95% of 50 hand-checked items
before any downstream number is trusted. We drew 50 items at random (25 GPT-4, 25 Mixtral) and checked
each by hand: **48 of 50 (96%) were graded correctly**, all 25 GPT-4 items and 23 of 25 Mixtral items.
Both errors are false negatives on correct Mixtral answers — one states the total before its
breakdown, the other gives the answer in a different unit — so Mixtral's GSM8K accuracy may be
understated by one or two points and the GPT-4–Mixtral gap correspondingly overstated
(`notes/gsm8k_grader_check.md`).

## 3.5 Gold answers for multiple choice

Scoring our own local model (Qwen3.5-4B, Chapter 6) on the multiple-choice datasets requires the
correct option letter for each question, which RouterBench does not store. We recover it from the
models themselves: whenever a model's answer was scored correct, the option letter it chose is the gold
letter. A letter is kept only if every correct model chose the same one and the letter is among the
options the question offers.

Across all four multiple-choice datasets the correct models **never disagreed** (zero conflicts).
Coverage is in Table 3.3; the missing questions are those no model answered correctly.

**Table 3.3 — Coverage of recovered gold letters.**

| Dataset | Questions | With gold letter | Coverage |
|---|--:|--:|--:|
| ARC-Challenge | 1,470 | 1,456 | 99.1% |
| MMLU | 14,042 | 13,407 | 95.5% |
| Winogrande | 1,267 | 1,267 | 100.0% |
| HellaSwag | 10,042 | 9,800 | 97.6% |

The questions without a gold letter are, by construction, the hardest in the corpus. Results for our
local model on multiple choice exclude them, which flatters its accuracy slightly; this is stated beside
every such number.

## 3.6 Near-duplicates and the frozen splits

### 3.6.1 Why grouping is needed

Benchmarks contain items that are not copies but are close enough to leak information between splits:
several MMLU questions on one reading passage, Winogrande's deliberate twin sentences (one word changed,
opposite answer), GSM8K problems rewritten with new names. We therefore assign every question a
`group_id`, join near-duplicates into one group, and split by group so that no group crosses
train, calibration and test (DESIGN §2.5). Splits are 60/20/20 within each dataset, with a fixed seed.

### 3.6.2 Lexical pass

The first pass finds pairs that share wording (`router/dedup.py`). Each prompt becomes a set of word
3-gram shingles; shingles that appear in more than 2% of a dataset's prompts (template text such as
MMLU's "Please answer with the letter of the correct answer") are removed first, so two different
questions do not look alike merely because they share a template. MinHash with locality-sensitive
hashing proposes candidate pairs, and each candidate is then checked exactly: Jaccard similarity of
at least 0.8, or containment of at least 0.8 for "the same text with a prefix added". Together with
seven exact-duplicate groups found while loading, this pass found **585 pairs**: MMLU 466 (mostly
questions sharing a passage), Winogrande 109 (twins), MBPP 7, ARC 2 and GSM8K 1.

### 3.6.3 Semantic pass, and why raw embeddings fail

The second pass looks for rewrites that share meaning but not wording, using the 768-dimensional
`bge-base-en-v1.5` sentence embeddings that lane B also uses as router features [@xiao2023bge].
The obvious rule — join two questions if the cosine similarity of their embeddings is at least 0.95 —
turned out to be unusable. On MMLU it produced 253,297 pairs and a single group containing almost all
895 questions of the *moral scenarios* subject.

The cause is the prompt template. Every moral-scenarios question opens with the same 29-word preamble
("For which of these two scenarios does the main character…"), and the embedding of a short question is
dominated by that shared text: two random moral-scenarios questions have a **median cosine similarity of
0.954**, above the threshold, even though they ask about unrelated situations.

We remove the template before comparing. Each embedding is centred on the mean embedding of its template
group — its MMLU subject, or its dataset for the others — and re-normalised
(`router.dedup.center_embeddings`). What remains is the part of the question that is specific to it:
the median similarity of two random moral-scenarios questions drops to **−0.01**. A pair is joined when
the centred cosine is at least 0.90. We set this threshold by reading samples of pairs at several
levels; at 0.90 and above they are genuine near-duplicates of four kinds:

- MMLU questions on the same reading passage, and the same question filed under two subjects (for
  example a sports-physiology question that appears in both college-medicine and clinical-knowledge);
- GSM8K rewrites with names changed and numbers kept ("Josie received $50 as a gift…" / "Amanda received
  $50 as a gift…");
- HellaSwag items that continue the same video scene, one item's context being the start of another's;
- Winogrande twins already found by the lexical pass.

**Figure 3.1 (to draw):** distribution of pairwise cosine similarity within MMLU moral-scenarios, raw
versus centred, with the 0.95 and 0.90 thresholds marked.

### 3.6.4 Result and freeze

Table 3.4 summarises both passes. The final corpus has 33,710 groups; the largest contains 15
questions. The splits were then frozen (decision D-11, corpus fingerprint `910a584c8279baa7`) and
published as a data release; every result in this thesis uses them.

**Table 3.4 — Near-duplicate pairs by dataset and method.**

| Dataset | Lexical | Semantic (centred) | Total |
|---|--:|--:|--:|
| MMLU | 466 | 799 | 1,265 |
| Winogrande | 109 | 260 | 369 |
| HellaSwag | 0 | 194 | 194 |
| GSM8K | 1 | 32 | 33 |
| MBPP | 7 | 7 | 14 |
| ARC-Challenge | 2 | 6 | 8 |
| MT-Bench | 0 | 0 | 0 |
| **Total** | **585** | **1,298** | **1,883** |

Note that a pair can be found by both passes, so the semantic column counts pairs from the second pass,
not pairs that only it found. Joining all pairs merged 1,061 groups.

**Table 3.5 — Frozen splits.**

| Dataset | Train | Cal | Test |
|---|--:|--:|--:|
| MMLU | 8,425 | 2,809 | 2,808 |
| HellaSwag | 6,025 | 2,009 | 2,008 |
| GSM8K | 4,470 | 1,490 | 1,490 |
| ARC-Challenge | 882 | 294 | 294 |
| Winogrande | 761 | 253 | 253 |
| MBPP | 256 | 86 | 85 |
| MT-Bench | 48 | 16 | 16 |
| **Total** | **20,867** | **6,957** | **6,954** |

The calibration split (cal) is used only to fit calibrators and choose the routing threshold; the test
split is used once, for the final numbers (Chapter 5).

## 3.7 Choosing the model pairs

A router can only save money where the small model is sometimes right and the frontier model is
sometimes better. We measure this *headroom* as the frontier model's mean score minus the small model's,
per dataset (Table 3.6), and require at least 0.05 on every headline dataset (the kill-check of the
design).

**Table 3.6 — Mean score per dataset and headroom against GPT-4 (frozen corpus).**

| Dataset | n | GPT-4 | Mixtral-8x7B | Yi-34B | Headroom vs Mixtral | Headroom vs Yi |
|---|--:|--:|--:|--:|--:|--:|
| MMLU | 14,042 | 0.812 | 0.635 | 0.659 | 0.177 | 0.153 |
| HellaSwag | 10,042 | 0.840 | 0.417 | 0.743 | 0.423 | 0.097 |
| GSM8K | 7,450 | 0.949 | 0.593 | 0.660 | 0.356 | 0.289 |
| ARC-Challenge | 1,470 | 0.962 | 0.832 | 0.861 | 0.130 | 0.101 |
| Winogrande | 1,267 | 0.819 | 0.552 | 0.629 | 0.267 | 0.190 |
| MBPP | 427 | 0.686 | 0.541 | 0.386 | 0.145 | 0.300 |
| MT-Bench | 80 | 0.806 | 0.648 | 0.644 | 0.159 | 0.162 |

Every headline dataset passes the kill-check for both candidate small models. We use the pairs in two
roles (D-08). **Mixtral-8x7B vs GPT-4** is the development pair: its answers exist today, so every lane
could build on it from the start, and it is the pair RouteLLM evaluated [@ong2025routellm], which makes
our results directly comparable. **Qwen3.5-4B vs GPT-4** is the headline pair and the one demonstrated
live: Qwen is small enough to run on a 16 GB laptop. Its answers are generated by us on the project's
Mac (Track B, Chapter 6) for the same questions and splits, and aligned with RouterBench's stored GPT-4
scores (`router/trackb.py`).

Headroom is an average. What a router actually needs is a minority of questions on which only the
frontier model succeeds. Table 3.7 shows this for the Mixtral pair, together with the *oracle* frontier
share: the fraction of questions a router with perfect knowledge would have to escalate to recover 95%
of the quality gap.

**Table 3.7 — Where the gap comes from (Mixtral pair, full corpus).**

| Dataset | Only GPT-4 right | Only Mixtral right | Both wrong | Oracle frontier share at PGR 0.95 |
|---|--:|--:|--:|--:|
| ARC-Challenge | 13.9% | 1.0% | 2.9% | 12.4% |
| MMLU | 21.6% | 3.9% | 14.9% | 16.8% |
| Winogrande | 35.7% | 9.0% | 9.1% | 25.4% |
| GSM8K | 36.6% | 1.0% | 4.1% | 33.8% |
| HellaSwag | 45.6% | 3.4% | 12.7% | 40.2% |

Pooled over the four headline datasets, a perfect router would send 22% of questions to GPT-4. That is
the ceiling against which Chapters 4 and 5 measure real routers; a random router needs about 95%.

## 3.8 Limitations of the data

- **One answer per model.** RouterBench stores a single response per model and question, with no
  log-probabilities and no record of the sampling temperature, so we cannot measure a model's
  uncertainty on Track A. Track B, generated by us, records both.
- **Grader errors.** The GSM8K grader misjudges about 4% of answers, all false negatives in our check
  (§3.4); gold letters are missing for the hardest 0–4.5% of multiple-choice questions (§3.5).
- **Possible contamination.** The GSM8K items come from its training split, which some of the evaluated
  models may have seen during training.
- **2023 prices.** RouterBench's costs are 2023 API prices. All costs in this thesis are recomputed at
  2026 list prices from token counts (`configs/models.yaml`), with the frontier arm priced as Claude
  Sonnet 4.6.
- **Benchmarks are not traffic.** The corpus is benchmark questions, not real user queries. Chapter 7
  shows this matters in practice: a router trained on it sends short everyday questions to the frontier
  model.

## Reproducing this chapter

```bash
gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench \
  -p corpus.parquet -p scores.parquet -p dedup_pairs.parquet -p manifest.json -p SHA256SUMS
python scripts/verify_data.py                 # fingerprints
python scripts/audit_routerbench.py           # §3.3
python scripts/01_acquire_routerbench.py      # §3.2, §3.4, §3.5 (needs the raw pickle and GSM8K train)
python scripts/02_dedup.py --embeddings artifacts/features/features.parquet   # §3.6
```
