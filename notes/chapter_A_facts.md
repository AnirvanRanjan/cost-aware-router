# Chapter A (T50): Data — outline and facts pack

Everything the data chapter needs, with where each number comes from. This is a skeleton and a
fact sheet, not the chapter: write the prose in your own words (and follow UPES's rules on declaring
AI assistance). Every number below is reproducible from the frozen release `data-910a584c8279baa7`.

## Suggested structure (8–12 pages)

1. **Why the data decides the result** — a router can only learn what its labels say; three ways
   benchmark data misleads a router (format failures, wrong graders, leakage between splits).
2. **Source: RouterBench** — what it is, what we use, what we drop (§A1).
3. **Audit** — what we found by reading responses, not score columns (§A2).
4. **Re-grading GSM8K** (§A3) and **gold answer letters** (§A4).
5. **Near-duplicates and splits** (§A5): lexical pass, embedding pass, the template problem, freeze.
6. **Choosing the model pairs** — headroom table, why Mixtral is the development pair and Qwen the
   headline (§A6).
7. **Limitations** (§A7).

## A1. Source

| Fact | Value | Where |
|---|---|---|
| File | `routerbench_0shot.pkl` (withmartian/routerbench), 36,497 rows × 37 columns, 11 models | `notes/routerbench_audit.md` |
| Kept | 34,778 questions in 7 datasets | `artifacts/routerbench/manifest.json` |
| Dropped | 1,719 rows from 21 small custom tasks (Chinese-language tasks, accounting_audit, bias_detection, consensus_summary, abstract2title, test-match) | audit item 6 |
| Per dataset | MMLU 14,042 · HellaSwag 10,042 · GSM8K 7,450 · ARC-Challenge 1,470 · Winogrande 1,267 · MBPP 427 · MT-Bench 80 | manifest `kept` |
| Storage quirk | prompts and responses are stringified Python lists; parsed with `ast.literal_eval` | audit item 5 |

## A2. Audit findings (read responses, not scores)

| Finding | Evidence | Consequence |
|---|---|---|
| Mistral-7B answers "A" on 94% of MMLU, 100% of HellaSwag, 86% of ARC | audit item 1 | Dropped (D-05); its MCQ scores are a formatting failure |
| Mixtral answers "A" on 64% of HellaSwag | D-06 | HellaSwag reported separately |
| RouterBench GSM8K scores take 5 values (0, .25, .5, .75, 1); GPT-4 never scores 1.0; mean 0.66 vs ~0.9 true accuracy | audit item 2 | Re-graded (A3) |
| GSM8K items are the **train** split (7,450 of 7,473) | audit item 3 | Gold answers come from GSM8K train |
| MMLU is the full **test** set despite the `val` label (57 subjects, counts match) | audit item 4 | — |
| MT-Bench has 80 items | audit item 7 | Point estimate only, no guarantee |

## A3. GSM8K re-grading

| Fact | Value |
|---|---|
| Items matched to a gold answer | 7,450 / 7,450 |
| Extractor comparison (GPT-4 / Mixtral accuracy) | last number 0.844 / 0.540 · last "= X" 0.942 / 0.596 · **chosen rule 0.949 / 0.593** |
| Hand check (DESIGN gate ≥ 95%) | **48/50 = 96%**: GPT-4 25/25, Mixtral 23/25 (both misses are false negatives) |
| Effect | Mixtral's GSM8K accuracy may be understated by 1–2 points, so headroom 0.356 is slightly overstated |

Source: `notes/gsm8k_grader_check.md`, `router/graders.py` (`extract_number`).

## A4. Gold answer letters (T07)

Voted from every RouterBench model that answered correctly; kept only if all agree and the letter is
an offered option. **0 conflicts** in every dataset.

| Dataset | Coverage |
|---|---|
| ARC-Challenge | 1,456 / 1,470 = 99.1% |
| MMLU | 13,407 / 14,042 = 95.5% |
| Winogrande | 1,267 / 1,267 = 100% |
| HellaSwag | 9,800 / 10,042 = 97.6% |

Limitation: questions no model got right have no letter; they are the hardest ones, and Track B's MCQ
numbers exclude them.

## A5. Near-duplicates and splits (T04, D-11)

| Step | Result |
|---|---|
| Exact duplicates (loader) | 7 groups |
| Lexical pass (MinHash, word 3-grams, per-dataset boilerplate removed, Jaccard or containment ≥ 0.8) | 585 pairs: MMLU 466 (shared passages), Winogrande 109 (twin sentences), MBPP 7, ARC 2, GSM8K 1 |
| Embedding pass, raw bge cosine ≥ 0.95 | **Unusable**: chained all 895 MMLU moral-scenarios questions into one group (shared preamble) |
| Embedding pass, template-centred (subtract dataset / subject mean), residual cosine ≥ 0.90 | 1,298 more pairs; largest group 15 |
| Total | **1,883 pairs**, 1,061 groups merged, 33,710 groups |
| Frozen splits (60/20/20 per dataset, grouped, seed 7) | train 20,867 · cal 6,957 · test 6,954 |

Examples worth quoting (from `notes/dedup_report.md`): the same MMLU question filed under
college-medicine and clinical-knowledge; GSM8K "Josie"/"Amanda" rewrites with identical numbers;
HellaSwag items continuing the same video scene.

**Figure idea:** histogram of raw vs centred cosine for MMLU moral-scenarios, showing why centring is needed.

## A6. Model pairs and headroom (mean score, frozen corpus)

| Dataset | n | GPT-4 | Mixtral | Yi-34B | Gap vs Mixtral | Gap vs Yi |
|---|--:|--:|--:|--:|--:|--:|
| MMLU | 14,042 | 0.812 | 0.635 | 0.659 | 0.177 | 0.153 |
| HellaSwag | 10,042 | 0.840 | 0.417 | 0.743 | 0.423 | 0.097 |
| GSM8K | 7,450 | 0.949 | 0.593 | 0.660 | 0.356 | 0.289 |
| ARC-Challenge | 1,470 | 0.962 | 0.832 | 0.861 | 0.130 | 0.101 |
| Winogrande | 1,267 | 0.819 | 0.552 | 0.629 | 0.267 | 0.190 |
| MBPP | 427 | 0.686 | 0.541 | 0.386 | 0.145 | 0.300 |
| MT-Bench | 80 | 0.806 | 0.648 | 0.644 | 0.159 | 0.162 |

Every headline dataset passes the kill-check (gap ≥ 0.05). Pair roles: D-08.
Oracle CPT(95%) for Mixtral (share of questions that truly need GPT-4): MMLU 17%, GSM8K 34%, ARC 12%,
Winogrande 25%, HellaSwag 40%; pooled headline test 22%.

## A7. Limitations to state

- One stored response per model, at temperature unknown to us; no log-probabilities.
- GSM8K grader false negatives (A3); gold-letter gaps on the hardest questions (A4).
- GSM8K items are from its train split, so they may have been in some models' training data.
- RouterBench prices are 2023 prices; costs are re-priced at 2026 rates (Rev A, `configs/models.yaml`).

## References to cite

RouterBench (Hu et al., 2024); RouteLLM (Ong et al., 2024); GSM8K (Cobbe et al., 2021);
MMLU (Hendrycks et al., 2021); HellaSwag (Zellers et al., 2019); Winogrande (Sakaguchi et al., 2020);
ARC (Clark et al., 2018); MinHash/LSH (Broder, 1997); BGE embeddings (Xiao et al., 2023).
