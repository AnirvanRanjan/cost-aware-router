# Chapter 3 — Data: Corpus, Labels and Splits

## 3.1 Foundations: Why Data Integrity Governs Routing Capacity

A cost-aware router is fundamentally a statistical discriminator trained to delineate the operational boundary between small and large language models. The router possesses no intrinsic knowledge beyond the empirical supervision encoded in its training data. Consequently, three fundamental properties of the corpus establish the theoretical upper bound on all downstream claims:

1. **Labels Must Faithfully Measure Task Correctness:** If benchmark scoring functions conflate stylistic choices, formatting idiosyncrasies, or parsing errors with genuine reasoning capability, the router optimizes for superficial output artefacts rather than underlying problem difficulty.
2. **The Lightweight Model Must Possess Non-Trivial Headroom:** If a candidate small model operates near chance level across a domain, it provides zero discriminative gradient: every query appears uniformly intractable, eliminating opportunities for selective routing.
3. **Evaluation Partitions Must Be Free of Semantic Leakage:** Standard academic benchmarks contain near-identical paraphrases, twin formulations, and shared reading passages. If duplicate pairs straddle the training and testing sets, evaluation metrics quantify memorization rather than out-of-distribution generalization.

This chapter details the systematic curation, forensic auditing, label reconstruction, and leakage-free partitioning of our empirical corpus.

---

## 3.2 Source Corpus: RouterBench

Our empirical foundation is derived from **RouterBench** [@hu2024routerbench], a public repository containing pre-computed inference outputs from 11 frontier and open-weight language models evaluated across identical prompt sequences. Its primary utility for this research is economic: by leveraging RouterBench's cataloged GPT-4 responses across tens of thousands of complex queries, the substantial API expenditure required to generate frontier ground-truth outputs was eliminated during development.

We ingest the complete zero-shot database (`routerbench_0shot.pkl`, comprising 36,497 rows and 37 attributes). Each entry records the raw prompt, the task identifier (`eval_name`), and the generated response, numeric score, and token consumption for all 11 models. Ingesting this data required two non-trivial pre-processing steps:
- **Stringified AST Parsing:** Prompts and raw model completions are stored as string-encoded Python lists; these are safely parsed into native text primitives using `ast.literal_eval` prior to tokenization or feature extraction.
- **Hierarchical Dataset Mapping:** The source metadata catalogs tasks at fine-grained sub-domain levels (e.g., individual MMLU sub-disciplines such as `clinical_knowledge` or `abstract_algebra`). The ingestion pipeline maps these fine categories to standardized benchmark datasets via `router.data.source_of`.

To ensure statistical rigor, we systematically prune 21 specialized micro-benchmarks totaling 1,719 rows (e.g., small translation tasks, `accounting_audit`, `bias_detection`, `consensus_summary`, and `abstract2title`). These excluded tasks feature fewer than 400 total questions, lack representation in standard LLM literature, and exhibit negative model headroom (where the smaller model inexplicably out-scores the frontier model due to evaluation noise). 

The sanitized base corpus comprises **34,778 questions** spanning seven canonical benchmarks (Table 3.1).

**Table 3.1 — Composition of the Sanitized Base Corpus.**

| Benchmark Dataset | Total Prompts | Underlying Task Architecture |
|---|--:|---|
| **MMLU** [@hendrycks2021mmlu] | 14,042 | 4-option multiple-choice spanning 57 academic & professional subjects |
| **HellaSwag** | 10,042 | 4-option commonsense sentence completion |
| **GSM8K** [@cobbe2021gsm8k] | 7,450 | Multi-step grade-school arithmetic word problems with numeric solutions |
| **ARC-Challenge** | 1,470 | Complex grade-school science multiple-choice |
| **Winogrande** | 1,267 | 2-option pronoun disambiguation and coreference resolution |
| **MBPP** | 427 | Short Python programming problems verified via test suites |
| **MT-Bench** | 80 | Multi-turn open-ended conversational evaluation |
| **Total Corpus** | **34,778** | |

---

## 3.3 Forensic Audit: Inspecting Responses Beyond Aggregate Scores

Prior to training predictive models, we performed a comprehensive forensic audit of RouterBench completions (`scripts/audit_routerbench.py`, `notes/routerbench_audit.md`). Evaluating raw generated text rather than relying on provided score columns exposed four major methodological failures in the upstream dataset:

### 3.3.1 Formatting Collapse and Chance-Level Small Models
A small model selected for routing must exhibit genuine task competence. Our audit revealed that **Mistral-7B completely collapsed under RouterBench's multiple-choice prompt template**, outputting the single token "A" on 94% of MMLU questions, 100% of HellaSwag items, and 86% of ARC-Challenge questions. Its recorded benchmark accuracies (0.25, 0.26, and 0.39, respectively) reflected pure formatting failure rather than reasoning capacity. A router trained against Mistral-7B would merely learn to detect multiple-choice formatting and route all structured prompts to GPT-4. Consequently, Mistral-7B was formally eliminated as a candidate small model (Decision D-05). 

A milder version of this pathology affected **Mixtral-8x7B on HellaSwag**, where the model defaulted to option "A" on 64% of items. HellaSwag was therefore sequestered from headline routing metrics and reported as a separate diagnostic task (Decision D-06).

### 3.3.2 Truncated and Corrupted GSM8K Evaluation
RouterBench’s stored scores for GSM8K were severely degraded: accuracy took only five discrete fractional values ($0, 0.25, 0.5, 0.75, 1.0$), GPT-4 never attained a score of $1.0$ on any individual problem, and GPT-4’s mean recorded accuracy was an abysmal $0.66$—radically divergent from its true performance ($>0.92$). Inspection revealed that RouterBench evaluated math responses using fuzzy string overlap rather than mathematical target extraction. Fortunately, the raw generated chain-of-thought texts were preserved intact, enabling complete numeric re-grading (§3.4).

### 3.3.3 Mislabeled Dataset Splits
We identified that RouterBench’s 7,450 GSM8K questions originate entirely from GSM8K’s official *training partition* (7,473 prompts), rather than its canonical test split. Conversely, the 14,042 MMLU questions designated as `val` within RouterBench correspond exactly to MMLU’s official *test set* (verified by per-subject row counts). While our leak-free re-partitioning maintains internal validity, these historical origins are carefully accounted for in §3.8.

### 3.3.4 Statistical Under-Powering on MT-Bench
MT-Bench contains only 80 total interactions. Evaluating statistical risk control (Chapter 5) requires approximately 1,000 calibration queries to certify finite-sample confidence intervals without trivial bound inflation. Consequently, MT-Bench was isolated from formal certification sweeps and evaluated strictly via empirical point estimation.

---

## 3.4 Rigorous Numeric Re-Grading of GSM8K

To restore ground-truth integrity on mathematical reasoning, we developed an exact numeric extraction engine (`router.graders.extract_number`). Every stored GSM8K problem was matched against canonical GSM8K records to recover ground-truth numeric values.

Extracting a final mathematical prediction from unstructured free-form text requires robust parsing rules. Table 3.2 benchmarks three extraction heuristics across all 7,450 GSM8K instances.

**Table 3.2 — Impact of Numeric Extraction Heuristics on Graded Model Accuracy ($N=7,450$).**

| Extraction Heuristic | Parsing Mechanism | GPT-4 Graded Accuracy | Mixtral-8x7B Graded Accuracy | Mistral-7B Graded Accuracy |
|---|---|--:|--:|--:|
| **Rule 1: Last Number** | Scans backward from text termination; selects first regex match `[-+]?\d*\.?\d+` | 0.844 | 0.540 | 0.343 |
| **Rule 2: Last Equation** | Extracts target value $X$ from the terminal assignment equation `= X` | 0.942 | 0.596 | 0.384 |
| **Rule 3: Contextual Parser (Chosen)** | Extracts terminal `= X`, with exception handling for trailing explanatory sentences and question-number exclusion | **0.949** | **0.593** | **0.374** |

Rule 1 fails catastrophically when a model restates problem parameters in its concluding remark (e.g., *"...yielding 480 peanuts across 6 distinct bags"*, where 6 is erroneously captured instead of 480). Rule 2 fails when intermediate calculation lines are abandoned. 

Our chosen **Contextual Parser** resolves both failure modes: it targets terminal equality statements while verifying whether the trailing clause states an alternate consolidated total, explicitly filtering numbers mirrored from the prompt text.

### Grader Verification Audit
To ensure zero grader bias prior to downstream modeling, we executed a double-blind audit across 50 randomly selected problems (25 GPT-4 solutions, 25 Mixtral solutions) evaluated by human experts:
- **Human-Grader Agreement:** **48 out of 50 solutions (96.0%)** were graded identically to human consensus.
- **Error Analysis:** All 25 GPT-4 evaluations were perfectly accurate. The two discrepancies were conservative false negatives on Mixtral responses (one solution stated the correct aggregate prior to itemized breakdown; the other converted units correctly without repeating raw digits). Thus, Mixtral's graded accuracy is at worst slightly understated by 1–2 percentage points, ensuring that our reported headroom against GPT-4 is conservative (`notes/gsm8k_grader_check.md`).

---

## 3.5 Consensus-Based Gold Answer Recovery for Multiple Choice

Deploying our on-device edge model (Qwen3.5-4B, Chapter 6) required evaluating multiple-choice queries where RouterBench provided model binary correctness indicators but omitted the correct target option letter (`A`, `B`, `C`, or `D`).

To reconstruct missing ground truth, we developed an exact consensus recovery algorithm:
$$\text{Gold Letter}(q) = \ell \iff \forall m \in \mathcal{M}_{\text{correct}}(q), \quad \text{Choice}(m, q) = \ell$$
where $\mathcal{M}_{\text{correct}}(q)$ denotes the set of all RouterBench models marked correct on prompt $q$. A candidate letter is validated only if every correct model selected identical option letters and the recovered character exists within the valid choice set of prompt $q$.

Across all multiple-choice benchmarks, the correct models exhibited **zero conflicting choices** (100% mutual consensus). Table 3.3 summarizes the recovered ground-truth coverage.

**Table 3.3 — Ground-Truth Target Letter Recovery Rates across Multiple-Choice Datasets.**

| Multiple-Choice Dataset | Total Ingested Questions | Successfully Recovered Gold Letters | Effective Target Coverage |
|---|--:|--:|--:|
| **ARC-Challenge** | 1,470 | 1,456 | **99.1%** |
| **MMLU** | 14,042 | 13,407 | **95.5%** |
| **Winogrande** | 1,267 | 1,267 | **100.0%** |
| **HellaSwag** | 10,042 | 9,800 | **97.6%** |
| **Combined Multiple Choice** | **26,821** | **25,930** | **96.7%** |

The unrecovered entries (e.g., 635 questions in MMLU) correspond to edge-case problems where every single evaluated model in RouterBench failed. These ultra-challenging items are excluded from edge-model evaluations, slightly flattering small model performance; this filtering boundary is explicitly noted across all results.

---

## 3.6 Semantic Deduplication and Frozen Split Partitioning

### 3.6.1 The Imperative for Cluster-Based Splitting
Academic benchmarks frequently incorporate near-duplicate formulations that induce severe statistical leakage if distributed randomly across splits:
- **Passage-Sharing Clusters:** In MMLU, 5 to 10 distinct questions often analyze a shared clinical case or legal contract.
- **Adversarial Twins:** Winogrande constructs paired sentences differing by a single contrasting word with inverted target labels.
- **Template Paraphrases:** GSM8K contains identical word problems modified only by character names and currency values.

To eliminate data leakage, we assign every prompt to an invariant `group_id`, join all identified near-duplicates into unified disjoint clusters, and execute group-stratified partitioning such that no cluster straddles train, calibration, and test boundaries (DESIGN §2.5).

### 3.6.2 Lexical Deduplication Pass
The first deduplication tier executes lexical MinHash Locality-Sensitive Hashing (LSH) on word 3-gram shingles (`router/dedup.py`). Crucially, raw shingling on benchmark prompts risks massive false clustering driven by boilerplate prompt instructions (e.g., MMLU's universal prompt suffix: *"Please respond with the letter corresponding to the correct option"*). 

We construct a frequency-based stop-shingle filter: any 3-gram occurring in more than 2% of a dataset's prompts is stripped prior to hashing. Candidate pairs identified by LSH are verified under exact string metrics:
$$\text{Match}(q_1, q_2) = \mathbf{1}\left[\text{Jaccard}(q_1, q_2) \ge 0.80 \;\lor\; \text{Containment}(q_1, q_2) \ge 0.80\right]$$
The lexical pass isolated **585 duplicate pairs** (MMLU: 466, Winogrande: 109, MBPP: 7, ARC-Challenge: 2, GSM8K: 1).

### 3.6.3 Semantic Pass: Why Raw Sentence Embeddings Fail
To detect paraphrases lacking lexical overlap, we extract 768-dimensional sentence embeddings using `bge-base-en-v1.5` [@xiao2023bge]. However, naive cosine similarity thresholding failed catastrophically: enforcing a standard threshold of $\cos(e_1, e_2) \ge 0.95$ on MMLU yielded 253,297 pairwise edges, collapsing all 895 prompts of the `moral_scenarios` category into a single massive cluster.

The root cause is prompt template dominance: every `moral_scenarios` question begins with an identical 29-word instructional preamble:
> *"For which of these two scenarios does the main character violate standard societal behavioral expectations?..."*

Because the semantic mass of short questions is overwhelmed by this shared preamble, any two random `moral_scenarios` questions exhibit a **median raw cosine similarity of 0.954**, despite describing unrelated social situations.

We resolved this pathology by implementing **template-centered embedding projections** (`router.dedup.center_embeddings`). For each task or sub-discipline $k$, we compute the centroid embedding:
$$\bar{\mathbf{e}}_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathbf{e}_i$$
Each individual prompt embedding $\mathbf{e}_i$ is centered and re-normalized onto the unit hypersphere:
$$\tilde{\mathbf{e}}_i = \frac{\mathbf{e}_i - \bar{\mathbf{e}}_k}{\|\mathbf{e}_i - \bar{\mathbf{e}}_k\|_2}$$
Centering isolates the query-specific semantic variance: the median pairwise similarity between unrelated `moral_scenarios` prompts plummeted from **+0.954 to −0.012**. 

Enforcing a centered threshold of $\cos(\tilde{\mathbf{e}}_1, \tilde{\mathbf{e}}_2) \ge 0.90$ isolated **1,298 genuine semantic duplicate pairs** across the corpus (Table 3.4), successfully capturing shared-passage questions, identical arithmetic structures with altered entities, and overlapping video descriptions in HellaSwag.

**Table 3.4 — Duplicate Pair Recovery Breakdown across Deduplication Passes.**

| Benchmark Dataset | Lexical LSH Matches | Centered Semantic Matches | Total Unique Merged Pairs |
|---|--:|--:|--:|
| **MMLU** | 466 | 799 | 1,265 |
| **Winogrande** | 109 | 260 | 369 |
| **HellaSwag** | 0 | 194 | 194 |
| **GSM8K** | 1 | 32 | 33 |
| **MBPP** | 7 | 7 | 14 |
| **ARC-Challenge** | 2 | 6 | 8 |
| **MT-Bench** | 0 | 0 | 0 |
| **Corpus Total** | **585** | **1,298** | **1,883** |

### 3.6.4 Final Split Architecture and Release Freeze
Merging all 1,883 duplicate pairs collapsed the 34,778 prompts into **33,710 strictly independent question groups** (maximum group size: 15 questions). 

We partitioned the groups using a deterministic pseudo-random seed into a **60% Train, 20% Calibration, and 20% Test** split structure (Table 3.5). The partition was frozen under release fingerprint `data-910a584c8279baa7` (Decision D-11).

**Table 3.5 — Frozen Partition Structure across Independent Benchmark Groups.**

| Benchmark Dataset | Training Split ($N_{\text{train}}$) | Calibration Split ($N_{\text{cal}}$) | Test Split ($N_{\text{test}}$) | Total Ingested Prompts |
|---|--:|--:|--:|--:|
| **MMLU** | 8,425 | 2,809 | 2,808 | 14,042 |
| **HellaSwag** | 6,025 | 2,009 | 2,008 | 10,042 |
| **GSM8K** | 4,470 | 1,490 | 1,490 | 7,450 |
| **ARC-Challenge** | 882 | 294 | 294 | 1,470 |
| **Winogrande** | 761 | 253 | 253 | 1,267 |
| **MBPP** | 256 | 86 | 85 | 427 |
| **MT-Bench** | 48 | 16 | 16 | 80 |
| **Total Frozen Corpus** | **20,867** | **6,957** | **6,954** | **34,778** |

The calibration split is restricted exclusively to hyperparameter tuning, calibrator fitting, and Learn-then-Test threshold certification. The test split was queried strictly once for final certified reporting.

---

## 3.7 Model Pair Selection and Headroom Dynamics

A model router provides economic utility only if the small model succeeds on a substantial fraction of prompts while the frontier model resolves a significant subset of the small model's failures. We quantify this viability through **Headroom** ($\Delta$):
$$\Delta = \mathbb{E}[Y_{\text{frontier}}] - \mathbb{E}[Y_{\text{small}}]$$
Our design requires $\Delta \ge 0.05$ across every headline evaluation dataset as an explicit viability kill-check (Table 3.6).

**Table 3.6 — Mean Benchmark Accuracy and Headroom against Frontier GPT-4.**

| Benchmark Dataset | Evaluation Prompts ($N$) | Frontier GPT-4 | Mixtral-8x7B | Yi-34B | Headroom $\Delta$ (Mixtral) | Headroom $\Delta$ (Yi-34B) |
|---|--:|--:|--:|--:|--:|--:|
| **MMLU** | 14,042 | 0.812 | 0.635 | 0.659 | **+0.177** | +0.153 |
| **HellaSwag** | 10,042 | 0.840 | 0.417 | 0.743 | **+0.423** | +0.097 |
| **GSM8K** | 7,450 | 0.949 | 0.593 | 0.660 | **+0.356** | +0.289 |
| **ARC-Challenge** | 1,470 | 0.962 | 0.832 | 0.861 | **+0.130** | +0.101 |
| **Winogrande** | 1,267 | 0.819 | 0.552 | 0.629 | **+0.267** | +0.190 |
| **MBPP** | 427 | 0.686 | 0.541 | 0.386 | **+0.145** | +0.300 |
| **MT-Bench** | 80 | 0.806 | 0.648 | 0.644 | **+0.159** | +0.162 |

### Dual Role Architecture (Decision D-08)
1. **Development Pair (Mixtral-8x7B vs. GPT-4):** Provides a rapid development arm using pre-computed RouterBench outcomes, facilitating direct comparisons with RouteLLM [@ong2025routellm].
2. **Headline Edge Pair (Qwen3.5-4B vs. GPT-4):** Serves as our primary real-world edge deployment target. Qwen3.5-4B runs efficiently within 16 GB unified RAM on Apple Silicon. Its responses were generated directly across the frozen corpus partitions on our dedicated compute cluster and aligned with RouterBench ground-truth records (`router/trackb.py`).

### Oracle Routing Bounds
To contextualize real-world router efficiency, Table 3.7 presents the joint success matrix on the development pair alongside the theoretical **Oracle Frontier Share**—the absolute minimum fraction of queries an omniscient router would escalate to achieve $\text{PGR} \ge 0.95$.

**Table 3.7 — Error Decomposition and Oracle Escalation Ceiling (Mixtral vs. GPT-4).**

| Benchmark Dataset | Only GPT-4 Correct | Only Mixtral Correct | Both Models Incorrect | Oracle Frontier Share ($\text{PGR} = 0.95$) |
|---|--:|--:|--:|--:|
| **ARC-Challenge** | 13.9% | 1.0% | 2.9% | **12.4%** |
| **MMLU** | 21.6% | 3.9% | 14.9% | **16.8%** |
| **Winogrande** | 35.7% | 9.0% | 9.1% | **25.4%** |
| **GSM8K** | 36.6% | 1.0% | 4.1% | **33.8%** |
| **HellaSwag** | 45.6% | 3.4% | 12.7% | **40.2%** |
| **Pooled Headline Benchmarks** | **28.4%** | **2.6%** | **8.6%** | **22.0%** |

Across pooled headline benchmarks, an omniscient oracle escalates only **22.0%** of prompts to cloud endpoints. This establishes the theoretical performance ceiling against which our learned classifiers and cascade routers are calibrated.

---

## 3.8 Corpus Limitations and Boundary Conditions

- **Single-Sample Outputs on Track A:** RouterBench catalogs a single stochastic completion per prompt without associated logit distributions or token probabilities, preventing the extraction of output uncertainty features on historical models.
- **Grader Conservatism:** Our numeric GSM8K parser exhibits a 4.0% false-negative rate on edge-case formatting; gold option recovery leaves the top 3.3% most difficult multiple-choice prompts unindexed.
- **Historical Benchmark Contamination:** Because GSM8K prompts derive from its official training partition, certain evaluated models may have encountered these sequences during pre-training.
- **Pricing Currency:** All financial models in this work supersede RouterBench's obsolete 2023 rates, recomputing expenditure under 2026 enterprise token pricing (`configs/models.yaml`).
- **Academic vs. Production Domain Shift:** Academic prompts reflect stylized, curated evaluation patterns; Chapter 7 demonstrates that zero-shot routers trained exclusively on academic corpora encounter distributional drift when confronted with colloquial, open-ended conversational traffic.

---

## 3.9 Reproducibility Pipeline

The complete data pipeline, audit routines, and deduplication manifests are released under cryptographic verification:

```bash
# 1. Acquire cryptographic data release
gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router \
  -D artifacts/routerbench -p corpus.parquet -p scores.parquet -p dedup_pairs.parquet

# 2. Verify dataset SHA-256 fingerprints
python scripts/verify_data.py

# 3. Reproduce forensic audit metrics
python scripts/audit_routerbench.py

# 4. Execute numeric re-grading and consensus gold-letter extraction
python scripts/01_acquire_routerbench.py

# 5. Run template-centered semantic deduplication
python scripts/02_dedup.py --embeddings artifacts/features/features.parquet
```
