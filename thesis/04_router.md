# Chapter 4 — Features and Router Architecture

**Author:** Huwaid Siddiqui (Lane B: Features & Router)  
**Primary Artifacts:** `router/features.py`, `router/models.py`, `router/calibrate.py`, `router/signals.py`, `router/cascade.py`, `router/bundle.py`  
**Reproducibility Scripts:** `scripts/04_train_two_head.py`, `scripts/05_calibrate.py`, `scripts/06_run_ablation.py`, `scripts/08_threshold_report.py`, `scripts/11_build_bundle.py`, `scripts/17_t60_seed_free_wsr.py`

---

## 4.1 Overview and Routing Formulation

The primary function of a cost-aware model router is to determine whether an incoming query $q$ can be successfully resolved by a lightweight on-device edge model $S$ or requires escalation to an expensive proprietary cloud endpoint $L$. In an interactive deployment, the router must arbitrate between two competing objectives: maximizing monetary savings by keeping execution local, while guaranteeing that the composite system's accuracy satisfies a predefined performance floor relative to the frontier model.

Under our system architecture, this arbitration occurs across two distinct operational paradigms:
1. **Pre-Execution Query-Only Routing:** Before generation begins, the router evaluates representations extracted exclusively from the prompt text $q$. This paradigm incurs zero initial generation latency and zero upfront API cost, but relies entirely on external linguistic proxies of problem difficulty.
2. **Post-Execution Cascade Routing:** The lightweight local model $S$ generates an answer first (incurring zero marginal API expenditure under Cost Model M1). The router then inspects internal generation dynamics—such as option probabilities and multi-sample consensus—to decide whether to accept the local output or escalate $q$ to the frontier API.

Crucially, in production environments, the router receives only the raw query string $q$. It has no access to task taxonomy, source benchmark metadata, or dataset identifiers. Any reliance on benchmark identity during inference constitutes severe data leakage. All feature representations developed in this chapter are strictly derived from the query string and the local model's generation dynamics alone.

---

## 4.2 Feature Engineering for Query Routing

To evaluate query complexity from complementary linguistic perspectives, we construct a hierarchical feature extraction pipeline comprising heuristic surface signals ($F_0$), lexical $n$-grams ($F_1$), and dense semantic embeddings ($F_2$).

### 4.2.1 Heuristic Syntactic and Structural Signals ($F_0$)

The $F_0$ extractor (`router.features.heuristics`, defined in `router/features.py`) parses the prompt string into a 23-dimensional scalar feature vector capturing surface length, formatting complexity, syntactic depth, and targeted lexical cues:

1. **Surface Metrics**:
   - `token_count`: Whitespace token count.
   - `char_count`: Total character length.
   - `sentence_count`: Sentence frequency detected via terminal punctuation regex `[.!?]+`.
   - `mean_word_length`: Average character length per token.
2. **Character Densities**:
   - `digit_density`: Proportion of numeric ASCII characters.
   - `punctuation_density`: Proportion of ASCII punctuation symbols.
   - `uppercase_density`: Ratio of capitalized characters, detecting acronyms and emphasized terms.
3. **Syntactic and Structural Complexity**:
   - `question_mark_count`: Frequency of `?`, differentiating interrogatives from imperative instructions.
   - `max_bracket_depth`: Maximum nesting depth of balanced delimiters `()`, `[]`, and `{}`, identifying mathematical expressions and nested computer code.
   - `type_token_ratio`: Ratio of unique lowercase tokens to total tokens, measuring lexical variety.
   - `flesch_kincaid_grade`: Textual readability index computed as $0.39 \times \left(\frac{\text{words}}{\text{sentences}}\right) + 11.8 \times \left(\frac{\text{syllables}}{\text{words}}\right) - 15.59$.
4. **Formatting and Domain Identifiers**:
   - `code_fence_count`: Markdown code block delimiters (```` or `~~~`).
   - `latex_count`: Frequency of mathematical LaTeX expressions (`\frac`, `\sqrt`, `\sum`, `\int`, `\partial`, `\alpha`).
   - `url_count`: HTTP/HTTPS web links.
   - `table_row_count`: Markdown table rows containing pipe characters (`|`).
   - `numeric_literal_count`: Count of discrete integer and floating-point literals.
5. **Task and Semantic Cues**:
   - `num_options`: Count of extracted multiple-choice options parsed via `router.graders.parse_options` (`router/graders.py`).
   - `has_reading_passage`: Binary flag indicating reading comprehension framing (e.g., *"refers to the following text"*).
   - `cue_word_prove`, `cue_word_derive`, `cue_word_step_by_step`, `cue_word_explain`: Targeted regular expressions flagging explicit requests for multi-step reasoning.
   - `imperative_verb_count`: Frequency of operational verbs (*"calculate"*, *"compute"*, *"solve"*, *"determine"*, *"analyze"*).

The $F_0$ feature extractor executes in pure Python with negligible runtime overhead ($<0.5\text{ ms}$ per query), making it highly suitable for latency-critical inference.

### 4.2.2 Lexical N-Gram Representations ($F_1$)

The $F_1$ extractor (`router.features.TfidfFeatureExtractor`, defined in `router/features.py`) captures domain terminology through sublinear term-frequency inverse-document-frequency ($\text{TF-IDF}$) representations. To represent both domain-specific keywords and morphological suffixes, $F_1$ fuses:
- Word $n$-grams ($n \in \{1, 2\}$), capped at the 50,000 most frequent features;
- Character $n$-grams ($n \in \{3, 4, 5\}$), capped at 50,000 features.

**Strict Anti-Leakage Protocol:** Under our cross-validation framework, the TF-IDF vocabulary, document frequencies, and idf weights are fitted *exclusively* on the training partition ($N_{\text{train}} = 20,868$). Fitting vectorizers on pooled corpus data would leak test-partition vocabulary distributions into the feature space, producing invalid out-of-sample evaluations.

### 4.2.3 Dense Semantic Embeddings ($F_2$)

While lexical $n$-grams capture surface vocabulary, they cannot model conceptual semantics or domain abstraction. The $F_2$ extractor (`router.features.SentenceEmbeddingExtractor`, defined in `router/features.py`) encodes queries using the `BAAI/bge-base-en-v1.5` dense embedding model [@xiao2023bge]. Queries are tokenized and mapped to a 768-dimensional vector representation with $L_2$ normalization:

$$\mathbf{e}(q) = \frac{\text{BERT}(q)}{\|\text{BERT}(q)\|_2} \in \mathbb{R}^{768}$$

Embeddings are precomputed across the frozen corpus and stored in `artifacts/features/features.parquet` (SHA256: `3d5eda199c6dff272f48d158c960d88cb068feec4bbb724dfceca214c0847bf0`). Inputs exceeding the maximum context window (512 tokens) are logged; across the frozen 34,778-query corpus, only 148 prompts ($0.42\%$) required truncation.

---

## 4.3 Router Architectures and Loss Formulation

A routing model predicts the expected capability differential between the frontier and small model:

$$\Delta(q) = s_L(q) - s_S(q) \in [-1, 1]$$

where $s_L(q), s_S(q) \in \{0, 1\}$ represent binary ground-truth correctness. Rather than optimizing standard classification accuracy, the router's primary training objective is ranking quality—measured by the Spearman rank correlation $\rho(\hat{\Delta}, \Delta)$—because cost-quality thresholding operates directly on the ordered spectrum of predicted scores.

```
                  Query Prompt q
                        │
                ┌───────┴───────┐
                ▼               ▼
             F2 (768d)       F0 (23d)
             Embedding      Heuristics (z-scored)
                └───────┬───────┘
                        ▼
               Fused Feature Vector x ∈ R^791
                        │
               ┌────────▼────────┐
               │  Trunk: Linear  │ 791 -> 256
               │      GELU       │
               │  Dropout(0.1)   │
               └────────┬────────┘
                        ├──────────────────┐
                        ▼                  ▼
                 ┌─────────────┐    ┌─────────────┐
                 │   Head S    │    │   Head L    │
                 │ Linear(256) │    │ Linear(256) │
                 │   Sigmoid   │    │   Sigmoid   │
                 └──────┬──────┘    └──────┬──────┘
                        ▼                  ▼
                    s_hat_S            s_hat_L
                        └────────┬─────────┘
                                 ▼
                   delta_hat = s_hat_L - s_hat_S
```

### 4.3.1 Baseline Classifiers

We implement three reference baselines:
1. **$F_0$ Logistic Regression (`f0_logreg`):** Regularized logistic regression ($L_2$, $C=1.0$) trained on the 23 $z$-score normalized heuristic features. The binary target indicates strict frontier superiority: $y = \mathbb{I}(s_L > s_S)$. Predicted probabilities are rescaled to $\hat{\Delta} = 2P(y=1|x) - 1$.
2. **$F_1$ Logistic Regression (`f1_logreg`):** An identical regularized linear classifier trained on the 100,000-dimensional sparse TF-IDF space.
3. **XGBoost Regressor (`xgboost_f0_f2`):** An ensemble of 100 gradient-boosted decision trees (maximum depth 4, learning rate 0.05, subsample ratio 0.8) trained directly on the continuous gap $\Delta = s_L - s_S$ using the concatenated $[F_2 \parallel F_0]$ feature representation.

### 4.3.2 The Two-Head MLP Architecture

Our primary neural router (`router.models.TwoHeadRouter`, defined in `router/models.py`) implements a dual-head neural network that jointly estimates the marginal success probabilities of both candidate models: $\hat{s}_S(q) \approx \mathbb{E}[s_S|q]$ and $\hat{s}_L(q) \approx \mathbb{E}[s_L|q]$.

The network receives the fused vector $\mathbf{x} = [\mathbf{e}(q) \parallel \mathbf{z}(q)] \in \mathbb{R}^{791}$, combining the 768-dimensional dense embedding with the 23 $z$-score normalized heuristic features. The architecture consists of:
1. **Shared Representation Trunk:**
   $$\mathbf{h} = \text{Dropout}_{0.1}\left(\text{GELU}\left(\mathbf{W}_{\text{trunk}} \mathbf{x} + \mathbf{b}_{\text{trunk}}\right)\right), \quad \mathbf{h} \in \mathbb{R}^{256}$$
2. **Decoupled Marginal Prediction Heads:**
   $$\hat{s}_S = \sigma\left(\mathbf{w}_S^T \mathbf{h} + b_S\right), \quad \hat{s}_L = \sigma\left(\mathbf{w}_L^T \mathbf{h} + b_L\right)$$
   where $\sigma(z) = (1 + e^{-z})^{-1}$ denotes the logistic sigmoid.

Decoupling the prediction into two marginal heads provides two distinct system advantages:
- **Interpretability and Cost-Benefit Normalisation:** By explicitly modeling the marginal success probabilities $\hat{s}_S$ and $\hat{s}_L$ rather than an abstract scalar difficulty index, the router enables query-specific cost-benefit calculations under dynamic pricing schedules.
- **Independent Modular Retraining:** When updating or replacing the small model (e.g., migrating from Mixtral-8x7B to Qwen3.5-4B in Track B, Task T36), the shared trunk and frontier head $\mathbf{w}_L$ can be frozen, requiring only $\mathbf{w}_S$ to be retrained on new edge labels.

### 4.3.3 Multi-Task Auxiliary Gap Loss

Training independent heads with standard binary cross-entropy alone ignores the correlation between model capabilities on identical prompts. We train the network using a composite multi-task objective (`router.models.two_head_loss` in `router/models.py`):

$$\mathcal{L}(\theta) = \ell_{\text{BCE}}(\hat{s}_S, s_S) + \ell_{\text{BCE}}(\hat{s}_L, s_L) + \lambda_{\text{gap}} \left\| (\hat{s}_L - \hat{s}_S) - (s_L - s_S) \right\|_2^2$$

Setting $\lambda_{\text{gap}} = 0.5$ penalizes errors in the predicted capability delta $\hat{\Delta} = \hat{s}_L - \hat{s}_S$, directly aligning gradient updates with downstream ranking performance.

---

## 4.4 Anti-Leakage Protocol and Probability Calibration

### 4.4.1 Grouped Sub-Splitting and Early Stopping

Neural networks trained with early stopping require a validation subset to prevent overfitting. Standard random splitting risks two severe forms of data leakage:
1. **Calibration Leakage:** Using the held-out calibration split (`cal`, $N=6,956$) for early stopping invalidates it for downstream conformal risk control and threshold certification.
2. **Duplicate Query Leakage:** Standard benchmarks contain paraphrased questions, thematic clusters, and shared reading passages. If related queries span train and validation partitions, early stopping measures memorisation rather than generalisation.

To enforce strict isolation:
- The network is trained *exclusively* on the training partition (`train`, $N=20,868$).
- Early stopping carves an 85/15 grouped sub-split (`train_sub` and `val_sub`) inside `train` using `router.splits.grouped_split` (`router/splits.py`). No duplicate or paraphrased group identifier (`group_id`) spans both partitions (`assert_no_leakage`).
- Optimization uses AdamW (learning rate $10^{-3}$, weight decay $10^{-2}$, batch size 64) with a patience of 8 epochs evaluated on `val_sub`.
- Normalisation statistics ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) are fitted strictly on `train_sub`.

### 4.4.2 Non-Parametric Isotonic Calibration

Raw neural network outputs passed through a sigmoid often display miscalibration. To ensure that $\hat{s}_S$ and $\hat{s}_L$ correspond to empirical frequencies of correctness, we fit post-hoc isotonic regression models (`router.calibrate.TwoHeadCalibrator`, implementing `sklearn.isotonic.IsotonicRegression` in `router/calibrate.py`) on the held-out calibration split:

$$\tilde{s}_S = g_S(\hat{s}_S), \quad \tilde{s}_L = g_L(\hat{s}_L)$$

where $g_S$ and $g_L$ are piecewise constant, monotonically non-decreasing functions. Calibration error is assessed using 15-bin uniform-mass Expected Calibration Error ($\text{ECE}_{15}$):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

**Table 4.1 — Calibration performance on held-out calibration split (`scripts/05_calibrate.py`).**

| Model Head | Uncalibrated $\text{ECE}_{15}$ | Isotonic Calibrated $\text{ECE}_{15}$ | Relative Error Reduction |
|---|:---:|:---:|:---:|
| **Frontier Head ($L$: `gpt-4-1106-preview`)** | $1.71\%$ | **$0.74\%$** | **$-56.7\%$** |
| **Small Head ($S$: `mistralai/mixtral-8x7b-chat`)** | $2.88\%$ | **$1.73\%$** | **$-40.0\%$** |

Across real multi-benchmark distributions, isotonic calibration reduces expected error by $40.0\%$ to $56.7\%$, guaranteeing dependable marginal probabilities for downstream decision rules.

---

## 4.5 Empirical Evaluation and Ablation Ladder

We benchmark router variants on the development model pair ($S = \text{Mixtral-8x7B-Chat}$, $L = \text{GPT-4-1106-Preview}$) across the untouched test partition ($N = 6,954$ queries across MMLU, GSM8K, ARC-Challenge, and Winogrande).

### 4.5.1 The Core Ablation Ladder

Table 4.2 presents the stepwise ablation progression across the feature and architectural hierarchy (`notes/ablation_ladder.md`). Performance is evaluated by:
- **AUC:** Area under the ROC curve for escalating when $s_L > s_S$;
- **$\rho(\hat{\Delta}, \Delta)$:** Spearman rank correlation between predicted and true capability deltas;
- **$\text{CPT}(95\%)$:** Cost-to-Pass-Through, the minimum fraction of queries routed to $L$ required to achieve $\text{PGR} \ge 0.95$ on the test curve;
- **APGR:** Area under the empirical Pareto Gain Ratio curve from $0\%$ to $100\%$ frontier budget.

**Table 4.2 — Core Ablation Ladder on the untouched test split ($N=6,954$, Mixtral development pair).**

| Configuration | Features | Architecture | AUC | $\rho(\hat{\Delta}, \Delta)$ | $\text{CPT}(95\%)$ | APGR |
|---|---|---|:---:|:---:|:---:|:---:|
| **Random router** | None | Uniform random score | 0.4974 | -0.0084 | 0.9600 | 0.4959 |
| **$F_0$ heuristics** | 23 surface signals | Logistic Regression | 0.6776 | 0.1359 | 0.8650 | 0.6372 |
| **$F_1$ lexical $n$-grams** | 100k TF-IDF terms | Logistic Regression | 0.6640 | 0.1420 | 0.8700 | 0.6251 |
| **$F_0 + F_2$ tree ensemble** | Heuristics + BGE embeddings | XGBoost Regressor | 0.6858 | 0.1484 | 0.8400 | 0.6453 |
| **$F_2$ dense embeddings** | 768d BGE embeddings | MLP (single head) | 0.6806 | 0.1585 | 0.8450 | 0.6395 |
| **$F_2 + F_0$ (Headline)** | 791d Fused representation | Two-Head MLP (Rule v1) | **0.6816** | **0.1719** | 0.8550 | 0.6396 |
| **+ Cost-normalized rule** | 791d Fused representation | Two-Head MLP (Rule v2) | 0.6758 | 0.1573 | 0.8600 | 0.6354 |
| **Oracle Router** | Ground truth $s_L - s_S$ | Exact capability delta | 1.0000 | 1.0000 | 0.2750 | 0.9306 |

### 4.5.2 Architectural and Feature Insights

1. **Information Content Across Tiers:**
   - $F_0$ heuristics alone achieve an AUC of $0.6776$, showing that surface markers (token length, LaTeX density, option formatting) provide an effective initial baseline of task difficulty.
   - High-dimensional $F_1$ lexical features ($\text{AUC} = 0.6640$) perform slightly worse than $F_0$, suffering from vocabulary sparsity and distribution shift between splits.
   - Dense $F_2$ embeddings provide semantic domain generalization, improving rank correlation from $0.1359$ ($F_0$) to $0.1585$.
   - **Fused Representation ($F_2 + F_0$):** Concatenating dense embeddings with normalized syntactic features yields the highest ranking correlation ($\rho = 0.1719$). Dense embeddings identify topical domain, while heuristic features preserve explicit syntactic and formatting constraints that embeddings smooth away.

2. **Decision Rule Comparison: Rule v1 vs Rule v2:**
   - **Rule v1 (Quality Gap Thresholding):** Routes to $L$ iff $\hat{\Delta}(q) = \hat{s}_L(q) - \hat{s}_S(q) > \tau$, optimizing pure capability recovery regardless of token length.
   - **Rule v2 (Cost-Normalized Gain):** Routes to $L$ iff expected quality gain per dollar exceeds a threshold:
     $$G(q) = \frac{\hat{s}_L(q) - \hat{s}_S(q)}{\hat{c}_L(q)} > \tau$$
     where $\hat{c}_L(q)$ estimates query cost from prompt token count ($n_{\text{in}} \approx \frac{\text{len}(q)}{4}$) and frontier pricing. Rule v2 deprioritizes lengthy queries where the small model has a modest capability gap but escalation would consume substantial token budget.

---

## 4.6 Statistical Guarantees and Certified Savings

A standard pitfall in LLM routing research is quoting "savings" directly from the test curve—selecting the threshold $\tau^*$ on test data that matches the $95\%$ performance target. Such test-picked numbers represent optimistic point estimates that fail in deployment.

To provide deployable guarantees, our pipeline determines $\tau^*$ on the **calibration split (`cal`)** using statistical risk-controlling bounds and applies $\tau^*$ exactly once to the untouched test split (`scripts/08_threshold_report.py`, `notes/threshold_report.md`).

### 4.6.1 Risk-Controlling Prediction Framework

Let the target Performance Gap Recovered be $\text{PGR} \ge 0.95$ with risk tolerance $\delta = 0.05$. We calibrate thresholds using three formal bounds:
1. **WSR Bound [@waudbysmith2024betting]:** Finite-sample valid concentration bound based on betting martingales, as utilized in Learn-then-Test [@angelopoulos2021ltt] and Risk-Controlling Prediction Sets [@bates2021rcps].
2. **Paired Bootstrap:** Asymptotic paired bootstrap over calibration queries with confidence level $1 - \delta$.
3. **Hoeffding Bound:** Classical distribution-free concentration bound.

**Table 4.3 — Out-of-sample test results for thresholds selected on `cal` (Target $\text{PGR} \ge 95\%$, $\delta=0.05$, Cost Model M1, Mixtral development pair).**

| Router Model | Decision Rule | Calibration Bound | Certified? | Frontier Calls ($\alpha_{\text{cal}}$) | Frontier Calls ($\alpha_{\text{test}}$) | Test PGR [95% CI] | Certified Savings [95% CI] |
|---|:---:|:---:|:---:|--:|--:|:---:|:---:|
| `f0_logreg` | v1 | WSR | Yes | 91.4% | 91.0% | 96.6% [95.3, 97.8] | **3.9% [3.4, 4.4]** |
| `f0_logreg` | v1 | Bootstrap | Yes | 90.2% | 89.9% | 95.9% [94.5, 97.2] | **4.1% [3.7, 4.6]** |
| `f1_logreg` | v2 | WSR | Yes | 91.6% | 91.7% | 96.5% [95.1, 97.8] | **3.2% [2.6, 3.8]** |
| `two_head_mlp` | v1 | WSR | Yes | 89.6% | 89.5% | 95.4% [94.0, 96.7] | **4.2% [3.7, 4.7]** |
| `two_head_mlp` | v2 | Hoeffding | Yes | 97.1% | 97.5% | 98.4% [97.6, 99.2] | **5.6% [4.3, 6.8]** |
| `two_head_mlp` | **v2** | **WSR** | **Yes** | **96.2%** | **96.2%** | **96.3% [95.2, 97.5]** | **9.4% [7.8, 10.9]** |
| `two_head_mlp` | **v2** | **Bootstrap** | **Yes** | **95.9%** | **95.6%** | **95.6% [94.3, 96.8]** | **11.0% [9.4, 12.7]** |
| *`two_head_mlp`* | *v2* | *Naive (point est.)* | *No* | *95.1%* | *95.0%* | *95.1% [93.6, 96.4]* | *12.7% [11.0, 14.5]* |

### 4.6.2 Certified Savings vs. Naive Optimism

As documented in Table 4.3:
- The naive point estimate on `cal` yields $12.7\%$ savings, but provides zero statistical coverage guarantee.
- Under the mathematically rigorous **WSR bound**, the Two-Head MLP with Rule v2 achieves a **certified saving of $9.4\%$ [95% CI: 7.8%, 10.9%]** while maintaining test PGR at $96.3\% \ge 95.0\%$.
- Under the paired bootstrap bound, certified savings reach **$11.0\%$ [9.4%, 12.7%]**.
- Hoeffding's bound is excessively conservative, demanding $97.5\%$ frontier calls and yielding only $5.6\%$ savings.

### 4.6.3 Breakdown by Target Benchmark

At the optimal certified threshold ($\tau^* = 0.0435$, selected via WSR on `two_head_mlp` v2), the routing behavior reflects distinct domain difficulties:
- **MMLU** ($N=2,808$): $96.8\%$ routed to frontier, yielding $\text{PGR} = 97.5\%$ and **$8.9\%$ monetary savings**.
- **GSM8K** ($N=1,490$): $93.8\%$ routed to frontier, yielding $\text{PGR} = 94.6\%$ and **$9.7\%$ monetary savings**.
- **ARC-Challenge** ($N=294$) and **Winogrande** ($N=253$): Both route $100\%$ to the frontier. Because these smaller test sets exhibit higher variance and large headroom gaps ($Q_L - Q_S \approx 0.13 - 0.25$), the risk-controlling bound conservatively escalates all queries to satisfy the strict global $\delta = 0.05$ risk ceiling.

---

## 4.7 Headroom Compression and the Cascade Router Architecture

The empirical findings on the Mixtral development pair expose a fundamental constraint: **query-only routers face an intrinsic ceiling on complex reasoning tasks.**

Within a domain such as grade-school math (GSM8K), whether an edge model succeeds or fails depends on dynamic execution factors—such as an intermediate arithmetic error or an incorrect sign—that cannot be deduced from prompt semantics alone. Consequently, query-only routing achieves modest certified savings ($9.4\%$).

### 4.7.1 The Headroom Compression Collapse on Track B

This ceiling becomes catastrophic when transitioning to modern high-capacity edge models. On our headline deployment pair (Track B: $S = \text{Qwen3.5-4B}$, $L = \text{GPT-4-1106-Preview}$), the small model achieves **$81.3\%$** baseline accuracy against **$88.7\%$** for the frontier model (`notes/t27_t28_locked_numbers.md`). The capability headroom contracts from $0.237$ (Mixtral) down to merely **$0.075$** ($\Delta = 0.0748$).

Under this compressed headroom, pre-execution query routers collapse: the certified savings of the feature-based query router (`query_f2`, Rule v2, WSR) reach only **$2.5\%$ [95% CI: 1.8%, 3.1%]** at test PGR $98.6\%$ (`notes/trackb_cascade_report.md`), while the Two-Head MLP achieves merely **$0.5\%$ [95% CI: 0.3%, 0.7%]** (`notes/qwen_query_report.md`). Text features simply do not contain sufficient discriminative signal to navigate this narrow margin.

### 4.7.2 The Cascade Formulation and Generation Confidence Signals

This limitation motivates the **Cascade Router Architecture** (`router/cascade.py`, Decisions D-12 and D-14):
1. The small model $S$ executes locally first (incurring zero marginal API cost under model M1).
2. The router inspects internal generation dynamics directly from the small model's completions (`router/signals.py`):
   - **Multiple-Choice Questions:** A single forward evaluation over option letters via `MLXOptionScorer` (`router/mcq_mlx.py`) yields the softmax option probabilities:
     - `p_chosen`: Probability of the most likely option letter.
     - `margin`: Probability margin separating the top option from the runner-up.
   - **Numeric and Arithmetic Questions (GSM8K):** $k$ stochastic completions ($T=0.7$, top-$p=0.8$) yield consensus metrics:
     - `agree`: Proportion of the $k$ sampled answers matching the majority output.
     - `n_distinct`: Number of unique numerical solutions produced.
     - `lp_mean`: Mean of the answers' token log-probabilities.
     - `lp_min`: Minimum mean log-probability across completions.
     - `len`: Log-transformed mean token length $\ln(1 + \bar{n}_{\text{tokens}})$.
3. If the small model exhibits high consensus and confident log-probabilities, the answer is accepted locally; otherwise, the query escalates to the cloud frontier API.

Exploratory analysis on the Mixtral pair (`scripts/12_cascade_explore.py`, `notes/cascade_explore.md`) demonstrated that incorporating a second opinion from Yi-34B (cross-model agreement between Mixtral-8x7B and Yi-34B-Chat, not self-consistency) boosted certified savings from $9.4\%$ to **$26.0\%$ [95% CI: 24.2%, 27.9%]** at $\text{PGR} = 97.9\%$. On Track B, Qwen's own self-agreement across $k=3$ samples provides the runnable on-device equivalent of that second opinion.

### 4.7.3 Track B Certified Results and Seed-Free Martingale Averaging

On the official Track B evaluation ($N=4,707$ test queries, release `trackb-6265adf7ffa4a602`), we evaluate gradient-boosted decision trees trained on these internal signals:

1. **Seed Variance in Sequential Martingales:** In standard Learn-then-Test, the sequential WSR betting martingale depends on the random permutation order of calibration queries. Sweeping permutation seeds $0$ through $19$ revealed significant seed sensitivity: certified test savings oscillated between **$40.8\%$ and $74.9\%$** (`notes/t60_certification_report.md`). A single random seed yielded **$40.8\%$ [95% CI: 38.7%, 43.0%]** at $\tau^* = 8.1093$.
2. **Seed-Free Certification via Terminal E-Value Averaging:** To eliminate permutation variance without violating statistical guarantees, Lane C formulated a seed-free martingale averaging procedure (`scripts/17_t60_seed_free_wsr.py`). By averaging terminal martingale wealth across $M=50$ random permutations:
   $$\bar{E} = \frac{1}{M} \sum_{m=1}^M E^{(m)}$$
   linearity of expectation guarantees that $\bar{E}$ remains an exact finite-sample e-value. Inverting this test via Markov's inequality yields a completely deterministic threshold ($\tau^* = 8.219$), certifying **$41.3\%$ [95% CI: 39.0%, 43.5%] API savings** at an empirical test PGR of **$102.8\%$ [95% CI: 99.1%, 107.0%]** (escalating $75.9\%$ of queries).
3. **Single-Sample Cascade Baseline (Decision D-14):** For environments where multi-sample decoding latency is prohibitive, the single-sample cascade (`cascade_1sample`, Rule v2) evaluates only one completion's mean log-probability (or option logit). This model achieves **$27.8\%$ [95% CI: 25.9%, 29.6%] certified savings** at $\text{PGR} = 97.7\%$ ($\tau^* = 5.86$, escalating $68.7\%$ of test queries).
4. **Feature Combination Pathology:** Adding dense $F_2$ embeddings back into the cascade feature vector (`cascade_3sample_f2_shrunk` and `cascade_conf+f2_raw`) severely degraded certified performance down to **$1.2\% - 3.4\%$** savings (`notes/trackb_cascade_report.md`). High-dimensional embedding noise caused tree splits to overfit spurious semantic correlations on the training set, degrading the clean epistemic signal provided by generation agreement and log-probabilities.

---

## 4.8 The Production Demonstration Router (Decision D-15)

In translating these models to the interactive on-device demonstration (`router/serve.py`), our architectural strategy underwent an important empirical evolution between Decision D-14 and Decision D-15.

### 4.8.1 The Failure Mode of Answer-Text Heuristics

Decision D-14 originally proposed packaging a single-sample cascade bundle (`router/bundle.py`, `--mode cascade`) that extracted heuristic signals from Qwen's answer *text*—including explicit arithmetic slip checks, regex-parsed multiple-choice letters, and targeted hedge words checked by the codebase (`assume`, `assuming`, `approximately`, `not sure`, `unclear`, `cannot`, `can't`, `however`, `wait`, `actually` in `router/cascade.py`)—combined with one answer's mean token log-probability.

However, during interactive validation on the demo suite (Task T45, `notes/demo_set_check.md`, `notes/demo_set_check_run2.md`), this text-heuristic bundle failed on free-form prompts:
- **Out-of-Distribution Sensitivity:** In run 1 with raw prompt generation, Qwen answered all 10 easy questions correctly, but the router escalated 8 of the 10 easy questions to the frontier API. In run 2 with the brief prompt template, all 10 easy questions were escalated despite Qwen answering every one correctly.
- **Failure Mechanism:** Free-form answers to conversational questions (e.g. *"What is the capital of Australia?"*) naturally exhibit lower token log-probabilities ($-0.07$ to $-0.47$) than formulaic arithmetic strings, causing a threshold calibrated for $\text{PGR} \ge 0.95$ to interpret natural prose variation as low confidence and escalate to the frontier. (Across the wider Track B test partition, this 1-sample bundle escalated 85.2% of queries, `notes/d15_demo_router.md`).
- **Research Disconnect:** Furthermore, this answer-text heuristic bundle was not the model certified in the thesis; the research results derived from the internal confidence and consensus signals of `cascade_3sample` (T61).

### 4.8.2 Decision D-15: Unifying the Demo with the Certified Research Router

To eliminate this discrepancy, the team ratified **Decision D-15** (`DECISIONS.md`): the production demonstration rebuilds and deploys Lane B's official T61 `cascade_3sample` research router (`--mode cascade_t61` in `scripts/11_build_bundle.py`), driven by unified confidence signals (`router/signals.py`):

1. **Multiple-Choice Prompts:** Evaluated via a single forward pass over candidate option letters using `MLXOptionScorer` (`router/mcq_mlx.py`) formatted in RouterBench's MMLU convention, extracting exact softmax probabilities (`p_chosen`, `margin`).
2. **Numeric and Arithmetic Prompts:** Evaluated by generating $k=3$ sampled completions ($T=0.7$, top-$p=0.8$) and calculating exact numeric consensus (`agree`, `n_distinct`, `lp_mean`).
3. **Conversational and Free-Form Prompts:** Evaluated via 3 brief stochastic completions and normalized text consensus (`router.signals.text_agreement_signals` in `router/signals.py`), with the user interface transparently flagging the query as uncertified.
4. **Certified Threshold and Dependency Pinning:** Operating under the pinned `scikit-learn==1.8.0` dependency with seed-free threshold $\tau^* = 8.219$, the deployed bundle escalates **75.9%** of test queries at a realized test PGR of **102.8%** (`notes/t27_t28_locked_numbers.md`, `notes/d15_demo_router.md`). (The 65.8% escalation rate observed in exploratory cloud runs was an artifact of an unpinned `scikit-learn==1.9.1` refit, resolved by strictly pinning version 1.8.0 in `requirements.txt`).

**Operational Trade-off:** Generating 3 brief samples on arithmetic queries introduces approximately $6\text{ seconds}$ of local decoding latency on an Apple Silicon M2. However, this latency trade-off ensures that multiple-choice and arithmetic queries are governed by the certified research signals, while free-form conversational queries are evaluated via text consensus and explicitly labelled uncertified on the user interface. Lane B's single-sample bundle (`--mode cascade`) remains preserved in the codebase as the D-14 comparative baseline for latency-constrained applications.

---

## 4.9 Summary and Lane Handoff

This chapter has established the feature extraction and routing foundations of the project:
1. **Leak-Free Query Feature Extraction:** Built $F_0$ heuristic, $F_1$ lexical, and $F_2$ dense embedding pipelines operating strictly on prompt text under grouped sub-split anti-leakage isolation.
2. **Calibrated Two-Head MLP Router:** Formulated a two-head neural architecture with shared 256-d GELU trunk, decoupled marginal quality heads, and multi-task capability gap loss, achieving up to $56.7\%$ relative calibration error reduction.
3. **The Information Limits of Text Routing:** Demonstrated that pre-execution query routers face an intrinsic ceiling ($9.4\%$ certified savings on Mixtral, collapsing to $0.5\% - 2.5\%$ on Qwen3.5-4B due to severe headroom compression).
4. **The Cascade Confidence Breakthrough:** Engineered generation-level consensus and probability signals that overcome text limits, unlocking **$41.3\%$ [95% CI: 39.0%, 43.5%] certified API savings** (seed-free WSR martingale) and **$27.8\%$ [95% CI: 25.9%, 29.6%]** on the single-sample baseline.
5. **Production Demo Alignment (Decision D-15):** Rebuilt the live serving bundle from the certified 3-sample research router, resolving out-of-distribution text brittleness and ensuring exact parity between academic certification and real-time execution.

The resulting routing models and certified parameters are exported to the production bundle (`router/bundle.py`, `artifacts/bundle/router.pkl`), providing the direct analytical inputs for:
- **Lane C (Chapter 5):** Comprehensive economic evaluation across multi-tier pricing models (M1, M2, Batch-API discounts) and formal Learn-then-Test statistical bounds.
- **Lane D (Chapter 6):** Local edge runtime deployment, 4-bit MLX quantization on Apple Silicon M2, sub-10ms decision latency, and interactive FastAPI web serving.
