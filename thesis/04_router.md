# Chapter 4 — Features and Router Architecture

**Author:** Huwaid Siddiqui (Lane B: Features & Router)  
**Affiliation:** Department of Computer Science & Engineering, School of Computer Science, UPES Dehradun  
**Code Artifacts:** `router/features.py`, `router/models.py`, `router/calibrate.py`, `router/signals.py`, `router/cascade.py`, `router/bundle.py`  
**Key Scripts:** `scripts/04_train_two_head.py`, `scripts/05_calibrate.py`, `scripts/06_run_ablation.py`, `scripts/08_threshold_report.py`, `scripts/11_build_bundle.py`, `scripts/17_t60_seed_free_wsr.py`

> **Declaration on AI Assistance:**  
> In compliance with UPES academic integrity standards, I declare that generative AI tools were used solely as writing and syntax refinement aids during the drafting of this chapter. All feature pipelines, neural architectures, training routines, calibration logic, and experimental analyses were designed, coded, debugged, and evaluated by the author.

---

## 4.1 Introduction and Problem Formulation

When deploying large language models, the central practical challenge is cost. Flagship cloud models such as GPT-4 deliver high accuracy across difficult tasks, but commercial API fees accumulate rapidly at scale. Conversely, modern edge models like Qwen3.5-4B can run directly on consumer laptops (such as an Apple M2 MacBook) with essentially zero marginal cost per query.

The role of a cost-aware router is to sit in front of both models and make an automatic decision for every incoming user query $q$:
- Can the local model $S$ answer this satisfactorily on its own?
- Or is the query difficult enough that we must pay to escalate it to the frontier model $L$?

Our goal is not simply to route easy questions locally, but to do so while providing a verifiable statistical guarantee: the hybrid system must preserve at least $95\%$ of the frontier model's performance advantage ($\text{PGR} \ge 0.95$).

In this project, we explored two distinct ways to route:

1. **Pre-Execution Query-Only Routing:** The router inspects only the raw text prompt $q$ before any model starts generating. This approach introduces zero initial generation delay and zero token expenditure, but it has to infer problem difficulty purely from text cues.
2. **Post-Execution Cascade Routing:** The local model $S$ generates its answer first on device. Because local execution carries zero marginal API cost under Cost Model M1, the router can inspect the generated answer and internal model confidence signals (such as option probabilities and agreement across sampled answers) before deciding whether to accept the local answer or escalate to the cloud API.

A strict rule in our implementation is that the router receives only the raw query string $q$ (and, for cascades, the local output). It never receives benchmark names, subject tags, or dataset metadata. Any router that relies on knowing whether a question came from GSM8K or MMLU at inference time would suffer from benchmark leakage. All features described below are extracted strictly from the prompt text and local generation outputs alone.

---

## 4.2 Feature Extraction from Query Text

To evaluate prompt complexity from different angles, we built a three-tier feature extraction pipeline: surface heuristics ($F_0$), lexical $n$-grams ($F_1$), and dense semantic embeddings ($F_2$).

### 4.2.1 Heuristic Syntactic and Structural Features ($F_0$)

The $F_0$ extractor (`router.features.heuristics` in `router/features.py`) parses the prompt string in pure Python and computes 23 scalar features:

1. **Basic Surface Metrics**:
   - `token_count`: Number of whitespace-separated tokens.
   - `char_count`: Total character length of the prompt.
   - `sentence_count`: Number of sentences, detected using regex terminal punctuation `[.!?]+`.
   - `mean_word_length`: Average character length per token.
2. **Character Densities**:
   - `digit_density`: Proportion of ASCII digits.
   - `punctuation_density`: Proportion of ASCII punctuation characters.
   - `uppercase_density`: Ratio of uppercase letters, which helps catch acronyms and emphasis.
3. **Syntactic Complexity**:
   - `question_mark_count`: Total count of `?`, separating direct questions from imperative instructions.
   - `max_bracket_depth`: Maximum nesting depth of delimiters `()`, `[]`, and `{}`, highlighting mathematical formulas or nested programming blocks.
   - `type_token_ratio`: Ratio of unique lowercase words to total words, measuring vocabulary variety.
   - `flesch_kincaid_grade`: Standard reading ease score: $0.39 \times \left(\frac{\text{words}}{\text{sentences}}\right) + 11.8 \times \left(\frac{\text{syllables}}{\text{words}}\right) - 15.59$.
4. **Formatting and Domain Triggers**:
   - `code_fence_count`: Markdown code block delimiters (```` or `~~~`).
   - `latex_count`: Count of mathematical LaTeX tokens (such as `\frac`, `\sqrt`, `\sum`, `\int`, `\partial`, `\alpha`).
   - `url_count`: HTTP and HTTPS links.
   - `table_row_count`: Markdown table rows marked with pipes (`|`).
   - `numeric_literal_count`: Count of numeric integer and floating-point literals.
5. **Task Indicators**:
   - `num_options`: Count of parsed multiple-choice option letters using regex extraction (`router.graders.parse_options` in `router/graders.py`).
   - `has_reading_passage`: Boolean flag checking for reading comprehension framing (e.g., *"refers to the following passage"*).
   - `cue_word_prove`, `cue_word_derive`, `cue_word_step_by_step`, `cue_word_explain`: Explicit regular expressions matching requests for multi-step reasoning.
   - `imperative_verb_count`: Frequency of instructional action verbs (*"calculate"*, *"solve"*, *"determine"*, *"analyze"*).

Because $F_0$ runs in pure Python without calling neural networks, it executes in under $0.5\text{ ms}$ per query, adding almost zero latency to inference.

### 4.2.2 Lexical N-Gram Representations ($F_1$)

The $F_1$ extractor (`router.features.TfidfFeatureExtractor` in `router/features.py`) models vocabulary using term-frequency inverse-document-frequency ($\text{TF-IDF}$) representations with sublinear term-frequency scaling. To capture both specific domain terms and morphological word endings, $F_1$ extracts:
- Word $n$-grams ($n \in \{1, 2\}$), capped at the top 50,000 features.
- Character $n$-grams ($n \in \{3, 4, 5\}$), capped at the top 50,000 features.

**Leakage Prevention:** To avoid data leakage, the TF-IDF vocabulary and document frequencies are fitted *only* on the training split prompts ($N_{\text{train}} = 20,868$). Fitting the vectorizer on the pooled corpus would expose test-set word frequencies to the router during feature extraction, which would invalidate out-of-sample claims.

### 4.2.3 Dense Semantic Embeddings ($F_2$)

While lexical $n$-grams capture exact words, they cannot recognize that two differently phrased questions share the same underlying concept. The $F_2$ extractor (`router.features.SentenceEmbeddingExtractor` in `router/features.py`) encodes prompts using the `BAAI/bge-base-en-v1.5` dense embedding model [@xiao2023bge]. Prompts are tokenized and mapped to a 768-dimensional $L_2$-normalized vector:

$$\mathbf{e}(q) = \frac{\text{BERT}(q)}{\|\text{BERT}(q)\|_2} \in \mathbb{R}^{768}$$

We precomputed these dense vectors across the entire corpus and stored them in `artifacts/features/features.parquet` (SHA256: `3d5eda199c6dff272f48d158c960d88cb068feec4bbb724dfceca214c0847bf0`). Inputs that exceeded the 512-token context window were truncated; across the 34,778 questions in the frozen corpus, only 148 prompts ($0.42\%$) required truncation.

---

## 4.3 Router Architectures and Loss Formulation

The router aims to estimate the expected capability gap between the frontier and small model:

$$\Delta(q) = s_L(q) - s_S(q) \in [-1, 1]$$

where $s_L(q), s_S(q) \in \{0, 1\}$ represent whether each model answered query $q$ correctly. Our main evaluation metric during training is ranking quality—specifically Spearman rank correlation $\rho(\hat{\Delta}, \Delta)$—because downstream cost-quality thresholding depends on sorting queries accurately from easiest to hardest.

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

### 4.3.1 Reference Baselines

We implemented three baseline models to compare against:
1. **$F_0$ Logistic Regression (`f0_logreg`):** Regularized logistic regression ($L_2$, $C=1.0$) trained on the 23 $z$-score normalized heuristic features. The target is a binary label indicating whether the frontier model was strictly better than the small model: $y = \mathbb{I}(s_L > s_S)$. Predicted probabilities are scaled to $\hat{\Delta} = 2P(y=1|x) - 1$.
2. **$F_1$ Logistic Regression (`f1_logreg`):** The same regularized linear classifier trained over the 100,000-dimensional sparse TF-IDF space.
3. **XGBoost Regressor (`xgboost_f0_f2`):** An ensemble of 100 gradient-boosted trees (max depth 4, learning rate 0.05, subsample ratio 0.8) trained directly to predict the continuous difference $\Delta = s_L - s_S$ from fused $[F_2 \parallel F_0]$ features.

### 4.3.2 The Two-Head MLP Architecture

Our primary neural router (`router.models.TwoHeadRouter` in `router/models.py`) uses a two-head architecture that estimates the success probability of each candidate model separately: $\hat{s}_S(q) \approx \mathbb{E}[s_S|q]$ and $\hat{s}_L(q) \approx \mathbb{E}[s_L|q]$.

The input vector $\mathbf{x} = [\mathbf{e}(q) \parallel \mathbf{z}(q)] \in \mathbb{R}^{791}$ fuses the 768-dimensional dense embedding with the 23 $z$-score normalized heuristic features. The network consists of:
1. **Shared Representation Trunk:**
   $$\mathbf{h} = \text{Dropout}_{0.1}\left(\text{GELU}\left(\mathbf{W}_{\text{trunk}} \mathbf{x} + \mathbf{b}_{\text{trunk}}\right)\right), \quad \mathbf{h} \in \mathbb{R}^{256}$$
2. **Separate Model Heads:**
   $$\hat{s}_S = \sigma\left(\mathbf{w}_S^T \mathbf{h} + b_S\right), \quad \hat{s}_L = \sigma\left(\mathbf{w}_L^T \mathbf{h} + b_L\right)$$
   where $\sigma(z) = (1 + e^{-z})^{-1}$ is the standard sigmoid function.

Predicting each model's success probability independently offers two practical advantages:
- **Direct Interpretability:** Instead of returning a single uncalibrated score, the network outputs meaningful probabilities of success for both models, which allows us to calculate cost-versus-quality trade-offs directly.
- **Modular Retraining:** When upgrading or changing the small model (such as switching from Mixtral-8x7B to Qwen3.5-4B in Track B, Task T36), we can freeze the shared trunk and the frontier head $\mathbf{w}_L$, retraining only the small-model head $\mathbf{w}_S$ on the new labels.

### 4.3.3 Auxiliary Multi-Task Gap Loss

Training two heads solely with standard binary cross-entropy ignores how model performances correlate on the same question. We trained the network with a combined loss function (`router.models.two_head_loss` in `router/models.py`):

$$\mathcal{L}(\theta) = \ell_{\text{BCE}}(\hat{s}_S, s_S) + \ell_{\text{BCE}}(\hat{s}_L, s_L) + \lambda_{\text{gap}} \left\| (\hat{s}_L - \hat{s}_S) - (s_L - s_S) \right\|_2^2$$

We set $\lambda_{\text{gap}} = 0.5$. The added Mean Squared Error term directly penalizes errors in the predicted difference $\hat{\Delta} = \hat{s}_L - \hat{s}_S$, aligning training updates with our ranking goal.

---

## 4.4 Anti-Leakage Protocol and Probability Calibration

### 4.4.1 Grouped Sub-Splitting and Early Stopping

Neural networks need early stopping on validation data to prevent overfitting. Standard random splitting risks two subtle forms of leakage:
1. **Calibration Leakage:** If the held-out calibration split (`cal`, $N=6,956$) is used to guide early stopping, it is no longer an untouched sample for calculating statistical guarantees later.
2. **Duplicate Query Leakage:** Benchmark datasets often include duplicate queries, minor paraphrases, or questions sharing the same reading context. If these variants split across training and validation sets, validation loss measures memorization instead of real generalization.

To guarantee clean separation:
- The network is trained *exclusively* on the designated training partition (`train`, $N=20,868$).
- Early stopping splits the training data into an 85/15 grouped sub-split (`train_sub` and `val_sub`) using `router.splits.grouped_split` (`router/splits.py`). Queries with the same group identifier (`group_id`) never cross partitions (`assert_no_leakage`).
- We optimize using AdamW (learning rate $10^{-3}$, weight decay $10^{-2}$, batch size 64) with an early stopping patience of 8 epochs evaluated on `val_sub`.
- Normalization parameters ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) are calculated strictly on `train_sub`.

### 4.4.2 Non-Parametric Isotonic Calibration

Sigmoid outputs from neural networks often suffer from miscalibration. To ensure that $\hat{s}_S$ and $\hat{s}_L$ reflect true empirical probabilities of success, we apply post-hoc isotonic regression (`router.calibrate.TwoHeadCalibrator`, using `sklearn.isotonic.IsotonicRegression` in `router/calibrate.py`) fitted on the held-out calibration split:

$$\tilde{s}_S = g_S(\hat{s}_S), \quad \tilde{s}_L = g_L(\hat{s}_L)$$

where $g_S$ and $g_L$ are piecewise constant, monotonic non-decreasing functions. We evaluate calibration error using 15-bin uniform-mass Expected Calibration Error ($\text{ECE}_{15}$):

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

**Table 4.1 — Calibration performance on held-out calibration split (`scripts/05_calibrate.py`).**

| Model Head | Uncalibrated $\text{ECE}_{15}$ | Isotonic Calibrated $\text{ECE}_{15}$ | Relative Error Reduction |
|---|:---:|:---:|:---:|
| **Frontier Head ($L$: `gpt-4-1106-preview`)** | $1.71\%$ | **$0.74\%$** | **$-56.7\%$** |
| **Small Head ($S$: `mistralai/mixtral-8x7b-chat`)** | $2.88\%$ | **$1.73\%$** | **$-40.0\%$** |

On real multi-benchmark data, isotonic regression reduces expected calibration error by $40.0\%$ to $56.7\%$, producing dependable marginal probabilities for routing decisions.

---

## 4.5 Empirical Evaluation and Ablation Ladder

We benchmarked router architectures on the development model pair ($S = \text{Mixtral-8x7B-Chat}$, $L = \text{GPT-4-1106-Preview}$) across the frozen test split ($N = 6,954$ queries across MMLU, GSM8K, ARC-Challenge, and Winogrande).

### 4.5.1 The Core Ablation Ladder

Table 4.2 summarizes the stepwise ablation ladder (`notes/ablation_ladder.md`). We measure:
- **AUC:** Area under the ROC curve for escalating when $s_L > s_S$;
- **$\rho(\hat{\Delta}, \Delta)$:** Spearman rank correlation between predicted and actual performance deltas;
- **$\text{CPT}(95\%)$:** Cost-to-Pass-Through, the minimum fraction of queries sent to $L$ to achieve $\text{PGR} \ge 0.95$ on the test curve;
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

### 4.5.2 Key Takeaways from the Ablation

1. **Feature Synergies:**
   - $F_0$ heuristics alone reach an AUC of $0.6776$, showing that surface markers (character length, LaTeX usage, option formatting) provide an effective initial baseline of task difficulty.
   - High-dimensional $F_1$ lexical features ($\text{AUC} = 0.6640$) perform slightly below $F_0$, mainly due to feature sparsity and vocabulary shifts across benchmarks.
   - Dense $F_2$ embeddings capture semantic subject matter, boosting Spearman correlation from $0.1359$ ($F_0$) to $0.1585$.
   - **Fused Representation ($F_2 + F_0$):** Combining dense embeddings with normalized syntactic features yields the strongest ranking correlation ($\rho = 0.1719$). Dense embeddings detect domain topics, while heuristic counts preserve exact structural constraints that embeddings tend to smooth out.

2. **Comparing Decision Rules:**
   - **Rule v1 (Quality Gap Thresholding):** Escalates to $L$ whenever $\hat{\Delta}(q) = \hat{s}_L(q) - \hat{s}_S(q) > \tau$, focusing purely on accuracy gain regardless of token length.
   - **Rule v2 (Cost-Normalized Gain):** Escalates to $L$ whenever expected performance gain per dollar exceeds a threshold:
     $$G(q) = \frac{\hat{s}_L(q) - \hat{s}_S(q)}{\hat{c}_L(q)} > \tau$$
     where $\hat{c}_L(q)$ estimates query cost from prompt token count ($n_{\text{in}} \approx \frac{\text{len}(q)}{4}$) and frontier token prices. Rule v2 avoids spending budget on verbose prompts where the small model is only slightly worse than the frontier.

---

## 4.6 Statistical Guarantees and Certified Savings

A common problem in published routing literature is picking a threshold directly on the test set—choosing the cutoff $\tau^*$ on test data that hits $95\%$ PGR. Numbers selected this way are overly optimistic point estimates that degrade when deployed on new traffic.

To establish genuine, deployable guarantees, our pipeline chooses $\tau^*$ on the **calibration split (`cal`)** using statistical risk-controlling bounds, and then evaluates that fixed cutoff exactly once on the untouched test split (`scripts/08_threshold_report.py`, `notes/threshold_report.md`).

### 4.6.1 The Risk-Controlling Framework

To guarantee $\text{PGR} \ge 0.95$ with risk tolerance $\delta = 0.05$, we evaluated three calibration bounds:
1. **WSR Bound [@waudbysmith2024betting]:** A finite-sample valid concentration bound based on betting martingales, as used in Learn-then-Test [@angelopoulos2021ltt] and Risk-Controlling Prediction Sets [@bates2021rcps].
2. **Paired Bootstrap:** An asymptotic paired bootstrap over calibration queries at a $95\%$ confidence level.
3. **Hoeffding Bound:** Classical distribution-free concentration bound (very conservative).

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

### 4.6.2 Certified Savings vs. Naive Point Estimates

As shown in Table 4.3:
- Choosing a cutoff via the naive sample mean on `cal` suggests $12.7\%$ savings, but provides zero statistical guarantee.
- Under the mathematically valid **WSR bound**, the Two-Head MLP with Rule v2 delivers **$9.4\%$ [95% CI: 7.8%, 10.9%] certified savings** while strictly maintaining test PGR at $96.3\% \ge 95.0\%$.
- Under the paired bootstrap bound, savings reach **$11.0\%$ [9.4%, 12.7%]**.
- Hoeffding's bound is excessively conservative, sending $97.5\%$ of traffic to the frontier and leaving only $5.6\%$ savings.

### 4.6.3 Domain Breakdown

At the optimal certified threshold ($\tau^* = 0.0435$, selected via WSR on `two_head_mlp` v2), routing behavior reflects benchmark difficulty:
- **MMLU** ($N=2,808$): $96.8\%$ routed to frontier, giving $\text{PGR} = 97.5\%$ and **$8.9\%$ spend savings**.
- **GSM8K** ($N=1,490$): $93.8\%$ routed to frontier, giving $\text{PGR} = 94.6\%$ and **$9.7\%$ spend savings**.
- **ARC-Challenge** ($N=294$) and **Winogrande** ($N=253$): Both route $100\%$ to the frontier. Because these smaller subsets have higher statistical variance and wider capability gaps ($Q_L - Q_S \approx 0.13 - 0.25$), the risk-controlling bound conservatively escalates all queries to satisfy the strict global risk bound $\delta = 0.05$.

---

## 4.7 Headroom Compression and the Cascade Router Architecture

Our findings on the Mixtral development pair revealed a fundamental practical limitation: **query-only routers hit a ceiling when evaluating multi-step reasoning problems.**

In tasks like grade-school math (GSM8K), whether an edge model succeeds or fails depends on dynamic execution details—such as dropping a minus sign or making an arithmetic slip halfway through the solution. These calculation slips simply cannot be predicted from prompt wording alone. As a result, query-only routing maxed out at modest certified savings ($9.4\%$).

### 4.7.1 The Headroom Compression Problem on Track B

This limitation becomes much more pronounced when moving to stronger edge models. On our headline deployment pair (Track B: $S = \text{Qwen3.5-4B}$, $L = \text{GPT-4-1106-Preview}$), the small model achieves **$81.3\%$** baseline accuracy against **$88.7\%$** for the frontier model (`notes/t27_t28_locked_numbers.md`). The capability headroom contracts from $0.237$ (Mixtral) down to just **$0.075$** ($\Delta = 0.0748$).

Under such narrow headroom, query-only routers struggle: the certified savings of the feature-based query router (`query_f2`, Rule v2, WSR) reach only **$2.5\%$ [95% CI: 1.8%, 3.1%]** under single-seed WSR and **$2.1\%$ [95% CI: 1.5%, 2.7%]** seed-free at test PGR $98.6\%$ (`notes/trackb_cascade_report.md`, Chapter 5 Table 5.1). The Two-Head MLP achieves only **$0.5\%$ [95% CI: 0.3%, 0.7%]** (`notes/qwen_query_report.md`). Prompt text alone simply does not carry enough signal to reliably separate the few questions Qwen fails on from those it solves correctly.

### 4.7.2 The Cascade Formulation and Generation Confidence Signals

To overcome this, we developed the **Cascade Router Architecture** (`router/cascade.py`, Decisions D-12 and D-14, both agreed by the team):
1. The local small model $S$ generates an answer first (costing zero API dollars under Cost Model M1).
2. The router extracts internal confidence and consensus signals directly from the local generation outputs (`router/signals.py`):
   - **Multiple-Choice Questions:** A single forward pass over candidate option letters using `MLXOptionScorer` (`router/mcq_mlx.py`) produces exact softmax option probabilities:
     - `p_chosen`: Probability assigned to the top option letter.
     - `margin`: Difference between the top option's probability and the runner-up.
   - **Numeric and Arithmetic Questions (GSM8K):** $k$ stochastic completions ($T=0.7$, top-$p=0.8$) provide consensus metrics:
     - `agree`: Fraction of the $k$ sampled answers matching the majority output.
     - `n_distinct`: Number of unique numerical answers produced.
     - `lp_mean`: Average of the completions' mean token log-probabilities.
     - `lp_min`: Lowest mean log-probability across completions.
     - `len`: Log-transformed mean token length $\ln(1 + \bar{n}_{\text{tokens}})$.
3. If the local model shows high consensus and confident log-probabilities, the answer is accepted locally; otherwise, the query escalates to the cloud frontier API.

Our early tests on the Mixtral pair (`scripts/12_cascade_explore.py`, `notes/cascade_explore.md`) showed that adding a second opinion from Yi-34B (cross-model agreement between Mixtral-8x7B and Yi-34B-Chat, not self-consistency) boosted certified savings from $9.4\%$ to **$26.0\%$ [95% CI: 24.2%, 27.9%]** at $\text{PGR} = 97.9\%$. On Track B, Qwen's own agreement across $k=3$ samples provides a practical, single-model equivalent that runs on consumer hardware.

### 4.7.3 Track B Certified Results and Seed-Free Martingale Averaging

On the frozen Track B test set ($N=4,707$ queries, release `trackb-6265adf7ffa4a602`), we evaluated gradient-boosted decision trees trained on these internal signals:

1. **Permutation Variance in Sequential Martingales:** In standard Learn-then-Test, the sequential WSR betting martingale depends on the random ordering of calibration data. Sweeping permutation seeds $0$ through $19$ caused certified test savings to fluctuate between **$40.8\%$ and $74.9\%$** (`notes/t60_certification_report.md`). A single random seed produced **$40.8\%$ [95% CI: 38.7%, 43.0%]** at $\tau^* = 8.1093$.
2. **Seed-Free Certification via Terminal E-Value Averaging:** To remove ordering noise without weakening statistical guarantees, Lane C implemented a seed-free martingale averaging procedure (`scripts/17_t60_seed_free_wsr.py`). By taking the arithmetic mean of terminal martingale wealth across $M=50$ random permutations:
   $$\bar{E} = \frac{1}{M} \sum_{m=1}^M E^{(m)}$$
   linearity of expectation guarantees that $\bar{E}$ remains an exact finite-sample e-value. Inverting this combined test yields a deterministic threshold ($\tau^* = 8.2187$), certifying **$41.3\%$ [95% CI: 39.0%, 43.5%] API savings** at an empirical test PGR of **$102.8\%$ [95% CI: 99.1%, 107.0%]** (escalating $75.9\%$ of queries).
3. **Single-Sample Cascade Baseline (Decision D-14):** In latency-sensitive settings where sampling three answers is too slow, the 1-sample cascade (`cascade_1sample`, Rule v2) evaluates only one completion's mean log-probability (or option logit).
   - When scored against the majority-of-3 label as recorded in our original evaluation report, it certifies at **$27.8\%$ [95% CI: 25.9%, 29.6%]** savings (quoted in Chapter 5 Table 5.1).
   - When scored strictly on the first generated answer alone—which matches what a live 1-sample user actually receives—the seed-free bound yields **$29.5\%$ [95% CI: 27.8%, 31.5%]** certified savings. Both numbers reflect the same confidence cutoff, confirming that single-sample confidence retains the majority of the cascade advantage.
4. **Why Combining Embeddings Hurt Performance:** Re-introducing dense $F_2$ embeddings into the cascade feature vector (`cascade_3sample_f2_shrunk` and `cascade_conf+f2_raw`) degraded certified savings down to **$1.1\% - 3.4\%$** (lowest was `cascade_conf+f2_raw` at $1.1\%$, `notes/trackb_cascade_report.md`). High-dimensional dense vectors caused tree splits during training on `train` to latch onto spurious lexical correlations across benchmarks, diluting the clear confidence signals provided by generation agreement and token likelihoods.

---

## 4.8 The Production Demonstration Router (Decision D-15)

Translating these models to the interactive on-device demonstration (`router/serve.py`) required an important design correction between Decision D-14 and Decision D-15.

### 4.8.1 The Failure Mode of Answer-Text Heuristics

Decision D-14 initially proposed packaging a single-sample bundle (`router/bundle.py`, `--mode cascade`) that extracted heuristic signals from Qwen's answer *text*—including regexes for arithmetic checks, multiple-choice option parsing, and targeted hedge words checked by the codebase (`assume`, `assuming`, `approximately`, `not sure`, `unclear`, `cannot`, `can't`, `however`, `wait`, `actually` in `router/cascade.py`)—combined with one answer's mean token log-probability.

When we tested this text-heuristic bundle on our interactive demo suite (Task T45, `notes/demo_set_check.md`, `notes/demo_set_check_run2.md`), it encountered severe distribution shift on conversational and free-form prompts:
- **Validation Run 1 (Raw Prompt):** Qwen answered all 10 easy questions correctly, but the router escalated **8 out of the 10 easy questions** to the frontier API.
- **Validation Run 2 (Brief Prompt Template):** All **10 out of 10 easy questions** were escalated, despite Qwen answering every single one correctly.
- **Root Cause:** Free-form conversational answers (such as *"What is the capital of Australia?"*) naturally have lower token log-probabilities ($-0.07$ to $-0.47$) than formulaic arithmetic steps. Because natural language is inherently less predictable than math tokens, the threshold calibrated for $\text{PGR} \ge 0.95$ interpreted natural prose variation as uncertainty and escalated. Across the broader Track B test split, this 1-sample bundle escalated **$85.2\%$ of queries** (scikit-learn 1.9.1 refit, `notes/d15_demo_router.md`).
- **Research Disconnect:** This text-heuristic model was not the model certified in the thesis; the research numbers came from the confidence and consensus signals of `cascade_3sample` (T61).

### 4.8.2 Decision D-15: Unifying the Demo with the Certified Research Router

To resolve this issue, the team ratified **Decision D-15** (`DECISIONS.md`): the production demonstration rebuilds and runs Lane B's official T61 `cascade_3sample` research router (`--mode cascade_t61` in `scripts/11_build_bundle.py`), powered by unified confidence signals (`router/signals.py`):

1. **Multiple-Choice Questions:** Evaluated via a single forward pass over candidate option letters using `MLXOptionScorer` (`router/mcq_mlx.py`) formatted in RouterBench's MMLU convention, extracting exact softmax probabilities (`p_chosen`, `margin`).
2. **Numeric and Arithmetic Questions:** Evaluated by generating $k=3$ completions ($T=0.7$, top-$p=0.8$) and computing exact numeric consensus (`agree`, `n_distinct`, `lp_mean`).
3. **Conversational and Free-Form Questions:** Evaluated via 3 brief stochastic completions and normalized text consensus (`router.signals.text_agreement_signals` in `router/signals.py`), with the user interface transparently flagging the query as uncertified.
4. **Certified Threshold and Dependency Pinning:** Operating under pinned `scikit-learn==1.8.0` with seed-free cutoff $\tau^* = 8.219$, the deployed bundle escalates **$75.9\%$** of test queries at a realized test PGR of **$102.8\%$** (`notes/t27_t28_locked_numbers.md`, `notes/d15_demo_router.md`). (The 65.8% rate observed in exploratory cloud tests was an artifact of an unpinned `scikit-learn==1.9.1` refit, which was resolved by locking version 1.8.0 in `requirements.txt`).

**Operational Trade-off:** Generating 3 brief completions for arithmetic problems adds roughly $6\text{ seconds}$ of local decoding time on an Apple M2. However, this latency trade-off ensures that multiple-choice and arithmetic questions follow the certified research signals, while conversational prompts are evaluated cleanly via text consensus and labelled as uncertified in the UI. Lane B's 1-sample bundle (`--mode cascade`) remains available in the codebase as the D-14 comparative baseline for latency-constrained systems.

---

## 4.9 Summary and Lane Handoff

In summary, Lane B established the feature extraction and routing foundations for the project:
1. **Leak-Free Query Features:** Built $F_0$ heuristic, $F_1$ lexical, and $F_2$ dense embedding pipelines operating strictly on prompt text under grouped sub-split anti-leakage isolation.
2. **Calibrated Two-Head MLP Router:** Designed a two-head neural network with a shared 256-d GELU trunk, decoupled marginal quality heads, and multi-task capability gap loss, reducing expected calibration error by up to $56.7\%$.
3. **Information Limits of Text Routing:** Proved that pre-execution text routers hit a hard ceiling ($9.4\%$ certified savings on Mixtral, falling to $0.5\% - 2.5\%$ on Qwen3.5-4B due to compressed headroom).
4. **Cascade Confidence Architecture:** Implemented generation consensus and probability signals that overcome text limits, delivering **$41.3\%$ [95% CI: 39.0%, 43.5%] certified API savings** (seed-free WSR martingale) and **$27.8\%$ - $29.5\%$** on the single-sample baseline.
5. **Production Demo Alignment (Decision D-15):** Rebuilt the serving bundle from the certified 3-sample research router, fixing out-of-distribution text brittleness and aligning the live demonstration directly with our formal guarantees.

The resulting routing models and certified parameters are packaged into the production bundle (`router/bundle.py`, `artifacts/bundle/router.pkl`), providing the direct analytical inputs for:
- **Lane C (Chapter 5):** Multi-tier economic cost accounting (M1, M2, Batch-API discounts) and formal Learn-then-Test statistical bounds.
- **Lane D (Chapter 6):** Edge runtime deployment, 4-bit MLX quantization on Apple Silicon M2, sub-10ms decision latency, and interactive FastAPI serving.
