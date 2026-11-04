# Chapter 4 — Features and Router Architecture

> **DRAFT for Huwaid (lane B, task T51).** Generated from `notes/ablation_ladder.md`, `notes/threshold_report.md`, `router/models.py`, `router/features.py`, and the frozen release `data-910a584c8279baa7`. Every empirical quantity reported here can be reproduced with `scripts/04_train_two_head.py`, `scripts/05_calibrate.py`, `scripts/06_run_ablation.py`, and `scripts/08_threshold_report.py`. Rewrite in your own voice and adhere to university guidelines on declaring AI assistance before final submission.

---

## 4.1 Feature Engineering

A cost-aware router must decide whether to route an incoming query $q$ to a local small model $S$ or escalate to an expensive frontier model $L$ *before* generation begins. Crucially, in production deployment or live serving, the router receives only the raw text prompt. It has no metadata indicating the source benchmark, task taxonomy, or subject label. Any reliance on benchmark identities during inference constitutes data leakage. Consequently, all features extracted by our pipeline are strictly derived from the query string $q$ alone.

We design a three-tiered hierarchical feature space—heuristic signals ($F_0$), lexical $n$-grams ($F_1$), and dense semantic representations ($F_2$)—to evaluate query complexity from distinct linguistic perspectives.

### 4.1.1 Heuristic Syntactic and Structural Features ($F_0$)

The $F_0$ feature extractor (`router.features.heuristics`) parses the raw prompt string and computes a vector of 23 scalar signals capturing length, formatting complexity, syntactic depth, and task-specific lexical triggers:

1. **Surface and Length Metrics**: `token_count` (whitespace word count), `char_count` (character length), `sentence_count` (detected via terminal punctuation `[.!?]+`), and `mean_word_length`.
2. **Character Densities**: `digit_density`, `punctuation_density` (ASCII punctuation characters), and `uppercase_density` (capturing acronyms and emphasis).
3. **Syntactic and Structural Complexity**:
   - `question_mark_count`: frequency of `?`, separating declarative instructions from interrogatives.
   - `max_bracket_depth`: maximum nesting depth of delimiters `()`, `[]`, and `{}`, identifying deeply structured mathematical formulas or nested computer code.
   - `type_token_ratio`: unique lowercased tokens divided by total tokens, indexing lexical diversity.
   - `flesch_kincaid_grade`: readability index computed as $0.39 \times \left(\frac{\text{words}}{\text{sentences}}\right) + 11.8 \times \left(\frac{\text{syllables}}{\text{words}}\right) - 15.59$, measuring textual difficulty.
4. **Formatting and Domain Markers**:
   - `code_fence_count`: markdown fences (```` or `~~~`), identifying programming problems.
   - `latex_count`: frequency of mathematical LaTeX tokens (e.g., `\frac`, `\sqrt`, `\sum`, `\int`, `\partial`, `\alpha`).
   - `url_count`: HTTP/HTTPS web links.
   - `table_row_count`: markdown table structures (`| ... |`).
   - `numeric_literal_count`: frequency of integer and floating-point literals.
5. **Task and Semantic Cues**:
   - `num_options`: multiple-choice option count extracted via regex option parsing (`router.graders.parse_options`).
   - `has_reading_passage`: binary flag identifying reading comprehension framing (e.g., *"refers to the following passage"*, *"based on the text"*).
   - `cue_word_prove`, `cue_word_derive`, `cue_word_step_by_step`, `cue_word_explain`: targeted regular expressions flagging explicit requests for multi-hop reasoning.
   - `imperative_verb_count`: frequency of instructional verbs (*"calculate"*, *"compute"*, *"solve"*, *"determine"*, *"analyze"*).

The $F_0$ extractor operates in pure Python with negligible runtime overhead ($< 0.5\text{ ms}$ per query), making it ideally suited for latency-critical deployment.

### 4.1.2 Lexical N-Gram Representations ($F_1$)

The $F_1$ feature space (`router.features.TfidfFeatureExtractor`) models lexical vocabulary through sublinear term-frequency inverse-document-frequency ($\text{TF-IDF}$) representations. To capture both domain keywords and morphological variations, $F_1$ combines:
- Word $n$-grams ($n \in \{1, 2\}$), capped at the 50,000 most frequent features;
- Character $n$-grams ($n \in \{3, 4, 5\}$), capped at 50,000 features.

**Anti-Leakage Constraint**: Under strict cross-validation protocol, the TF-IDF vocabulary and document frequencies are fitted *exclusively* on the training split prompts ($N_{\text{train}} = 20,868$). Applying `fit` to the pooled dataset would leak test-set vocabulary distributions into the feature representation, invalidating out-of-sample generalisation.

### 4.1.3 Dense Semantic Embeddings ($F_2$)

While lexical $n$-grams register surface phrasing, they fail to represent semantic intent or domain abstraction. The $F_2$ feature space (`router.features.SentenceEmbeddingExtractor`) encodes queries using the `BAAI/bge-base-en-v1.5` dense embedding model [@xiao2023bge]. Queries are tokenized and mapped to a 768-dimensional vector representation with $L_2$ normalization:
$$\mathbf{e}(q) = \frac{\text{BERT}(q)}{\|\text{BERT}(q)\|_2} \in \mathbb{R}^{768}$$
Sentence embeddings are precomputed across the entire corpus and stored in `artifacts/features/features.parquet` (SHA256: `3d5eda199c6dff272f48d158c960d88cb068feec4bbb724dfceca214c0847bf0`). Inputs exceeding the maximum sequence length (512 tokens) are logged; across the frozen 34,778-query corpus, only 148 prompts ($0.42\%$) required token truncation.

---

## 4.2 Router Architectures and Loss Formulation

A router model estimates the expected performance differential $\Delta(q) = s_L(q) - s_S(q) \in [-1, 1]$ between the frontier model $L$ and the small model $S$. Rather than optimizing standard classification accuracy, the router's primary objective is ranking quality—quantified by the Spearman rank correlation $\rho(\hat{\Delta}, \Delta)$—because downstream cost-quality thresholding operates directly on the ranked spectrum of routing scores.

```
       Query Prompt q
             │
      ┌──────┴──────┐
      ▼             ▼
   F2 (768d)    F0 (23d)
   Embedding    Heuristics (z-scored)
      └──────┬──────┘
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

### 4.2.1 Baselines: Linear Models and Gradient-Boosted Trees

We establish three reference baselines:
1. **$F_0$ Logistic Regression (`f0_logreg`)**: Logistic regression with $L_2$ regularisation ($C=1.0$) trained on the 23 $z$-score normalized heuristic features. The binary target indicates whether the frontier model strictly outperformed the small model: $y = \mathbb{I}(s_L > s_S)$. The predicted probability $P(y=1|x)$ is rescaled to $\hat{\Delta} = 2P(y=1|x) - 1$.
2. **$F_1$ Logistic Regression (`f1_logreg`)**: Identical linear formulation fitted over the high-dimensional sparse TF-IDF space (100,000 features).
3. **XGBoost Forest (`xgboost_f0_f2`)**: An ensemble of 100 gradient-boosted decision trees (maximum depth 4, learning rate 0.05, subsample ratio 0.8) trained directly on the continuous gap $\Delta = s_L - s_S$ using fused $[F_2 \parallel F_0]$ features.

### 4.2.2 The Two-Head MLP Architecture

Our headline router (`router.models.TwoHeadRouter`) implements a two-head neural network that jointly predicts the marginal success probabilities of both candidate models: $\hat{s}_S(q) \approx \mathbb{E}[s_S|q]$ and $\hat{s}_L(q) \approx \mathbb{E}[s_L|q]$.

The fused input $\mathbf{x} = [\mathbf{e}(q) \parallel \mathbf{z}(q)] \in \mathbb{R}^{791}$ concatenates the 768-dimensional $L_2$-normalized dense embedding with the 23 $z$-score normalized heuristic features. The network consists of:
1. **Shared Representation Trunk**:
   $$\mathbf{h} = \text{Dropout}_{0.1}\left(\text{GELU}\left(\mathbf{W}_{\text{trunk}} \mathbf{x} + \mathbf{b}_{\text{trunk}}\right)\right), \quad \mathbf{h} \in \mathbb{R}^{256}$$
2. **Decoupled Marginal Heads**:
   $$\hat{s}_S = \sigma\left(\mathbf{w}_S^T \mathbf{h} + b_S\right), \quad \hat{s}_L = \sigma\left(\mathbf{w}_L^T \mathbf{h} + b_L\right)$$
   where $\sigma(z) = (1 + e^{-z})^{-1}$ is the logistic sigmoid.

Decoupling the prediction into two heads provides two fundamental advantages:
- **Interpretability and Cost Normalisation**: Rather than outputting an abstract scalar difficulty index, the model explicitly outputs calibrated probabilities of success for each model, enabling query-specific cost-benefit calculations.
- **Independent Upstream Modular Retraining**: When replacing the small model (e.g., migrating from Mixtral-8x7B to Qwen3.5-4B in Track B), the frontier head $\mathbf{w}_L$ and shared trunk $\mathbf{W}_{\text{trunk}}$ can be frozen, requiring only $\mathbf{w}_S$ to be retrained on new labels.

### 4.2.3 Auxiliary Multi-Task Loss Formulation

Training separate heads with standard binary cross-entropy alone ignores the correlation between model capabilities on the same query. We train the network using a composite loss function (`router.models.two_head_loss`) that combines marginal log-likelihood with an explicit gap penalty:
$$\mathcal{L}(\theta) = \ell_{\text{BCE}}(\hat{s}_S, s_S) + \ell_{\text{BCE}}(\hat{s}_L, s_L) + \lambda_{\text{gap}} \left\| (\hat{s}_L - \hat{s}_S) - (s_L - s_S) \right\|_2^2$$
We set $\lambda_{\text{gap}} = 0.5$. The auxiliary Mean Squared Error ($\text{MSE}$) term directly penalizes errors in the predicted capability delta $\hat{\Delta} = \hat{s}_L - \hat{s}_S$, aligning gradient updates with the ranking objective.

---

## 4.3 Anti-Leakage Protocol and Probability Calibration

### 4.3.1 Grouped Sub-Splitting and Early Stopping Protocol

Neural routers trained with early stopping require validation data to terminate training before overfitting. However, standard cross-validation risks two severe forms of leakage:
1. **Calibration Leakage**: If the held-out calibration split (`cal`, $N=6,956$) is used for early stopping, it ceases to be a clean out-of-sample set for downstream conformal risk control and threshold selection.
2. **Duplicate Query Leakage**: Datasets such as MMLU and GSM8K contain thematic clusters, paraphrased variants, and shared contexts. If two related queries fall across the train and validation subsets, early stopping measures memorisation.

To eliminate leakage:
- The network is trained *strictly* on the training split (`train`, $N=20,868$).
- Early stopping carves a grouped 85/15 sub-split (`train_sub` and `val_sub`) inside `train` using `router.splits.grouped_split`. No group identifier (`group_id`) spans both partitions (`assert_no_leakage`).
- Optimization uses AdamW (learning rate $10^{-3}$, weight decay $10^{-2}$, batch size 64) with a patience of 8 epochs evaluated against validation loss on `val_sub`.
- Normalisation statistics ($\boldsymbol{\mu}, \boldsymbol{\sigma}$) are computed strictly on `train_sub`.

### 4.3.2 Non-Parametric Isotonic Calibration

Raw neural network logits passed through a sigmoid often exhibit miscalibration. To ensure that $\hat{s}_S$ and $\hat{s}_L$ correspond to empirical frequencies of correctness, we fit post-hoc isotonic regression models (`router.calibrate.TwoHeadCalibrator`, implementing `sklearn.isotonic.IsotonicRegression`) on the held-out calibration split (`cal`):
$$\tilde{s}_S = g_S(\hat{s}_S), \quad \tilde{s}_L = g_L(\hat{s}_L)$$
where $g_S$ and $g_L$ are piecewise constant, monotonically non-decreasing functions.

We evaluate calibration error using 15-bin uniform-mass Expected Calibration Error ($\text{ECE}_{15}$):
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

**Table 4.1 — Calibration performance on held-out calibration split (`scripts/05_calibrate.py`).**

| Model Head | Uncalibrated $\text{ECE}_{15}$ | Isotonic Calibrated $\text{ECE}_{15}$ | Relative Error Reduction |
|---|:---:|:---:|:---:|
| **Frontier Head ($L$: `gpt-4-1106-preview`)** | $1.71\%$ | **$0.74\%$** | **$-56.7\%$** |
| **Small Head ($S$: `mistralai/mixtral-8x7b-chat`)** | $2.88\%$ | **$1.73\%$** | **$-40.0\%$** |

On synthetic benchmarks with extreme label clustering, isotonic regression can occasionally overfit; on real multi-benchmark distributions, however, calibration reduces expected error by $40.0\%$ to $56.7\%$, guaranteeing dependable marginal probabilities for decision rules.

---

## 4.4 Empirical Results and Ablation Ladder

We benchmark router variants on the development model pair ($S = \text{Mixtral-8x7B-Chat}$, $L = \text{GPT-4-1106-Preview}$) across the frozen headline benchmark test split ($N = 6,954$ queries across MMLU, GSM8K, ARC-Challenge, and Winogrande).

### 4.4.1 The Ablation Ladder

Table 4.2 presents the stepwise ablation progression across the feature and architectural hierarchy (`notes/ablation_ladder.md`). Performance is assessed by:
- **AUC**: Area under the ROC curve for escalating when $s_L > s_S$;
- **$\rho(\hat{\Delta}, \Delta)$**: Spearman rank correlation between predicted and true capability deltas;
- **$\text{CPT}(95\%)$**: Cost-to-Pass-Through, the minimum fraction of queries routed to $L$ required to achieve a Performance Recovery Ratio ($\text{PGR}$) of at least $0.95$ on the test curve;
- **APGR**: Area under the empirical Pareto Gain Ratio curve from $0\%$ to $100\%$ frontier budget.

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

### 4.4.2 Discussion of Ablation Insights

1. **Information Content Across Tiers**:
   - $F_0$ heuristics alone achieve an AUC of $0.6776$, demonstrating that syntactic markers (length, LaTeX density, multiple-choice option structure) provide a strong baseline signal of query difficulty.
   - High-dimensional $F_1$ lexical features ($\text{AUC} = 0.6640$) perform slightly worse than $F_0$, suffering from high dimensionality and vocabulary shift between train and test.
   - Dense $F_2$ embeddings provide semantic generalization, increasing Spearman correlation from $0.1359$ ($F_0$) to $0.1585$.
   - **Fused Representation ($F_2 + F_0$)**: Concatenating dense embeddings with normalized syntactic features yields the highest ranking correlation ($\rho = 0.1719$). The dense representation identifies topic and domain, while the heuristic features register syntactic formatting constraints that embeddings smooth away.

2. **Decision Rule Comparison: Rule v1 vs Rule v2**:
   - **Rule v1 (Quality Gap Thresholding)**: Routes to $L$ iff $\hat{\Delta}(q) = \hat{s}_L(q) - \hat{s}_S(q) > \tau$. This optimizes pure capability recovery regardless of token length.
   - **Rule v2 (Cost-Normalized Gain)**: Routes to $L$ iff the expected quality gain per dollar exceeds a threshold:
     $$G(q) = \frac{\hat{s}_L(q) - \hat{s}_S(q)}{\hat{c}_L(q)} > \tau$$
     where $\hat{c}_L(q)$ estimates query expenditure based on prompt token count ($n_{\text{in}} \approx \frac{\text{len}(q)}{4}$) and frontier token pricing. Rule v2 deprioritizes verbose queries where the small model has a modest gap but escalation would consume substantial budget.

---

## 4.5 Statistical Guarantees and Certified Savings

A critical pitfall in LLM routing research is quoting "savings" directly from the test curve—selecting the threshold $\tau^*$ on test data that matches the $95\%$ performance target. Such test-picked numbers represent optimistic point estimates that degrade in deployment.

To provide rigorous, deployable guarantees, our pipeline determines $\tau^*$ on the **calibration split (`cal`)** using statistical risk-controlling bounds and applies $\tau^*$ exactly once to the untouched test split (`scripts/08_threshold_report.py`, `notes/threshold_report.md`).

### 4.5.1 Risk-Controlling Prediction Framework

Let the target Performance Gain Ratio be $\text{PGR} \ge 0.95$ with risk tolerance $\delta = 0.05$. We calibrate thresholds using three formal bounds:
1. **WSR Bound [@waudbysmith2024betting]**: Finite-sample valid concentration bound based on betting martingales, as utilized in Learn-then-Test [@angelopoulos2021ltt] and Risk-Controlling Prediction Sets [@bates2021rcps].
2. **Paired Bootstrap**: Asymptotic paired bootstrap over calibration queries with confidence level $1 - \delta$.
3. **Hoeffding Bound**: Classical distribution-free concentration bound (conservative and loose).

**Table 4.3 — Out-of-sample test results for thresholds selected on `cal` (Target $\text{PGR} \ge 95\%$, $\delta=0.05$, cost model M1).**

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

### 4.5.2 Certified Savings vs. Naive Optimism

As documented in Table 4.3:
- The naive point estimate on `cal` yields $12.7\%$ savings, but provides zero statistical guarantee.
- Under the mathematically rigorous **WSR bound**, the Two-Head MLP with Rule v2 achieves a **certified saving of $9.4\%$ [95% CI: 7.8%, 10.9%]** while strictly maintaining test PGR at $96.3\% \ge 95.0\%$.
- Under the paired bootstrap bound, certified savings reach **$11.0\%$ [9.4%, 12.7%]**.
- Hoeffding's bound is excessively conservative, demanding $97.5\%$ frontier calls and yielding only $5.6\%$ savings.

### 4.5.3 Breakdown by Target Benchmark

At the optimal certified threshold ($\tau^* = 0.0435$, selected via WSR on `two_head_mlp` v2), the routing behavior reflects distinct domain difficulties:
- **MMLU** ($N=2,808$): $96.8\%$ routed to frontier, yielding $\text{PGR} = 97.5\%$ and **$8.9\%$ monetary savings**.
- **GSM8K** ($N=1,490$): $93.8\%$ routed to frontier, yielding $\text{PGR} = 94.6\%$ and **$9.7\%$ monetary savings**.
- **ARC-Challenge** ($N=294$) and **Winogrande** ($N=253$): Both route $100\%$ to the frontier. Because these smaller test sets have higher variance and large headroom gaps ($Q_L - Q_S \approx 0.13 - 0.25$), the risk-controlling bound conservatively escalates all queries to satisfy the strict global $\delta = 0.05$ risk ceiling.

---

## 4.6 The Limits of Query-Only Routing and the Cascade Router

The empirical findings in § 4.4 and § 4.5 reveal a fundamental property of language model routing: **query-only routers face an intrinsic ceiling when routing hard reasoning problems.**

Within a given task (e.g., GSM8K grade-school arithmetic), whether a small model succeeds or fails often depends on dynamic generation factors—such as an intermediate arithmetic slip or a miscalculated subtraction—that cannot be deduced from prompt phrasing alone. Consequently, query-only routers achieve modest certified savings ($9.4\%$).

This motivates the **Cascade Router Architecture** (`router.cascade`, Proposal D-12):
1. The small model $S$ is executed locally first (incurring zero marginal API cost under model M1).
2. The router inspects both the query features ($F_0, F_2$) and the small model's generated answer features:
   - Extracted answer letters and length;
   - Chain-of-thought verification (detecting explicit arithmetic errors $\text{slip}(a \text{ op } b \ne c)$ in reasoning steps);
   - Local consensus and confidence: agreement across $k$ sampled paths (self-consistency) or next-token option probability $\max P(\text{option})$ from `MLXOptionScorer`.
3. If the small model exhibits high confidence and zero arithmetic slips, the answer is accepted; otherwise, the query escalates to $L$.

Exploratory analysis (`scripts/12_cascade_explore.py`, `notes/cascade_explore.md`) demonstrates that adding a second opinion (agreement with a second small model, Yi-34B; not Qwen's self-consistency, which is measured on Track B in Chapter 5) reduces frontier calls from $96.2\%$ to **$79.0\%$**, boosting certified savings from $9.4\%$ to **$26.0\%$ [95% CI: 24.2%, 27.9%]** at $\text{PGR} = 97.9\%$. The implementation and evaluation of the cascade router are formalized in Chapter 5.

---

## 4.7 Summary and Handoff

This chapter has established:
1. Purely text-derived feature extraction pipelines ($F_0, F_1, F_2$) with rigorous anti-leakage isolation.
2. A two-head MLP router with shared 256-d GELU trunk, decoupled marginal quality heads, and auxiliary gap loss.
3. Post-hoc isotonic calibration reducing expected calibration error by up to $56.7\%$.
4. A certified out-of-sample saving of **$9.4\%$** (WSR bound) and **$11.0\%$** (bootstrap bound) at $95\%$ PGR under cost-normalized decision rule v2.

The trained models and calibrated thresholds are exported to the project bundle (`router/bundle.py`, `artifacts/bundle/router.pkl`). They serve as direct inputs to:
- **Lane C (Chapter 5)**: Evaluating complex multi-tier pricing models (M1, M2, Batch-API discounts) and cross-dataset generalization.
- **Lane D (Chapter 6)**: Integrating the bundle into the live local inference demo with sub-15ms routing latency.
