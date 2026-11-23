# Lane B Viva Voce Preparation Guide

**Candidate:** Huwaid Siddiqui (Lane B: Features & Router)  
**Project:** Cost-Aware LLM Routing (Major Project, UPES Dehradun)

---

## Question 1: Why did you choose Gradient-Boosted Decision Trees (GBDTs) for the cascade router instead of an MLP or Deep Neural Network?

### The Core Answer:
We used `HistGradientBoostingRegressor` from scikit-learn for three primary technical reasons:

1. **Native Handling of Structural Missing Values (NaNs):**
   - The cascade feature space is inherently heterogeneous across task types. Multiple-choice questions have option probability signals (`p_chosen`, `margin`) but have no multiple answer completions (`agree`, `n_distinct` are NaN).
   - Conversely, GSM8K arithmetic questions have multi-sample agreement metrics (`agree`, `n_distinct`, `lp_mean`, `lp_min`) but have no multiple-choice option letters (`p_chosen`, `margin` are NaN).
   - In neural networks (MLPs), missing inputs require artificial imputation (zero-filling or mean imputation), which injects artificial bias into gradient updates. `HistGradientBoostingRegressor` natively assigns missing values to the child branch that minimizes loss at each split, allowing clean, task-conditional decision pathways without imputation artifacts.

2. **Invariance to Monotonic Feature Scaling:**
   - The cascade inputs span vastly different numerical domains:
     - `agree` $\in [0.33, 1.0]$ (discrete ratio)
     - `n_distinct` $\in \{1, 2, 3\}$ (integer counts)
     - `p_chosen` $\in [0, 1]$ (probability)
     - `lp_mean` $\in [-5.0, 0.0]$ (log probabilities)
     - `len` $\in [0.0, 6.0]$ (log token counts)
   - Decision trees split on order statistics rather than metric distances, making them completely invariant to feature scale, skewness, and non-linear monotonic transformations.

3. **Orthogonal, Axis-Aligned Decision Boundaries:**
   - Confidence-based routing naturally follows threshold conjunctions:
     *Example:* "If `agree == 1.0` AND `lp_mean > -0.05` $\implies$ keep local."
   - Decision trees learn orthogonal hyperplanes with high sample efficiency ($N=20,868$), avoiding the optimization instability and overfitting common when training deep MLPs on tabular data.

---

## Question 2: Why did adding dense BGE sentence embeddings ($F_2$) make the cascade router worse?

### The Core Answer:
In our ablation experiments (`notes/trackb_cascade_report.md`), adding dense 768-dimensional BGE embeddings to the cascade confidence features (`cascade_conf+f2_raw` and `cascade_3sample_f2_shrunk`) caused certified savings to plummet from **41.3% down to 1.1% – 3.4%**.

This counter-intuitive failure is explained by **extreme dimensionality imbalance and semantic overfitting**:

1. **Dimensionality Asymmetry ($11 \ll 779$):**
   - The clean cascade feature vector consists of only 7 confidence features (`agree`, `n_distinct`, `lp_mean`, `lp_min`, `len`, `p_chosen`, `margin`) plus 4 benchmark one-hot indicators (11 total features).
   - Concatenating 768 dense embedding dimensions means that $>98.5\%$ of the input columns represent semantic text coordinates rather than confidence indicators.

2. **Tree Splitting on Semantic Noise:**
   - When building trees on the training split (`train`, $N=20,868$), the algorithm evaluates candidate split points across all features.
   - The dense embedding dimensions capture subject matter (e.g. quantum mechanics vs ancient history). Within the finite training set, specific semantic clusters happen to correlate spuriously with higher or lower small-model error rates.
   - The tree splits chose these topical dimensions, effectively memorizing benchmark-specific subject errors instead of learning pure, domain-invariant confidence thresholds.

3. **Out-of-Sample Collapse on Test:**
   - When evaluated on the held-out test split, those spurious semantic correlations failed to generalize.
   - Under the strict risk-controlling WSR bound ($\text{PGR} \ge 0.95$, $\delta = 0.05$), the increased prediction variance forced the calibration bound to select a highly conservative threshold, escalating nearly all queries and wiping out certified cost savings.
   - **Conclusion:** Generative confidence signals directly reflect the model's epistemic uncertainty; diluting them with high-dimensional external text vectors harms generalization.

---

## Question 3: What does each of the confidence signals in `router/signals.py` actually represent?

### The Core Answer:

| Feature Name | Formulation | Operational Meaning & Empirical Threshold |
|---|---|---|
| `agree` | $\frac{\max_a \text{count}(a)}{k}$ | **Consensus Ratio across $k=3$ samples.** On GSM8K, when `agree = 1.0` (all 3 samples match), Qwen's accuracy is **96.9%**. When outputs disagree (`agree < 1.0`), accuracy drops sharply to **60.5%**. |
| `n_distinct` | $|\{a_1, \dots, a_k\}|$ | **Number of distinct numeric answers produced.** High distinct counts ($2$ or $3$) signal output instability and guessing. |
| `lp_mean` | $\frac{1}{k}\sum_{i=1}^k \left(\frac{1}{T_i}\sum_{t=1}^{T_i} \log P(w_t)\right)$ | **Average token log-probability.** Measures the local model's token-level certainty. Higher values (near $0$) indicate high model confidence in its own generation. |
| `lp_min` | $\min_{i=1}^k \left(\frac{1}{T_i}\sum_{t=1}^{T_i} \log P(w_t)\right)$ | **Worst-case token log-probability.** Detects whether any single generation path encountered severe confusion or rare token sequences. |
| `len` | $\ln(1 + \bar{n}_{\text{tokens}})$ | **Log token length.** In mathematical reasoning, long, meandering answers often correlate with circular logic, backtracking, and calculation slips. |
| `p_chosen` | $\frac{\exp(\ell_{\text{top}})}{\sum_{j} \exp(\ell_j)}$ | **Softmax probability of the top multiple-choice option letter.** On MMLU, when $p_{\text{chosen}} \ge 0.90$, Qwen's accuracy is **94.2%**; when $p_{\text{chosen}} < 0.90$, accuracy drops to **58.5%**. |
| `margin` | $p_{\text{chosen}} - p_{\text{runner-up}}$ | **Probability margin.** Measures separation between the top choice and the second-best choice. Low margin signals ambiguity between competing options. |

---

## Question 4: Why does the live production demo need 3 answers for arithmetic instead of just 1 answer?

### The Core Answer:

1. **Token Log-Probabilities Do Not Reflect Arithmetic Correctness:**
   - Under Decision D-14, we initially tested a 1-sample cascade using the mean token log-probability of a single answer.
   - However, language models generate tokens based on linguistic predictability, not mathematical truth. A model can hallucinate a fluent, formulaic, completely incorrect calculation (e.g., "$15 \times 4 = 75$") with very high token probability ($\text{lp} \approx -0.06$).
   - A single generation pass cannot distinguish a fluent calculation error from a correct calculation.

2. **Self-Consistency Directly Tests Reasoning Stability:**
   - By sampling $k=3$ completions with temperature $T=0.7$ and top-$p=0.8$, we introduce stochastic perturbation into the reasoning trajectory.
   - If the underlying mathematical derivation is sound, independent rollouts converge to the same final number (`agree = 1.0`).
   - If the model is guessing or stumbling on complex multi-step arithmetic, different rollouts follow different calculation paths and diverge (`agree = 0.33` or $0.67$).

3. **Empirical Validation Failure and Three-Part Demo Fix (Task T45 / Decision D-15):**
   - In our live demo tests (`notes/demo_set_check.md`, `notes/demo_set_check_run2.md`), the 1-sample bundle escalated **8 of 10** easy questions in run 1 and **10 of 10** easy questions in run 2, even though Qwen got all 10 right!
   - Natural conversational answers have lower token predictability than formulaic math, causing a 1-sample threshold to mistake natural prose for uncertainty.
   - We resolved this via a three-part fix:
     a. **Option Logit Scoring:** 1 forward pass on MMLU option letters instead of open-ended generation.
     b. **3-Answer Self-Consistency:** Generating 3 brief completions to check consensus for arithmetic reasoning.
     c. **Rebuilt Demo Set:** Selecting representative questions directly from the Track B test distribution rather than ad-hoc prompts.
   - Under this deployed bundle (`scikit-learn==1.8.0`), the router escalates **75.9%** of queries on the Track B test split at a realized test PGR of **102.8%**.

4. **Acceptable Latency Overhead on Apple Silicon M2:**
   - Generating 3 brief arithmetic answers takes approximately **$6\text{ seconds}$** on Apple M2 (running at 28.5 tokens/sec).
   - This 6-second delay only occurs on arithmetic queries. Multiple-choice questions execute in under **$50\text{ ms}$** via a single forward pass over option letters.

---

## Question 5: Why does `scikit-learn==1.8.0` vs `1.9.1` change the calibrated threshold $\tau^*$?

### The Core Answer:

1. **Internal Library Changes Across Minor Versions:**
   - `HistGradientBoostingRegressor` is a histogram-based gradient-boosting algorithm. Continuous numerical inputs are discretized into integer bins prior to tree building.
   - Between scikit-learn versions 1.8.0 and 1.9.1, an internal library change occurred in the histogram gradient boosting implementation.
   - Consequently, fitting the exact same model on the exact same training split with identical hyperparameters and random seeds produces slightly different tree split points across library versions. Pinning `scikit-learn==1.8.0` in `requirements.txt` eliminates this discrepancy completely across all environments.

2. **Impact on Model Predictions:**
   - Comparing the predictions on the Track B test set (`notes/d15_demo_router.md`), the release model (fitted under scikit-learn 1.8.0) and the refitted model (under 1.9.1) exhibit small predicted gap differences of up to $\pm 0.15$ on individual queries.

3. **Propagation Through Learn-then-Test (LTT) Certification:**
   - In the betting martingale calibration framework (`scripts/17_t60_seed_free_wsr.py`), the threshold sweep evaluates the sequential betting wealth process over the calibration set.
   - Small shifts in the sorted score distribution shift the exact cutoff where the empirical risk bound satisfies the $\delta = 0.05$ risk ceiling:
     - Under `scikit-learn==1.8.0`: $\tau^* = 8.2187 \implies \alpha_{\text{test}} = 75.9\% \implies \mathbf{41.3\% [39.0\%, 43.5\%]}$ **certified savings**.
     - Under `scikit-learn==1.9.1`: $\tau^* = 9.657 \implies \alpha_{\text{test}} = 65.8\% \implies \mathbf{48.3\% [46.1\%, 50.5\%]}$ **certified savings**.

4. **Engineering Resolution:**
   - Both certificates are mathematically valid for their own respective fitted models.
   - However, because the project's official frozen evaluation files (`cascade_predictions.parquet`, release `trackb-6265adf7ffa4a602`) were trained and exported under version 1.8.0, the thesis locks and quotes **41.3% certified savings**.
   - To guarantee complete reproducibility across all team laptops (MacBooks, Linux workstations, cloud instances), we strictly pinned `scikit-learn==1.8.0` in `requirements.txt`.
