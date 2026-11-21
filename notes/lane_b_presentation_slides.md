# Lane B Presentation Slides: Features & Router Architecture

**Presenter:** Huwaid Siddiqui (Lane B: Features & Router)  
**Project:** Cost-Aware LLM Routing (Major Project, UPES Dehradun)  
**Audience:** Faculty Evaluation Panel, External Examiners, Team Defense

---

## Slide 1: Title & Executive Overview
- **Title:** Engineering a Cost-Aware Model Router: From Query Embeddings to Cascade Confidence
- **Presenter:** Huwaid Siddiqui (Lane B Lead — Features, Model Architecture & Calibration)
- **Core Challenge:** Frontier models (GPT-4) cost \$3.00 / \$15.00 per million tokens. Edge models (Qwen3.5-4B) run for \$0 on Apple Silicon laptops.
- **Lane B Objective:** Build a low-latency routing policy that decides when to run locally vs. escalate to the cloud, guaranteeing at least $95\%$ performance gap recovered ($\text{PGR} \ge 0.95$).
- **Key Takeaway:** Pre-execution text routing hits a ceiling as small models improve ($2.5\%$ savings on Qwen). Inspecting post-execution self-consistency and logit confidence unlocks **41.3% certified spend reduction**.

---

## Slide 2: Feature Engineering — The Three-Tier Feature Space
- **Strict Anti-Leakage Rule:** The router only sees prompt text $q$ (no benchmark names, subject labels, or dataset tags).
- **Tier 1: Heuristic Syntactic Signals ($F_0$, 23 dimensions)**
  - Surface: token count, character count, sentence count, mean word length.
  - Structure: bracket nesting depth (code/math nesting), ASCII digit/punctuation/uppercase densities.
  - Task markers: parsed MCQ option count, LaTeX commands (`\frac`, `\sqrt`), code fences, reasoning triggers (`prove`, `derive`, `step_by_step`).
  - *Speed:* Pure Python execution in $< 0.5\text{ ms}$.
- **Tier 2: Lexical N-Grams ($F_1$, 100,000 dimensions)**
  - Sublinear TF-IDF combining word $n$-grams (1, 2) and character $n$-grams (3, 4, 5).
  - *Anti-Leakage Guard:* Vectorizer fitted strictly on training split ($N=20,868$), never on pooled or test data.
- **Tier 3: Dense Semantic Embeddings ($F_2$, 768 dimensions)**
  - Pre-trained `BAAI/bge-base-en-v1.5` embeddings ($L_2$-normalized).
  - Encodes conceptual intent beyond literal keyword matches.

---

## Slide 3: Two-Head MLP Router & Multi-Task Loss Formulation
- **Fused Input:** $\mathbf{x} = [F_2 \parallel F_0] \in \mathbb{R}^{791}$ (768-d semantic + 23-d normalized syntactic).
- **Network Architecture:**
  - Shared Trunk: Linear(791 $\to$ 256) $\to$ GELU $\to$ Dropout(0.1).
  - Decoupled Heads: $\hat{s}_S = \sigma(\mathbf{w}_S^T \mathbf{h} + b_S)$ and $\hat{s}_L = \sigma(\mathbf{w}_L^T \mathbf{h} + b_L)$.
- **Why Decouple Heads?**
  1. *Interpretability:* Directly outputs estimated success probabilities for both models, enabling cost-benefit normalization (Rule v2).
  2. *Modular Retraining:* Upgrading the local model (Mixtral $\to$ Qwen) requires freezing the trunk and frontier head and retraining only the small-model head.
- **Multi-Task Composite Loss:**
  $$\mathcal{L} = \ell_{\text{BCE}}(\hat{s}_S, s_S) + \ell_{\text{BCE}}(\hat{s}_L, s_L) + 0.5 \left\| (\hat{s}_L - \hat{s}_S) - (s_L - s_S) \right\|_2^2$$
  The auxiliary MSE term directly optimizes ranking quality ($\rho(\hat{\Delta}, \Delta)$).
- **Calibration:** Non-parametric isotonic regression on held-out `cal` split cuts ECE by $56.7\%$ ($L$) and $40.0\%$ ($S$).

---

## Slide 4: Why Query-Only Routing Hits a Ceiling (The Headroom Collapse)
- **Track A (Mixtral-8x7B vs GPT-4):**
  - Baseline accuracies: Mixtral $65.0\%$, GPT-4 $88.7\%$ $\implies$ Headroom $\Delta = 0.237$.
  - Two-Head MLP achieves **$9.4\%$ certified savings** under WSR bound at $\text{PGR} \ge 95\%$.
- **Track B (Qwen3.5-4B vs GPT-4):**
  - Baseline accuracies: Qwen $81.3\%$, GPT-4 $88.7\%$ $\implies$ Headroom contracts to **$\Delta = 0.0748$**.
  - Query-only router collapses to **$2.5\%$ [1.8, 3.1]** certified savings (`query_f2`) and **$0.5\%$** (`two_head_mlp`).
- **Why Does Text-Only Routing Fail?**
  - Prompt semantics do not predict execution mistakes. Two GSM8K math questions with identical phrasing, length, and difficulty might result in one correct answer and one dropped negative sign in step 3.
  - Under narrow headroom ($0.075$), misclassifying just a few queries violates the risk floor ($\delta = 0.05$).

---

## Slide 5: The Cascade Router — Routing on Internal Confidence
- **Core Principle:** Execute small model $S$ locally first (zero marginal API cost under M1), then inspect generation dynamics.
- **Internal Signal Extraction (`router/signals.py`):**
  - *Multiple-Choice (MCQ):* 1 forward logit pass via `MLXOptionScorer` $\implies$ `p_chosen` (top softmax probability) and `margin` (lead over runner-up).
  - *Grade-School Math (GSM8K):* $k=3$ stochastic samples ($T=0.7$, top-$p=0.8$) $\implies$ `agree` (majority consensus), `n_distinct`, `lp_mean`, `lp_min`, `len`.
- **Model:** Gradient-Boosted Decision Trees (`HistGradientBoostingRegressor`).
- **Certified Results on Track B ($N=4,707$ test queries):**
  - **Seed-Free WSR ($M=50$):** **41.3% [39.0%, 43.5%] certified API savings** at realized test PGR of **102.8%** ($\tau^* = 8.2187$, escalating $75.9\%$).
  - Eliminates the $40.8\%–74.9\%$ seed variance of single-permutation martingales.
  - **1-Sample Cascade Baseline (Decision D-14):** **27.8% [25.9, 29.6]** certified savings at $97.7\%$ PGR.
- **Counter-Intuitive Finding:** Adding $F_2$ dense embeddings *degraded* cascade savings to $1.1\% - 3.4\%$. 768 continuous embedding dimensions overwhelmed tree splits on the 11 confidence features, fitting topic noise rather than uncertainty.

---

## Slide 6: The Production Demo Router (Decision D-15)
- **The Empirical Problem with D-14:**
  - D-14 extracted heuristic signals from answer text (arithmetic regexes, hedge words `assume`, `approximate`, `cannot`, etc., and single token log-probs).
  - In live interactive tests on colloquial queries, it escalated **8/10 (run 1)** and **10/10 (run 2)** easy questions that Qwen got right. Free-form text naturally has lower token log-probs than formulaic math, triggering false escalation.
- **Decision D-15: Unifying the Demo with the Certified Research Model**
  - Rebuilt production bundle from Lane B's T61 `cascade_3sample` research router (`--mode cascade_t61` in `scripts/11_build_bundle.py`).
  - *MCQ:* 1 forward pass in MMLU format (`p_chosen`, `margin`).
  - *Arithmetic:* 3 brief sampled completions and exact consensus.
  - *Free-form:* 3 brief completions and text consensus (transparently flagged uncertified in UI).
- **Real-World Trade-off:** Adds ~6 seconds local generation time on Apple M2 for math, but escalates only $65.8\%–75.9\%$ of test traffic (PGR $100.6\%–102.8\%$), matching certified research guarantees.
- **Software Reproducibility:** Strictly pinned `scikit-learn==1.8.0` in `requirements.txt` to guarantee tree split reproducibility across developer laptops.

---

## Slide 7: Summary & Lane B Deliverables
1. **Leak-Free Pipelines:** Full $F_0, F_1, F_2$ extractors operating strictly on input text without benchmark labels.
2. **Calibrated Two-Head MLP:** Multi-task gap loss and isotonic calibration cutting ECE by up to $56.7\%$.
3. **Identified Information Limits:** Proved mathematically and empirically why query-only routers hit a ceiling as models improve.
4. **Cascade Confidence Breakthrough:** 3-sample self-consistency + option logit scoring unlocked **41.3% certified spend savings**.
5. **Live Production Deployment:** Delivered the verified D-15 bundle running sub-10ms routing decisions on Apple Silicon M2.
