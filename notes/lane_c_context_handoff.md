# Context & Session Handoff — Lane C (Sarthak Jakhmola)
*Last Updated: 2026-11-20 22:15 IST*
*Repository:* `https://github.com/AnirvanRanjan/cost-aware-router`  
*Current Active Working Branch:* `c/T28-lock-results` (upstream tracking set, pushed cleanly to `origin/c/T28-lock-results`)

---

## 1. Quick Status Summary

All deliverables for Lane C's major tasks (**T26, T27, T28, T37, T60, D-14, D-15, and Thesis Chapter 5**) are complete, verified, tested, committed, and pushed.

- **Current Head Commit:** `ea8720d` on branch `c/T28-lock-results`.
- **Full Test Suite:** **405 passed, 2 skipped, 0 failures** (`pytest`).
- **PR Status:** Branch `c/T28-lock-results` is pushed to GitHub. Ready to open / merge PR into `main` via:
  👉 **https://github.com/AnirvanRanjan/cost-aware-router/pull/new/c/T28-lock-results**

---

## 2. Core Numbers Locked for Thesis & Benchmarks

The official numbers are derived from release `trackb-6265adf7ffa4a602` (`artifacts/trackb/cascade_predictions.parquet`, `cascade_3sample`) on the frozen Track B test set ($N=4,707$ queries across MMLU, GSM8K, ARC-Challenge, Winogrande):

| Metric | Locked Value | 95% Confidence Interval | Notes / Source |
|---|--:|:---:|---|
| **Calibrated Threshold $\tau^*$** | **8.2187** | — | Seed-free WSR terminal averaging ($M=50$) on calibration split |
| **Calibration Escalation Rate ($\alpha_{\text{cal}}$)** | **76.1%** | — | Selected on calibration only |
| **Test Escalation Rate ($\alpha_{\text{test}}$)** | **75.9%** | [74.7%, 77.1%] | One evaluation pass on frozen test split |
| **Realized Test PGR** | **102.8%** | [99.1%, 107.0%] | Target was $\text{PGR} \ge 0.95$, $\delta = 0.05$ |
| **Model M1 Spend Savings** | **41.3%** | **[39.0%, 43.5%]** | Direct API spend reduction (1.7× cost efficiency) |
| **Model M2 Amortized Savings** | **36.5%** | [34.3%, 38.7%] | Incorporating Apple M2 amortisation and power |
| **Uncertified Test-Curve Savings** | **78.9%** | — | Post-hoc scanning on test curve at CPT 39.5% (4.7× reduction) |

### Key Hardware & Economic Assumptions (Cost Model M2):
- **Pricing:** Reference frontier rates \$3.00/1M input, \$15.00/1M output tokens (`claude-sonnet-4.6` standard).
- **M2 Hardware:** \$1,199 purchase, 3-year straight-line depreciation, 8 hours/day, 20W power, \$0.12/kWh $\to$ **\$0.139/hour operational cost**.
- **Measured Autoregressive Speeds:** Qwen3.5-4B **28.5 tok/s**, Llama 3.2 3B **41.6 tok/s** on Apple M2 (batch 1, 4-bit MLX).
- **Crossover Throughput ($v_{\text{cross}}$):** Local execution beats Claude Haiku (Batch) above **7.1 tok/s**, Gemini 3 Flash (List) above **12.1 tok/s**, and GPT-5 Nano (Batch) above **102.5 tok/s**.

---

## 3. What Was Completed & Created in This Session

### 1. Release Download & Verification (T28 / T60)
- Downloaded `cascade_predictions.parquet` from GitHub release `trackb-6265adf7ffa4a602` into `artifacts/trackb/` (SHA256: `f9d3ed0b...`).
- Ran `scripts/17_t60_seed_free_wsr.py`: confirmed $\tau^* = 8.2187$, $\alpha_{\text{test}} = 75.9\%$, $\text{PGR} = 102.8\%$, and certified savings = **41.3% [39.0%, 43.5%]**.
- Validated seed variance: sweeping single permutation seeds 0–19 yields test savings swinging wildly from **40.8% to 74.9%**, formally proving why single-permutation martingales are insufficient and justifying terminal e-value averaging over $M=50$ permutations.

### 2. Scikit-Learn Dependency & Reproducibility Note
- Documented in `notes/d15_demo_router.md` and locked numbers: the release predictions were generated with **`scikit-learn==1.8.0`** (which refits to $\tau^* = 8.2187$, 41.3% savings). Refitting with scikit-learn 1.9.1 produces $\tau^* = 9.657$ (48.3% savings). `requirements.txt` correctly pins `scikit-learn==1.8.0`.

### 3. Removed All "PREVIEW" Labels
- Modified `notes/t27_t28_locked_numbers.md`:
  - Upgraded section 2 to **Task T37: Calibrated Threshold $\tau^*$ and Pareto Curve**.
  - Upgraded section 3 to **Task T28: Locked Headline Numbers**.
  - All test-picked curve points in the T27 common-question headroom sweep table ($N=4,707$) are cleanly marked as `(uncertified)`.

### 4. Thesis Chapter 5 (`thesis/05_results.md`)
- Deleted the old working draft `notes/chapter_C_draft.md` and created the official thesis chapter at **`thesis/05_results.md`** written completely in Lane C's own voice:
  - §5.1 Problem formulation and decision rules (Rule v1 vs. Rule v2).
  - §5.2 Dual cost models (M1 direct spend, M2 device amortisation, batch baselines, crossover speed $v_{\text{cross}}$).
  - §5.3 Exact $O(n \log n)$ prefix-sum breakpoint sweep and concave Pareto envelope.
  - §5.4 Learn-then-Test (LTT) risk control and FWER guarantees.
  - §5.5 Why bootstrap undercovers (15.2% empirical failure), Hoeffding baseline, single-seed WSR permutation variance (40.8%–74.9%), and the seed-free terminal e-value mixture solution.
  - §5.6 Locked headline empirical results on Track B (41.3% certified, 78.9% test-curve).
  - §5.7 Cross-model headroom sweep on Track B questions across Mixtral, Yi-34B, and Qwen.
  - §5.8 Leave-One-Subject-Out (LOSO) domain generalization on MMLU (10.1% savings on unseen subjects).
  - §5.9 Interactive replay profiling (200 test queries, 14.3 ms median routing latency).
- Updated `thesis/README.md` to reflect Chapter 5 status.

### 5. Verified Code & Tests
- Inspected `select_tau(method="wsr_avg")` in `router/threshold.py` (ported from `17_t60_seed_free_wsr.py`).
- Ran `pytest tests/test_signals.py` (4 passed).
- Ran `pytest tests/test_threshold.py` (216 passed).
- Ran full `pytest` suite: **405 passed, 2 skipped**.

### 6. Decisions D-14 & D-15 Status
- **D-14:** Headline router is the cascade on Qwen's confidence; agreed by B, C, D (`DECISIONS.md`).
- **D-15:** Demo routes with research router (T61 `cascade_3sample`) live on M2; agreed unanimously (`A: yes 2026-11-16; B: yes; C: yes; D: yes 2026-11-19`).

### 7. Statistical Review of Chapters 1, 7, and 8
- Verified `thesis/01_introduction.md`, `thesis/07_discussion.md`, and `thesis/08_conclusion.md`:
  - Table 1.1 and Table 7.1 accurately quote the locked numbers (41.3% certified, 102.8% PGR, 36.5% M2, 78.9% test-curve).
  - The distinction between uncertified test-curve points and calibrated finite-sample bounds is rigorously and consistently stated throughout.

---

## 4. Where Everything Lives in the Codebase

- **Thesis Chapter 5:** `thesis/05_results.md`
- **Locked Benchmark Documentation:** `notes/t27_t28_locked_numbers.md`
- **Seed-Free Certification Report:** `notes/t60_certification_report.md`
- **Standalone Reproducibility Script for T60:** `scripts/17_t60_seed_free_wsr.py`
- **Official Threshold Engine:** `router/threshold.py` (`sweep`, `upper_envelope`, `select_tau(method="wsr_avg")`)
- **Cost Engine & M2 Model:** `router/costs.py` (`PriceTable`, `pair_costs`, `local_query_cost`)
- **Track B Prediction Parquet:** `artifacts/trackb/cascade_predictions.parquet`
- **Team Decisions:** `DECISIONS.md`

---

## 5. Next Steps When Resuming Work

1. **Pull Request:** Open and merge the PR for branch `c/T28-lock-results` into `main` at:  
   `https://github.com/AnirvanRanjan/cost-aware-router/pull/new/c/T28-lock-results`
2. **Sync with Main:** Once merged, run `git checkout main && git pull origin main`.
3. **Defense / Slide Prep (If Required):** Key slide talking points are the **Safety Premium Gap** (41.3% certified vs. 78.9% uncertified test curve) and the **Permutation Variance Problem** solved by terminal e-value averaging.
