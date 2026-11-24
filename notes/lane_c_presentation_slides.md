# Lane C — presentation slides (T56)

*Draft for Sarthak (lane C) to cut down and speak from, same format as lanes A, B and D. Every number
carries the file it comes from; keep the citation when a number moves onto a slide. Target: ~10
minutes, so 10 slides. The story: why a threshold picked on test data is not a guarantee, and how
we got one.*

---

## Slide 1 — The question lane C answers

- The router outputs a score; above τ the question goes to the frontier.
- **Which τ?** Too low: we pay for everything. Too high: quality drops.
- Requirement: PGR ≥ **0.95** with probability ≥ **95%** (δ = 0.05), with τ chosen **before** the
  test split is opened.

## Slide 2 — The decision rule (rule v2)

- Rule v1: escalate if Δ̂ = ŝ_L − ŝ_S > τ.
- **Rule v2:** escalate if Δ̂ / ĉ_L > τ — quality gained **per dollar** of frontier cost; long,
  expensive questions need a bigger expected gain to justify the call (`thesis/05_results.md` §5.1.1).
- Costs come from lane A's M1/M2 models (`router/costs.py`).

## Slide 3 — Looking at the test curve is not allowed

- Exact breakpoint sweep over every threshold, O(n log n), plus the concave envelope (§5.3).
- Reading the operating point off the **test** curve: **78.9%** savings at 39.5% frontier calls.
- Picking τ naively on cal: **77.0%** at test PGR **97.2%** — no guarantee it stays above 95%.
- Both are optimistic: the threshold has seen the data it is judged on (Table 5.1).

## Slide 4 — Learn-then-Test

- Per question, the paired margin e_i(τ) = s_route(i) − [(1 − ρ) s_S,i + ρ s_L,i], ρ = 0.95.
- PGR ≥ ρ ⇔ E[e(τ)] ≥ 0 → one hypothesis test per τ on the **cal** split.
- Fixed sequence from the safest τ outward: stop at the first τ that fails → **no Bonferroni
  correction** needed (§5.4.2).

## Slide 5 — Which bound? The bootstrap fails here

- The margin is **0** on 70–80% of questions (both models right or both wrong), and the headroom
  is small (Q_L − Q_S = **0.075** for Qwen).
- Monte Carlo on n_cal = 2,000: the nominal 95% bootstrap bound misses the true value up to
  **15.2%** of the time — 3× the allowed 5% (§5.5.1).
- Hoeffding is valid but ignores variance and throws away most of the savings (§5.5.2).
- **WSR betting martingale**: valid at every sample size and uses the variance.

## Slide 6 — WSR depends on the order of the questions

- The betting martingale processes questions in a random order. Same data, seeds 0–19:
  τ* from **8.11 to 16.72**, test savings from **40.8% to 74.9%** (`notes/t60_certification_report.md`, T60).
- Each seed is a valid certificate — but choosing the seed you like is not.

## Slide 7 — The fix: average the evidence, not the seed

- For a fixed order, the final wealth E^(m) is an **e-value**: E[E^(m)] ≤ 1 under the null.
- The mean of e-values over M orders is still an e-value → p = min(1, 1 / Ē) is valid (Markov).
- M = 50 orders → one τ*, no seed: **8.2187** (`scripts/17_t60_seed_free_wsr.py`; ported to
  `router/threshold.py` as `method="wsr_avg"`, identical p-values).

## Slide 8 — Headline result (Table 5.1)

| Qwen3.5-4B vs GPT-4, test n = 4,707 | τ* | Frontier calls | PGR | M1 saved | M2 saved |
|---|--:|--:|--:|--:|--:|
| Question-only (`query_f2`) | 9.8934 | 98.1% | 98.6% | **2.1%** [1.5, 2.7] | 2.0% |
| Cascade, 1 answer | 13.6310 | 67.9% | 98.4% | **29.5%** [27.8, 31.5] | 24.7% |
| **Cascade, 3 answers** | **8.2187** | **75.9%** | **102.8%** | **41.3%** [39.0, 43.5] | **28.4%** [26.1, 30.6] |

- M2 bills all three local answers, including on questions that escalate.
- The gap between 78.9% (test curve) and 41.3% is the price of a guarantee.

## Slide 9 — Does it travel?

- **Headroom sweep** (T27, Table 5.2): question-only routing saves 12.0–21.1% (uncertified) on
  Mixtral, Yi-34B and Qwen alike; the cascade on Qwen reaches 78.9% on the test curve.
- **Leave-one-subject-out** on MMLU (T26, Table 5.3): savings 14.6% → 10.3% when the subject is
  unseen; per fold from **29.7%** down to **4.0%**.
- **Replay** (T44): 200 test questions through the live server → **200/200** identical decisions,
  37.3% saved on that slice, router p99 **5.4 ms**.

## Slide 10 — What we can and cannot claim

- **Can:** on questions like the test distribution, the 3-answer cascade keeps ≥ 95% of GPT-4's
  advantage with 95% confidence while cutting API spend by ~41%.
- **Cannot:** the guarantee does not cover free-form chat (the demo labels it "not certified"),
  a different model pair, or a refit on another scikit-learn version without re-certifying.
- Hand over to lane D: the same router, live on the Mac.
