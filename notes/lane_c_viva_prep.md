# Lane C Viva Voce Preparation Guide

**Candidate:** Sarthak Jakhmola (Lane C: Calibration, Evaluation and Risk Control)
**Project:** Cost-Aware LLM Routing (Major Project, UPES Dehradun)

*Draft answers, every number sourced. Say them in your own words; know the source of each number.*

---

## Q1. Why PGR, and not plain accuracy?

Accuracy alone doesn't say how much of the frontier's advantage you kept. With Qwen at 81.3% and GPT-4
at 88.7% (`notes/t27_t28_locked_numbers.md`), a router at 87% looks "only 1.7 points worse", but it
has kept (87 − 81.3)/(88.7 − 81.3) ≈ 77% of what the frontier adds. PGR = (Q − Q_S)/(Q_L − Q_S)
normalises by the gap, so a target like "≥ 95%" means the same thing for every model pair
(RouteLLM's definition, so results compare with the literature).

## Q2. How does choosing τ on the calibration split avoid leakage?

τ* is chosen on **cal** only (6,957 questions in the corpus split; the Track B test split is 4,707),
and the test split is evaluated **once**, after τ* is frozen. Lane A's splits are grouped by
near-duplicate clusters (1,883 duplicate pairs merged), so the same question cannot sit in cal and test.
Reading the operating point off the test curve gives 78.9% savings; that number is reported but marked
uncertified because the threshold has seen the data it is judged on.

## Q3. What exactly is being tested?

For each τ, the paired margin e_i(τ) = s_route(i) − [(1 − ρ) s_S,i + ρ s_L,i] with ρ = 0.95.
Its mean is ≥ 0 exactly when PGR ≥ ρ. H0: E[e(τ)] < 0. Reject H0 at level δ = 0.05 → τ is certified.

## Q4. You test many thresholds. Why no multiple-testing correction?

Fixed-sequence testing (Learn-then-Test): order the thresholds from the safest (send everything to the
frontier) outward, test in that order, and stop at the first failure. Because the order is fixed in
advance and we never skip a failure, the family-wise error stays at δ without Bonferroni.

## Q5. Why not the bootstrap? Everyone uses it.

Two properties of this problem break it: the margin is exactly 0 on 70–80% of questions (both models
agree), and the headroom is small (0.075 for Qwen), so the true mean sits near the null boundary.
In Monte Carlo on n_cal = 2,000 the nominal 95% bootstrap bound failed to cover up to **15.2%** of the
time, three times the allowed 5% (`thesis/05_results.md` §5.5.1). A guarantee that fails 15% of the
time isn't the guarantee we claim.

## Q6. Then why not Hoeffding?

It's valid, but it uses only the range of e_i, not its variance. With most margins at 0, the variance
is tiny compared with the range, so Hoeffding is far too conservative and discards most of the savings
(§5.5.2). WSR's betting martingale is valid at every n and adapts to the variance.

## Q7. What was wrong with plain WSR?

It processes the calibration questions in one random order. Changing only the seed (0–19) moved τ*
from 8.11 to 16.72 and test savings from **40.8% to 74.9%** (`notes/t60_certification_report.md`).
Each run is a valid certificate, but if you could pick the seed you could pick your result.

## Q8. Why is averaging over 50 orders still valid?

For a fixed order the final wealth E^(m) is an e-value: its expectation under H0 is ≤ 1. An average of
e-values is still an e-value (linearity of expectation, no independence needed). Markov's inequality
then gives p = min(1, 1/Ē) as a valid p-value. With M = 50 orders we get τ* = **8.2187** with no seed
left in it. It is slightly more conservative than the best single seed, which is the point.

## Q9. Your headline is 41.3%. What does M2 = 28.4% mean?

M1 counts only API spend (local = $0). M2 also charges the laptop: a $1,199 MacBook over 3 years at
8 h/day plus 20 W of electricity, per second of generation (lane A's `router/costs.py`). The cascade
runs Qwen three times on **every** question, including those that then escalate, so M2 bills all three
answers: 28.4% [26.1, 30.6]. Billing only one answer (the earlier convention) gave 36.5%, which
undercounts.

## Q10. Test PGR is 102.8%. How can you keep more than 100%?

On some questions Qwen is right and GPT-4 is wrong. Keeping those local makes the router better than
always-frontier, so Q can exceed Q_L and PGR exceeds 100%. The 95% CI is [99.1, 107.0].

## Q11. What if a different scikit-learn version gives a different τ*?

Refitting the trees on 1.9.1 gave τ* = 9.657 (48.3% certified) instead of 8.2187 (41.3%). Both are
valid certificates for their own fitted model; the guarantee is about a model, not a recipe. We report
the release model (fitted on 1.8.0) and pinned that version so the demo serves the certified model.

## Q12. Does the guarantee hold for anything a user types in the demo?

No. It holds for questions drawn like the test distribution (MMLU, GSM8K, ARC, Winogrande). Free-form
chat is answered the same way but the page labels those decisions "not certified". Leave-one-subject-out
on MMLU (T26) shows savings falling from 14.6% to 10.3% on unseen subjects, so distribution shift matters.
