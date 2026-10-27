# Exploratory: routing on the small model's own answer (cascade)

*`scripts/12_cascade_explore.py`, Mixtral-8x7B vs GPT-4, frozen splits (`910a584c8279baa7`), headline
sources (MMLU, GSM8K, ARC, Winogrande). Full tables: `notes/cascade_report.md`; figure:
`notes/cascade_pareto.png`. EXPLORATORY: one model pair, RouterBench's single stored answer per question.*

## Question

Query-only routers (F0, F1, F2) barely beat random on this pair: within one dataset their AUC is
0.47–0.66, and the best certified saving at PGR ≥ 0.95 is 9.4%. Does letting the router look at the
small model's answer first (a cascade: S answers, then the router keeps the answer or escalates) do better?

## Result (tau* chosen on cal with the WSR bound, test touched once, cost model M1)

| Router | Sees | Frontier calls (test) | PGR test [95% CI] | Savings [95% CI] | CPT(95%) on test |
|---|---|--:|--:|--:|--:|
| F0 heuristics + LogReg | question | 91.0% | 96.6% [95.3, 97.8] | 3.9% [3.4, 4.4] | 86.2% |
| F2 two-head (lane B), rule v2 | question | 96.2% | 96.3% [95.2, 97.5] | 9.4% [7.8, 10.9] | 95.0% |
| Cascade: S's answer | + S's answer | 95.2% | 97.2% [96.1, 98.3] | 8.4% [7.2, 9.7] | 95.2% |
| **Cascade: S's answer + second opinion** | + agreement with a 2nd local model | **79.0%** | **97.9% [96.8, 98.9]** | **26.0% [24.2, 27.9]** | **68.2%** |

Random needs ~95% frontier calls for PGR 0.95; the oracle needs 22%.

## What carries the signal

- **Agreement.** On test, when Mixtral and Yi-34B give the same answer (60% of questions), Mixtral
  is right 84% of the time; when they disagree, 31%. That one bit is worth more than every
  query-text feature combined.
- **The answer alone** (option letter, length, arithmetic slips in GSM8K working) helps little: 8.4%.
  RouterBench stores only a letter for multiple choice, so there is not much to read.
- Adding the 23 query heuristics on top of answer + agreement made things *worse* (19.9%): more
  features, same training set, more overfitting.

## What it means for the project (proposal D-12)

The second opinion here is Yi-34B, which does not run on a Mac. The honest equivalent for the
headline pair and the demo is **Qwen's agreement with itself**: k = 3 sampled answers (temperature
0.7) for GSM8K and the option probability from lane B's `mcq_logprob` scorer for MMLU/ARC/Winogrande.
Both are cheap on the M2 (`generate.py` already records `mean_logprob` and supports k samples).

So: (1) Track B should generate k = 3 GSM8K samples, not 1, and keep MCQ option probabilities;
(2) add a cascade router (query features + Qwen's agreement / confidence) beside the query-only
router; (3) report both, since the cascade pays S's latency on every escalated query. Under M1 that
is free; the latency cost goes in the demo chapter.

Caveats: one pair; a single stored sample per model; Yi is a stronger model than a second Qwen sample
would be, so self-consistency will probably give less than 26%. That is exactly what Track B measures.
