# D-15: the demo routes with the certified research router (2026-11-16)

*Lane A, after T45 runs 1–2 (`notes/demo_set_check.md`, `notes/demo_set_check_run2.md`). Numbers from
`scripts/11_build_bundle.py` on release `trackb-6265adf7ffa4a602`, test split used once per row.*

## The problem

The D-14 demo bundle (lane B, `--mode cascade`) reads features from Qwen's answer *text* (arithmetic
slips, hedge words, option letters) plus the mean log-probability of one answer. On the demo set it
escalated every easy question Qwen got right: free-form answers are out of distribution for those
features, multiple-choice questions were answered as text without the option probability, and one
answer's mean log-probability is a weak signal. It is also not the router behind the thesis numbers:
the research table's `cascade_3sample` (T61, `scripts/16_trackb_preview.py`) uses Qwen's confidence
signals (agreement of 3 answers, option probability), not answer-text features.

## Options compared (certified on cal, test used once, PGR target 0.95, δ = 0.05)

| Bundle | Bound | Test frontier calls | Test PGR | Frontier share: GSM8K / MMLU / ARC / Winogrande |
|---|---|--:|--:|---|
| Answer text, 1 answer (D-14, lane B) | WSR, one seed | 70.2% | 100.3% | 76 / 69 / 34 / 92% |
| Answer text, 1 answer (D-14, lane B) | seed-free WSR | 85.2% | 100.5% | 86 / 84 / 74 / 100% |
| Answer text, 3 answers | seed-free WSR | 87.3% | 102.0% | 62 / 99 / 100 / 100% |
| **T61 signals, 3 answers (D-15)** | **seed-free WSR** | **65.8%** | **100.6%** | **50 / 73 / 47 / 100%** |
| T61 signals, 1 answer | seed-free WSR | 86.4% | 100.3% | 86 / 86 / 79 / 98% |
| T61 signals, 3 answers, rule v1 | seed-free WSR | 87.3% | 102.8% | 60 / 100 / 100 / 100% |

Single-seed WSR numbers move by up to 25 points with the seed (T60), so only seed-free rows compare.
Separate models per question type (one tau) were also tried and were worse (96% frontier calls).

## Decision (proposed)

- The demo bundle is lane B's T61 `cascade_3sample` router, rebuilt from the release files with the
  same signals (`router/signals.py`, tested against `scripts/16_trackb_preview.py`), the same
  gradient-boosted trees, rule v2 and train grid, and tau* from the seed-free bound (`threshold.py`
  `method="wsr_avg"`, ported from lane C's T60 script, identical p-values).
  `python scripts/11_build_bundle.py --mode cascade_t61 --samples 3`.
- The server computes the signals as Track B did: multiple choice gets one forward pass over the
  option letters, with the question put into RouterBench's MMLU format; arithmetic gets 3 sampled
  answers (T = 0.7, top-p 0.8, the GSM8K prompt) and their agreement; anything else gets 3 brief
  answers and their textual agreement, labelled "not a certified question type" on the page.
- The demo set is rebuilt from Track B test questions (`configs/demo_set.yaml`): 10 Qwen got right and
  the router keeps local, 9 Qwen got wrong and the router escalates, plus 4 free-form and 3 chat.

Cost: about 3 answers per arithmetic question (~6 s on the M2 with brief answers) instead of 1.

## Reproducibility finding (for T28 / T55)

Refitting the same T61 model on the same data gives a slightly different router on another machine:
the release `cascade_predictions.parquet` (lane B's machine) and a refit with scikit-learn 1.9.1
differ by up to 0.15 in predicted gap. Through the identical certification pipeline
(`scripts/17_t60_seed_free_wsr.py`) the release file gives tau* 8.22, 75.9% frontier calls and
**41.3%** certified savings; the refit gives tau* 9.66, 65.8% and **48.3% [46.1, 50.5]**. Both are valid
certifications of their own fitted router; the thesis quotes the release file. Record the
scikit-learn version with every predictions file, and build the demo bundle once, on the M2, from the
release (the bundle is a pickle and is tied to that version).

**Update 2026-11-19:** the version is the cause. Ani's M1 bundle was fitted with scikit-learn 1.8.0 and reproduces the release model (tau* 8.2187); the cloud refit and the M2 used 1.9.1 (tau* 9.657). `requirements.txt` now pins `scikit-learn==1.8.0`.
