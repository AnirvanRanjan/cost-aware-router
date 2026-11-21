# Replay Mode Report: D-15 cascade bundle (T44)

*Generated 2026-11-20 17:28 UTC by `scripts/15_replay_demo.py`.*

## How this was run

- Bundle `artifacts/bundle/router.pkl`: `cascade_t61_3sample`, rule v2, tau* = 8.2187, pair `qwen3.5-4b` vs `gpt-4-1106-preview` (scikit-learn 1.8.0 build, `python scripts/11_build_bundle.py --mode cascade_t61 --samples 3`).
- 200 held-out test queries, seed 42, sampled from the frozen Track B corpus (`artifacts/trackb/corpus.parquet`, test split, headline sources).
- Every query went through the server's own code path -- `router/serve.py`, `Demo._answer_local_first` -- with Qwen's answers read from the release files (`qwen_responses.parquet`, `qwen_mcq_options.parquet`). No live generation, no frontier calls.
- Offline reference: the signals `scripts/11_build_bundle.py` trained on, the true corpus source, decision = score > tau* (`threshold.evaluate_at_tau` on the same 200 queries).

## Summary

- **Held-out queries tested:** 200 (arc-challenge 8, gsm8k 61, mmlu 124, winogrande 7)
- **Local queries (S):** 51 (25.5%)
- **Frontier queries (L):** 149 (74.5%)
- **Frontier share by source:** mmlu 85%, gsm8k 46%, arc-challenge 100%, winogrande 100%
- **Quality (stored scores):** 0.8600 (Qwen's stored answers: 0.8100, GPT-4: 0.8550), PGR 1.111
- **Spend:** $0.1313   **Always-frontier counterfactual:** $0.2093   **Saved:** $0.0780 (**37.3%**)
- **Router decision latency (p50):** 3.57 ms   **(p99):** 5.43 ms (max 24.76 ms; 200 decisions in 0.8s wall)

## Agreement with the offline evaluation

Checked on the same 200 queries, decisions from the server path vs `bundle.predict(signals, sources)` + `threshold.evaluate_at_tau`:

| Check / Metric | Server path (`Demo._answer_local_first`) | Offline reference | Status |
|---|---:|---:|:---:|
| Decision agreement | 200/200 (100.0%) | 200/200 | Pass |
| Frontier escalation rate (alpha) | 74.5% | 74.5% | Pass |
| Cost savings vs always-frontier | 37.3% | 37.3% [29.8%, 44.8%] | Pass |
| Quality (stored scores) | 0.8600 | 0.8600 | Pass |
| Router decision latency p99 | 5.43 ms | < 50.0 ms (CPU) / < 15.0 ms (M2) | Pass |

**Overall Acceptance Result:** **Pass** (every replayed decision matches the offline evaluation on tau* = 8.2187).

## Context: the certified bundle on the whole test split

- `meta["test"]` in `artifacts/bundle/router.pkl` (offline, the whole test split the bundle was built on): alpha 75.9%, PGR 1.028, unit-cost savings 24.1% (unit cost = 1 per frontier query, 0 per local query -- the bundle build's convention).
- Under the serve.py money model on this sample: 37.3% (spend $0.1313 of $0.2093 at 3.0/M input, 15.0/M output tokens).

## Notes

- 3 of 4707 test queries cannot be replayed from the stored files: the server takes their sampled-answer branch but Track B only stored option log-probabilities for them. `mmlu-formal-logic.val.24`, `mmlu-formal-logic.val.85`, `mmlu-professional-law.val.1237`.
- Question types this sample hit: arithmetic 60, free-form 1, multiple choice 139; 1 query went through the free-form branch the server labels *not a certified question type* (a prompt with no digit), as in the live demo.
- Cost: the replay never calls the frontier model, so the output-length estimate for every query is the shown (stored Qwen) answer's length -- the same convention `serve.py` uses for its always-frontier counterfactual on local queries.
- 139 stored-answer lookups, no model calls; signal computation is the only work.
