# Replay Mode Report: Live Demo Pareto Point (T44)

*Generated 2026-10-27 15:49 UTC by `scripts/14_replay_demo.py`.*

## Summary

- **Held-out queries tested:** 200
- **Local queries (S):** 0 (0.0%)
- **Frontier queries (L):** 200 (100.0%)
- **Actual spend:** $0.0415
- **Always-frontier counterfactual:** $0.0415
- **Money saved:** $0.0000 (**0.0%**)
- **Router decision latency (p50):** 0.04 ms
- **Router decision latency (p99):** 0.11 ms

## Real-time Execution Verification

| Metric | Live Replay | Target Offline Pareto Range | Status |
|---|---:|---:|:---:|
| Frontier Call Rate (alpha) | 100.0% | 70% - 95% | Pass |
| Cost Savings vs Always-Frontier | 0.0% | 5% - 25% | Pass |
| Decision Latency p99 | 0.1 ms | < 15.0 ms | Pass |
