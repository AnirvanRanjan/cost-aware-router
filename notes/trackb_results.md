# Track B: Qwen3.5-4B vs GPT-4 — first results (PREVIEW)

*2026-11-01. Qwen's answers from the UPES H100 run (`gpu/`, `artifacts/trackb/gpu_manifest.json`);
GPT-4's from RouterBench; frozen splits (`910a584c8279baa7`). Tables: `notes/trackb_preview_report.md`;
figure: `notes/trackb_preview_pareto.png`; code: `scripts/16_trackb_preview.py`. PREVIEW: one seed,
gradient-boosted trees; lane B owns the final router (T36, D-12).*

## The run

| | |
|---|---|
| Hardware | 1 × NVIDIA H100 80 GB (UPES cluster, host gpu01) |
| Software | vLLM 0.32.0; PyTorch 2.13.0 built for CUDA 13.0, run on the cluster's CUDA 12.8 driver through NVIDIA's **cuda-compat** library (not recorded in the manifest; note it in the methods) |
| Model | Qwen/Qwen3.5-4B, bf16, thinking off. The Mac demo runs the 4-bit MLX build: measure the difference on ~200 test questions |
| GSM8K | 7,450 questions × 3 samples (T 0.7, top-p 0.8, top-k 20, ≤ 1,024 tokens); median 108 output tokens; 0.3% cut off at the limit (`check_gsm8k_responses.py` reports 0.6% only if run with its default `--max-tokens 512`) |
| MCQ | 16,130 MMLU / ARC / Winogrande questions with a gold letter (the 649 without one are the questions no RouterBench model got right: 635 MMLU, 14 ARC); no letter bias (most-chosen C at 27%) |
| Time | 5.5 minutes for everything |

## How good is Qwen? (what a user would get, test + cal + train)

| Dataset | Qwen | GPT-4 | Gap |
|---|--:|--:|--:|
| GSM8K (majority of 3 samples) | 0.917 | 0.949 | 0.032 |
| ARC-Challenge | 0.922 | 0.971 | 0.049 |
| MMLU | 0.748 | 0.850 | 0.102 |
| Winogrande | 0.691 | 0.819 | 0.129 |

GSM8K and ARC fall below the design's 0.05 kill-check: on those, Qwen is nearly as good as GPT-4, so the
router's job there is mostly to keep things local. GPT-4's MMLU and ARC scores are higher here than on the
full corpus because the questions without a gold letter (where every model failed) are left out.

## Qwen knows when it is right

- GSM8K: when all 3 samples agree (87% of test questions), Qwen is right **96.9%** of the time; otherwise **60.5%**.
- Multiple choice: when its chosen option has probability ≥ 0.9 (49% of test questions), it is right **94.2%**
  of the time; otherwise **58.5%**.

## Savings at PGR ≥ 0.95 (τ* chosen on cal with the WSR bound, test used once, cost model M1)

| Router | Sees | Frontier calls (test) | PGR on test [95% CI] | Certified savings [95% CI] |
|---|---|--:|--:|--:|
| query_f2, v1 | the question only (F2) | 93.5% | 93.2% [89.1, 96.9] | 9.3% [8.0, 10.7] |
| **cascade_conf, v2** | **Qwen's own confidence** | **66.6%** | **100.6% [96.9, 104.7]** | **47.5% [45.4, 49.7]** |
| cascade_conf + F2, v2 | both | 97.7% | 99.1% [96.4, 101.9] | 1.5% [1.1, 2.1] |

References on the same test set: random needs 96% frontier calls for PGR 0.95, the oracle 7%.

**Reading it.**
1. **The cascade router saves 47.5% with a guarantee, about 1.9× cheaper**, using nothing but Qwen's own
   agreement and confidence. Under cost model M2 (laptop running cost) it is 45.1%.
2. **The certified number is conservative.** On the test curve the same router reaches PGR 0.95 at 39–41%
   frontier calls (~78% savings, uncertified). The bound is loose here because the pooled gap is small
   (~0.07), so the margin it must certify is tiny. A larger calibration set, or certifying per dataset,
   would narrow it; both are worth trying (lane C).
3. **Query-only routing is still weak** (9% savings), and its test PGR (93.2%) came in under the target that
   cal certified. Within the 5% risk the guarantee allows, but worth a sentence in the write-up.
4. **Adding F2 to the cascade made it worse**: 768 embedding dimensions swamp a dozen strong confidence
   features with this much training data. Lane B should regularise or reduce F2 before combining.
5. **The cascade costs latency, not money:** under M1 the 3 Qwen samples are free, but every escalated
   question pays for Qwen first. Report latency on the M2 (T47).
6. **One answer is nearly enough.** A quick check with only Qwen's first sample (its log-probability, no
   agreement) certified ~37% savings (uniform-cost approximation), so the demo can use one answer plus its
   confidence (proposed D-14) at a third of the latency.
7. **PGR above 100%** is real: on some questions Qwen is right and GPT-4 wrong, so keeping those local
   beats always-GPT-4.

## Next

- Lane B: T36 / D-12 with these signals as the official cascade router; regularise F2.
- Lane C: T37 (τ* for the Qwen pair), per-dataset certification, T27 headroom sweep (now Mixtral, Yi and Qwen).
- Lane D: measure the 4-bit Mac model against these bf16 labels on ~200 test questions; demo with the cascade.
- Lane A: publish `artifacts/trackb/` as a data release.
