# M2 checks: T59 (4-bit vs bf16), T31 (batch speed), T40, T42 (Nov 2026)

*Run by kartiken07 on the team's M2 (PR #19); H100 comparison computed from release `trackb-6265adf7ffa4a602`.*

## T59 — does the 4-bit Mac model answer like the bf16 H100 model?

Same questions: the first 100 GSM8K test and 100 MMLU test questions (sorted by qid).

| | Mac: 4-bit MLX, greedy, ≤ 512 tokens | H100: bf16 vLLM | Difference |
|---|--:|--:|--:|
| GSM8K accuracy | 0.840 | 0.860 (first sample) / 0.870 (mean of 3) | −2 to −3 points |
| MMLU accuracy | 0.820 | 0.790 | +3 points |
| Confidence recorded | 100% | 100% | — |

With 100 questions each, a 95% interval is roughly ±7 points, so **no measurable quantisation gap**:
the demo's 4-bit model behaves like the model the router was trained on, within noise. For the thesis,
say exactly that (and that the Mac run is greedy while the H100 sampled at T = 0.7). A 500-question rerun
would tighten it if a viva question demands it.

## T31 — batched decoding on the M2

Batch 4 = 1.4× (11.8 vs 8.4 tok/s), batch 8 = 1.35× (`notes/batch_speedup.json`). `max_batch` stays 1
because the batch path does not record `mean_logprob`, which the cascade router needs.

**Resolved 2026-11-15** (kartiken07, same M2, `mlx-lm` 0.32.2, low-power mode off; run on battery —
results match or beat the October figure, so power source did not depress them):

| Measurement | Result |
|---|---|
| `mlx_lm.generate` CLI, STEP1 protocol ("Solve: 17*23. Think step by step.", max-tokens 256, median of 3) | **29.1 tok/s** (28.1 / 29.1 / 29.3) |
| CLI on the demo's own chat-templated 122-token prompt, generation only | **29.5 tok/s** median (28.7 / 29.5 / 30.0); prompt processing 165 tok/s |
| `scripts/bench_batch.py --batches 1` rerun — same 16 GSM8K prompts, same code path | **23.4 tok/s** end-to-end (78.9 s vs 220.5 s on Nov 3; identical 1,848 out tokens) |

The 28.5 in `configs/models.yaml` **holds**: steady-state generation on this machine is ~29 tok/s.
The Nov 3 benchmark's 8.4 does not reproduce — the same script, prompts and model run 2.8× faster
today, so the gap was that run's conditions (then-installed mlx-lm version or machine load), not the
hardware. End-to-end through the demo path (prompt processing + Python overhead) is ~23 tok/s, so a
110-token answer is ~5 s, not ~13 s. Cost model M2 keeps `tokens_per_sec: 28.5` for generation; use
~23 tok/s only when pricing whole-call wall time.

## T40, T42 / T43

- Gemini: `gemini-3-flash-preview` confirmed; 3/3 smoke calls, 0 errors.
- Demo server running on the M2 with the interim bundle (τ\* = 0.090, T41 check 100%); router decision
  p99 **7.6 ms** (target < 15 ms); both routes verified.
- Fixed after the run: `--fake` and real runs shared `artifacts/cache/demo.sqlite`, so the first real query
  could return a fake answer. `--fake` now uses `demo_fake.sqlite` (scripts/serve_demo.py). Delete the old
  `artifacts/cache/demo.sqlite` on the M2 once.
