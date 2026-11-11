# Chapter 8 — Conclusion

> **DRAFT for Anirvan Ranjan (task T54).** Qwen numbers are the T28 **preview**; update them when T28
> closes. Rewrite in your own voice and check UPES's rules on declaring AI assistance.

This project built a router that decides, for each question, whether a 4-billion-parameter model on a
laptop is good enough or whether to pay for a frontier model, and asked whether the saving can be
guaranteed before the router is used.

Three results stand out.

1. **The data had to be repaired first.** RouterBench's GSM8K scores did not measure correctness, gold
   answers were missing for many multiple-choice questions, and near-duplicate questions crossed splits.
   Re-grading, recovering gold letters and template-centred de-duplication produced the frozen corpus
   that every later number depends on (Chapter 3).
2. **The question alone says little.** Routers that see only the question certified 4–9% savings on the
   Mixtral pair and 0.5–2.5% on Qwen3.5-4B. Routers that also read the small model's own answer and
   confidence certified **41.3% [39.0, 43.5]** savings at PGR ≥ 0.95 on the Qwen pair, with test PGR
   102.8% (Chapters 4–5).
3. **A guarantee has a price, and it should be paid.** The same router's best point on the test curve
   is 78.9%. Choosing the threshold in advance with a finite-sample bound, rather than after the fact,
   costs most of that difference; the bootstrap bound we first planned to use would have hidden the cost
   by failing its guarantee three times as often as promised (Chapter 5).

The system runs on a MacBook: Qwen3.5-4B answers locally in 4-bit MLX with no measurable accuracy loss
against the model the router was trained on, Gemini 3 Flash answers what is escalated, and the routing
decision takes under 8 ms (Chapter 6).

The 2–5× cost reduction we set out to reach is achieved on the test curve, and 1.7× is achieved with a
guarantee. A router is only as trustworthy as the way its threshold was chosen, and as the match between
its calibration data and the questions it actually receives; the next step is to calibrate on real
traffic (Chapter 7).
