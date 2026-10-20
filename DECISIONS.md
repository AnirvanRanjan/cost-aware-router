# Decisions

One line per decision the whole team is bound by. Change one only through a PR that every lane
reviews, and add a new line rather than editing an old one.

| ID | Date | Decision | Why | Status |
|---|---|---|---|---|
| D-01 | 2026-08-26 | Headline target is **PGR ≥ 0.95**, not "95% of frontier accuracy". | PGR is the standard; absolute 95% can be met by routing nothing. DESIGN §0. | Agreed |
| D-02 | 2026-08-26 | **Track A runs on RouterBench**: no generation, no API spend. Frontier arm priced at Sonnet 4.6 list rate, small arm at $0. | No API budget. DESIGN Rev A. | Agreed |
| D-03 | 2026-09-23 | **No paid API calls before the week-8 gate.** Free tiers only. | Budget. KICKOFF.md. | Agreed |
| D-04 | 2026-10-20 | **GSM8K is re-graded** by numeric exact match against the train-split answers. | RouterBench's GSM8K scores are a rubric, not correctness. `notes/routerbench_audit.md`, `notes/gsm8k_grader_check.md`. | In PR #4 |
| D-05 | 2026-10-20 | **Small models: Mixtral-8x7B (main), Yi-34B (second). Mistral-7B dropped.** | Mistral-7B answers "A" on 94% of MMLU and 100% of HellaSwag. Mixtral parses cleanly and matches RouteLLM; Yi-34B gives a smaller-gap pair for the headroom comparison. | Superseded by D-08 |
| D-06 | 2026-10-20 | **HellaSwag is reported separately**, not in the headline. Headline datasets: MMLU, GSM8K, ARC, Winogrande. MT-Bench as a point estimate only (80 items). | Mixtral answers "A" on 64% of HellaSwag, so part of its gap is answer bias. MT-Bench is too small for a bound. | **Proposed — agree by approving PR #4** |
| D-07 | 2026-10-20 | **The live demo runs on the M2**: Qwen3.5-4B locally via MLX, Gemini 3 Flash (free tier) as the frontier, router's small-model head retrained on Qwen's answers. | The project must end with the router working on a Mac. TASKS.md. | **Proposed** |
| D-08 | 2026-10-20 | **Roles of the model pairs.** **Qwen3.5-4B vs GPT-4** is the headline result and the live demo: the model the project is about, and the only one that runs on a Mac. **Mixtral-8x7B vs GPT-4** is the development pair (everyone builds on it now) and the comparison with RouteLLM, which used the same pair. **Yi-34B** is optional: already loaded, used only if lane C has time for a second comparison. Mistral-7B stays dropped. | Qwen's answers arrive around week 6; Mixtral's exist today. Building on Mixtral first means nobody waits, and Mixtral is the fallback if Qwen's runs slip. Qwen covers MMLU, GSM8K, ARC and Winogrande (~24,000 questions), enough for the confidence bounds. | **Proposed — agree by approving PR #4** |
