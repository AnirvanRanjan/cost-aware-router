# Chapter 7 — Discussion

> **DRAFT for Anirvan Ranjan (task T54).** Qwen numbers are the T28 **preview**; replace them with the
> locked numbers when T28 closes, and re-check every sentence that compares two numbers. Every number
> names the note it comes from. Rewrite in your own voice and check UPES's rules on declaring AI assistance.

## 7.1 The answer to the research question

Chapter 1 asked how much of the frontier model's cost can be saved while keeping 95% of its quality
advantage, and whether that can be guaranteed in advance. On the Qwen3.5-4B / GPT-4 pair the answer has
two parts (`notes/t60_certification_report.md`, `notes/trackb_cascade_report.md`):

| | Operating point chosen | API savings (M1) | Cost ratio | Guarantee |
|---|---|--:|--:|---|
| Cascade router, threshold certified on cal | before seeing test | **41.3% [39.0, 43.5]** | 1.7× | PGR ≥ 0.95 with probability ≥ 95% |
| Same router, best point on the test curve | after seeing test | 78.9% | 4.7× | none |
| Question-only router, certified | before seeing test | 0.5–2.5% | ≈1.0× | as above |

The original target of 2–5× lower cost at PGR ≥ 0.95 is reached on the test curve, but not with a
guarantee. We think the certified number is the one a deployment can rely on, and the gap between the two
rows is the main finding of this thesis rather than a shortfall: most routing results in the literature
are curves (Chapter 2), and our curve would have reported 4.7×.

## 7.2 Why the question alone is not enough

Every question-only router we built — heuristics, TF-IDF, sentence embeddings, a two-head network —
recovered 4–9% certified savings on the Mixtral pair and 0.5–2.5% on the Qwen pair (Chapter 4,
`notes/qwen_query_report.md`). Three observations explain this.

1. **Headroom shrinks as small models improve.** GPT-4's quality advantage over the small model falls
   from 0.237 for Mixtral to 0.075 for Qwen3.5-4B on the same test questions
   (`notes/t27_t28_locked_numbers.md`). With little headroom, PGR ≥ 0.95 leaves almost no room for a
   router's mistakes.
2. **Difficulty is not visible in the text.** Two GSM8K problems with the same template and length can
   differ in whether a 4B model gets them right. Leave-one-subject-out evaluation on MMLU (Mixtral pair) shows
   the router also loses part of what it has when the subject is new (AUC 0.665 pooled vs 0.617 out-of-fold,
   `notes/loso_report.md`).
3. **Benchmark prompts are not user prompts.** The interim router sent 8 of 10 easy, short questions in
   our demonstration set to the frontier model, because nothing in its training data looked like
   "Is 17 a prime number?" (`notes/demo_set_check.md`). This is distribution shift, and it is a threat to
   any router trained only on benchmarks.

## 7.3 Why the cascade works, and what it costs

The small model knows when it is right. On GSM8K, when all three of Qwen's samples agree (87% of test
questions) it is right 96.9% of the time, against 60.5% otherwise; on multiple choice, when its chosen
option has probability at least 0.9 (49% of questions) it is right 94.2% of the time, against 58.5%
(`notes/trackb_results.md`). A router that reads these signals no longer has to guess difficulty from
the question; it measures the small model's confidence in its own answer, as in earlier cascade work
[@yue2024cascades; @wang2023selfconsistency].

The savings are uneven across datasets (`notes/trackb_cascade_report.md`): 46.6% on GSM8K, 30.3% on ARC,
18.1% on MMLU and 1.6% on Winogrande. A single threshold is chosen for the pooled sources, so the router
keeps the cheap, confident GSM8K answers local and escalates almost all of Winogrande, where Qwen is weakest.

The cascade has two costs that a question-only router does not.

- **Latency on escalated queries.** Every query is answered by the local model first. When the router
  then escalates, the user waits for both models. At the certified threshold about 76% of headline test
  queries are escalated. For multiple choice the local step is a single forward pass, but a 110-token
  GSM8K answer takes about 4 s at 28.5 tokens per second, or 13 s at the 8.4 tokens per second measured
  by the batch benchmark (`notes/m2_checks.md`; the difference is still being re-measured).
- **Samples.** The three-sample router needs three local generations. The demonstration therefore uses a
  one-sample version (decision D-14), which certifies 27.8% [25.9, 29.6] instead of 40.8%
  (`notes/trackb_cascade_report.md`, single permutation seed). Self-agreement is the most useful signal,
  and it is the one that costs the most.

## 7.4 The price of a guarantee

The same predictions give very different savings depending on how the threshold is chosen
(`notes/trackb_cascade_report.md`, `cascade_3sample`, rule v2):

| Threshold rule | Valid? | Test PGR | Savings (M1) |
|---|---|--:|--:|
| Naive (point estimate on cal) | no guarantee | 97.2% | 77.0% |
| Paired bootstrap | asymptotic only; undercovers | 100.0% | 65.8% |
| Betting bound (WSR), seed-free | finite-sample | 102.8% | 41.3% |
| Hoeffding | finite-sample, loose | 100.3% | 4.6% |

Two findings matter here. First, the bootstrap bound our own design recommended undercovers: in
simulation it failed 15% of the time where it promised 5% (Chapter 5, `notes/chapter_C_facts.md`). We replaced it with the betting
bound. Second, a single run of the betting bound depends on the random order in which it reads the
calibration data: across 20 seeds, certified test savings ranged from 40.8% to 74.9%
(`notes/t60_certification_report.md`). Reporting the best seed would be the same optimism the guarantee
is meant to remove. Averaging the terminal betting wealth over 50 orderings gives a valid e-value and a
single answer, 41.3%, close to the most conservative seed. The realised test PGR above 100% shows the
certified threshold is cautious: it buys the guarantee by escalating more than strictly needed.

## 7.5 Is running the model locally actually cheaper?

Under cost model M1, local inference is free and the savings follow. Two comparisons qualify this.

- **The Batch API.** Calling GPT-4-class models through a batch interface costs half the list price.
  Against always-frontier at batch prices, the certified router is 18.5% *more* expensive
  (`notes/trackb_cascade_report.md`, "vs Batch" column). The router wins only where answers are needed
  interactively, which is the use case of this project; batch processing has hours of latency.
- **The laptop's own cost.** Counting the Mac's purchase price, power and lifetime (M2), the certified
  saving falls from 41.3% to 36.5% (at the configured 28.5 tokens per second). The break-even speed against Gemini 3 Flash at list price is
  12.1 tokens per second (Chapter 5). The October CLI measurement on the M2 (28.5 tokens per second) is
  well above it; the November benchmark (8.4) is below it. Which one holds decides whether, against a
  cheap frontier model, local generation saves money at all. Against Claude-Sonnet-class pricing, which
  we use as the reference, the break-even is a few tokens per second (`router/costs.py`), and local
  generation is cheaper at every speed we measured.

## 7.6 Threats to validity

- **Exchangeability.** The guarantee holds for queries drawn like the calibration split. It says nothing
  about user questions in the demonstration, which §7.2 shows are distributed differently.
- **The frontier model is fixed in time.** GPT-4's answers are RouterBench's stored November 2023
  outputs. The demonstration calls Gemini 3 Flash, whose quality and price differ; the router was not
  calibrated on Gemini's answers.
- **Local labels come from one GPU run.** Qwen's labels were generated once on an H100 in bf16 with
  sampling at temperature 0.7; the demonstration runs a 4-bit model with greedy decoding. On 100 GSM8K
  and 100 MMLU questions the difference was −2 and +3 points, within the ±7-point noise of that sample
  size (`notes/m2_checks.md`); a larger check would narrow this.
- **Excluded questions.** 649 multiple-choice questions with no recoverable gold letter (635 MMLU,
  14 ARC) are left out of Track B (`notes/trackb_results.md`). These are questions no RouterBench model
  answered correctly, so the headline pool is slightly easier than the full benchmarks.
- **One router fit.** The cascade routers are gradient-boosted trees fitted once with one seed. The
  certification is valid for that fitted router; a different fit would give a different, equally valid
  number.
- **Benchmarks only.** Four academic benchmarks stand in for real use. Open-ended chat, code and
  long-form writing were not evaluated with a guarantee.

## 7.7 Future work

1. **Calibrate on the deployment.** Collect a calibration set of real user questions with both models'
   answers, and re-certify the threshold on it; or use online, sequential versions of the same betting
   bound so the guarantee follows the traffic.
2. **Thresholds per task type.** One pooled threshold escalates nearly all of Winogrande while leaving
   GSM8K savings on the table. Group-conditional (Mondrian) versions of Learn-then-Test could certify a
   threshold per source.
3. **Cheaper confidence.** Self-agreement needs several samples. A small verifier, or confidence read
   from the first few tokens, could keep most of the signal at the cost of one generation.
4. **More than two models.** A chain of local and API models of increasing size, with a certified
   threshold at each step.
5. **Measure what the user feels.** Report end-to-end latency, not only cost, and include it in the
   objective.
