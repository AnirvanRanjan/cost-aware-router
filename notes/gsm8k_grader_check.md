# GSM8K grader check (DESIGN §3.1 gate: ≥ 95% agreement on 50 hand-checked items)

**Result: 48 / 50 = 96%. Gate passed.** Grader: `router/graders.py` `gsm8k_exact_match`, version `graders-1.1`.

## Why GSM8K is re-graded at all

RouterBench's GSM8K scores take five values (0, .25, .5, .75, 1) and GPT-4 never scores 1.0, so they
are not correctness (see `routerbench_audit.md`). We re-grade every stored response by comparing its
final number with the GSM8K train-split answer. All 7,450 RouterBench items matched a gold answer.

## How the final number is chosen

Compared on all 7,450 items. Accuracy is the share of responses graded correct:

| Extractor | GPT-4 | Mixtral-8x7B | Mistral-7B |
|---|---|---|---|
| Last number in the response | 0.844 | 0.540 | 0.343 |
| Result of the last "= X" | 0.942 | 0.596 | 0.384 |
| **Chosen:** last "= X", unless the closing sentence states a different number (numbers copied from the question excluded) | **0.949** | **0.593** | **0.374** |

"Last number" fails on closing sentences that restate the question ("480 peanuts *in 6 minutes*").
"Last = X" credits intermediate values the model later abandons ("… = 3. Therefore … 4 shelves").
The chosen rule fixes both; regressions for each are in `tests/test_graders.py`.

## Hand check

50 items drawn at random (seed 42): 25 GPT-4, 25 Mixtral. For each, the gold answer, the extracted
number and the end of the response were read by hand.

- **GPT-4: 25 / 25 graded correctly.**
- **Mixtral: 23 / 25.** Both misses are false negatives — the model was right, the grader said wrong:
  - "a total of 260 oranges (80 … + 60 … + 120 …)" → extracted 120; the answer is stated first.
  - "1800 inches or 150 feet of wood" → extracted 1800; the gold answer is in feet.

**Effect on results:** misses fall on the small model's correct answers, so GSM8K headroom
(GPT-4 − Mixtral = 0.356) may be overstated by a point or two. Note this beside any GSM8K number.
