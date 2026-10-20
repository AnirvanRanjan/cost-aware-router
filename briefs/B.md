# Brief: Person B · Features & Router (@HSID007)

You turn a question's text into features, and features into two calibrated predictions: how well
the **small** model will do on it (`ŝ_S`) and how well the **frontier** model will (`ŝ_L`). Lane C
turns those into a routing threshold; lane D runs your router live on the Mac at the defence.

**You own:** `router/features.py`, `router/models.py`, `router/calibrate.py`, and the multiple-choice
part of `router/graders.py`. **You hand off:** `features.parquet`, `predictions.parquet`, a
multiple-choice grader for lane D, and the router bundle for the demo. **Your thesis chapter:**
feature engineering and router architecture. **You review:** C's pull requests.

---

## Get set up (20 minutes)

```bash
git clone https://github.com/AnirvanRanjan/cost-aware-router.git && cd cost-aware-router
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pytest                 # all tests pass
gh release download data-3f774421cc116201 -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench
python scripts/verify_data.py                             # "Same data as everyone else."
pip install -r requirements-ml.txt                        # torch, sentence-transformers, xgboost, mlx-lm
```

Read first: `DECISIONS.md` (10 lines), `DESIGN.md` §1.3 and §2, and `router/schema.py`.

## What's already there for you

- **Real data:** 34,778 questions in `corpus.parquet` (column `prompt` is clean text; GSM8K is the
  question alone; MT-Bench joins both turns), and 104,334 scores in `scores.parquet`.
  **Build on the Mixtral pair** (`mistralai/mixtral-8x7b-chat` vs `gpt-4-1106-preview`): it's the
  development pair. The Qwen pair arrives around week 6 (DECISIONS D-08).
- **Contracts:** `schema.FEATURES` (`qid, feature_set, vector`) and `schema.PREDICTIONS`
  (`qid, router, seed, s_hat_S, s_hat_L, delta_hat, c_hat_L`). Always write with `schema.write`.
- **A test oracle:** `router/synthetic.py` makes fake data where difficulty is planted and known
  (`truth.difficulty`). Use it to check your code finds the signal, never to tune anything.
- **Graders:** `router/graders.py` already has GSM8K exact match. You add multiple choice.

---

## Your tasks, in order

### 1 · T32 — multiple-choice grader by option probability ⚠ do this first

**Why first:** Kartiken (D) can't score Qwen on MMLU, ARC and Winogrande without it, and his runs
are on the critical path to the headline result and the demo.

**What it does:** asking a small model to *write* a letter fails ("always A", see
`notes/routerbench_audit.md`). Instead, run **one forward pass** and read the model's probability
for each option label as the next token.

- `parse_options(prompt) -> list[str]`: the option labels in the prompt. Almost all are `A–D`;
  Winogrande is `A–B`; a few MMLU and ARC items have `A–E` or only `A–C`. Don't hard-code four.
- `mcq_score(option_logprobs: dict[str, float], gold: str) -> float`: softmax over the options,
  return **P(gold)** (a soft score, DESIGN §1.2). Pure Python, no model, fully unit-tested.
- `MLXOptionScorer` (new `router/mcq_mlx.py`): apply Qwen's chat template with the corpus prompt as
  the user turn (it already ends in `Answer:`), add the generation prompt, take the next-token
  logits, and read each label. Labels tokenise differently with and without a leading space
  (`"A"` vs `" A"`), so take the max over both.
- Agree the interface with Kartiken: his runner calls `scorer.option_logprobs(prompt, labels)` and
  writes SCORES rows with `grader="mcq_logprob"`, `mean_n_out=1`.

**Gold answers:** ✅ ready. `corpus.reference` holds the correct letter for MMLU, ARC, Winogrande
and HellaSwag (T07, release `data-3f774421cc116201`), voted from every model that answered correctly.
Coverage: 99.1% of ARC, 95.5% of MMLU, 100% of Winogrande; questions with no gold letter are
skipped.

**Done when:** unit tests pass with a fake model returning fixed logits; and on 20 real MMLU
questions on a Mac, Qwen's accuracy is clearly above chance (0.25) with no answer letter above ~40%.

### 2 · T10 — F0 heuristic features

`heuristics(prompts) -> (np.ndarray, list[str])`, a pure function. About 25 features: token, char
and sentence counts; mean word length; digit, punctuation and uppercase density; question marks;
code fences; LaTeX; URLs; number of options; whether there's a reading passage ("refers to the
following information"); counts of cue words ("prove", "derive", "step by step", "explain");
numeric literals; maximum bracket depth; type-token ratio; Flesch–Kincaid grade.

> **Features must come from the text alone.** Never use `source`, `subject` or `qid`: in the live
> demo a query arrives with no label saying which benchmark it's from.

**Done when:** hand-written prompts give hand-computed values in tests, and token count has
Spearman ρ > 0.3 with `truth.difficulty` on synthetic GSM8K.

### 3 · T11 — F1 TF-IDF

Word 1–2 grams plus character 3–5 grams. **Fit on the `train` split only**, then transform every
split. **Done when** a test proves fitting never sees `cal` or `test`.

### 4 · T12 — F2 sentence embeddings ⚠ hand the file to Ani

`BAAI/bge-base-en-v1.5` via sentence-transformers (use `device="mps"` on a Mac), normalised, for all
34,778 prompts → `features.parquet` with `feature_set="F2"`. A few minutes. Long MMLU passages are
truncated at 512 tokens: note how many.

**Then send the file to Ani** for the meaning-based duplicate check. That pass moves some questions
between splits once more, then the splits freeze. Your features are keyed by `qid`, so they don't
change; only refit F1 on the new `train` split.

### 5 · T13 — baselines

Logistic regression on F0 and on F1; XGBoost on F0 + F2. Predict the gap `Δ = s_L − s_S`.
**Metric: Spearman ρ(Δ̂, Δ)**, not accuracy — the threshold sweep consumes the ranking.

### 6 · T14 — the two-head router

MLP `[F2 ‖ F0] → 256 (GELU, dropout 0.1) → 2`, outputs `ŝ_S`, `ŝ_L`; loss
`BCE(ŝ_S) + BCE(ŝ_L) + 0.5 · MSE(ŝ_L − ŝ_S, s_L − s_S)`; AdamW; 5 seeds. Write `predictions.parquet`.

> **Early stopping can't use `cal`.** `cal` is reserved for calibration and for C's threshold, and
> using it twice leaks. Carve a grouped 85/15 validation split out of `train` (use
> `router.splits.grouped_split` so no `group_id` straddles it).

### 7 · T15 — calibration

Isotonic regression per head, fit on `cal`. Reliability diagrams and 15-bin ECE, before and after,
saved in `notes/`.

### 8 · T16 — the ablation ladder

Fill every row of DESIGN §4.3: F0 → F1 → XGBoost → F2 → F2+F0 → cost-normalised rule v2. This
table is the core of your chapter.

### Later

- **T36 (week 6–8):** when Ani's Track B data lands, retrain **only the `ŝ_S` head** on Qwen's
  answers and recalibrate. The two-head design exists for exactly this.
- **T41 (week 9–11):** the router bundle for the demo: one file with encoder name, heuristics
  version, weights, calibrators and τ\*, behind `router.predict(text)`. Kartiken's demo needs a
  decision in under 15 ms on the M2. If `bge-base` is too slow there, `all-MiniLM-L6-v2` is the
  fallback.
- **T51:** your chapter.

---

## Who you depend on, and who depends on you

| | What | When |
|---|---|---|
| You need from **A** (Ani) | Gold letters for MMLU/ARC/Winogrande (T07); Track B data (T35) | T07 this week; T35 ~week 6 |
| **A** needs from you | `features.parquet` with F2 | As soon as T12 is done |
| **D** (Kartiken) needs from you | The multiple-choice grader | This week, before his runs |
| **C** needs from you | `predictions.parquet` | Week 7–9 |
| **D** needs from you | The router bundle | Week 11 |

## Rules

- Every output goes through `schema.write`; every module gets tests.
- Fit anything learned (TF-IDF, scalers, calibrators) on `train` or `cal` only, never on everything.
- Never tune on synthetic data.
- Branches: `b/T32-mcq-grader`, `b/T10-heuristics`, …; PRs reviewed by someone outside lane B.
