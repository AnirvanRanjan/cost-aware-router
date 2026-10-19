# Cost-Aware Model Router — Technical Design

**Status:** Draft v0.1 (design only, no implementation)
**Owner:** Ani · **Lead ML:** Claude
**Last updated:** 2026-08-26

---

## TL;DR

Build a lightweight query-difficulty router that decides, per query, between a
locally-hosted small model (**S**) and a frontier API model (**L**). Target:
**2–5× cost reduction at ≥95% of frontier quality.**

Three decisions drive the whole project:

1. **The asset is the paired response corpus**, not the classifier. For every
   query we need *both* models' answers and a comparable score. That is the one
   expensive, irreversible step. Everything downstream (features, classifier,
   threshold sweep) is cheap and re-runnable against a frozen corpus. Design the
   repo so the corpus is generated once, content-hash cached, and never
   regenerated.
2. **Predict the quality gap, not the winner.** Train two calibrated heads that
   predict `E[s_S(q)]` and `E[s_L(q)]` separately. The routing score is
   `Δ̂(q) = ŝ_L(q) − ŝ_S(q)`. This handles the "both fail" case correctly (gap
   ≈ 0 → route small, save money on an unanswerable query), survives swapping
   either endpoint model, and extends to a 3-tier cascade without redesign.
3. **Threshold selection is a statistics problem, not a plotting problem.**
   `τ` picked on the same data used to report savings is optimistically biased.
   Use a three-way split and a **paired bootstrap lower confidence bound** on
   quality. This is the single largest source of fake results in routing papers
   and the easiest place for this project to be more rigorous than the field.

---

## Rev A — confirmed constraints and revised plan

*Supersedes §1.1, §4.2 Phase 2, Phase 8, and Appendix A. Confirmed 26 Aug 2026.*

| Constraint | Value | Consequence |
|---|---|---|
| **Local compute** | Apple Silicon, ≥16 GB unified memory, no CUDA | `S` = Qwen3-8B @ 4-bit via **MLX**. F3 fine-tuning becomes feasible (see below). |
| **Frontier API budget** | **None** | You cannot generate the frontier arm. So don't — acquire it off the shelf. |
| **Goal** | Academic report / thesis | Phase 8 (serving) dropped. Benchmark-first; Arena demoted to optional. |

### The central move: buy the expensive arm instead of generating it

Phase 2 was the only irreversible spend in the original plan. With no budget, it is replaced
by **acquisition**, and the project loses nothing — it actually gains statistical power.

#### Track A — primary, $0: RouterBench

[RouterBench](https://github.com/withmartian/routerbench) (Hu et al., 2024) released
**405,467 precomputed inference outcomes**: 11 models × 8 datasets × 64 tasks, with
**per-item quality scores and token costs already computed**.

- Models include GPT-4, GPT-3.5-turbo, Claude-v1/v2/instant, Llama-70B-chat,
  Mixtral-8x7B-chat, Yi-34B-chat, Mistral-7B-chat, WizardLM-13B, Code Llama-34B.
- Datasets: **GSM8K, MMLU, MT-Bench, MBPP**, ARC-Challenge, HellaSwag, Winogrande, RAG.
- Suggested pair: `S` = **Mistral-7B-chat**, `L` = **GPT-4**. Add Claude-v2 or Llama-70B as an
  alternate `L` for a robustness column.

Three things this buys you:

1. **Re-pricing is the contribution.** RouterBench's costs are 2024-era API prices. Recompute
   `c_L` from its token counts at **current 2026 rates** (Sonnet 4.6 at $3/$15), and — the key
   move — set **`c_S = 0`**, because in your framing the weak arm runs locally. *"What does the
   cost–quality frontier look like when the weak arm is free?"* is not a question the original
   paper asked, and it is a defensible thesis result.
2. **`n_cal` stops being the binding constraint.** §3.4 showed the confidence bound width is set
   by `n_cal`, and that 1,000–2,000 per dataset was the requirement. RouterBench hands you that
   for free. **The statistical rigor becomes the contribution rather than a caveat.**
3. **MT-Bench arrives pre-judged**, which removes leakage vector 4 (judge self-preference,
   §2.5) entirely for Track A.

#### Track B — validation, $0 marginal, costs wall-clock

Demonstrates the router works with a real local model you control, not only with logged responses.

- `S` = **Qwen3-8B, 4-bit, via MLX** — prefer MLX over Ollama on Apple Silicon; expect roughly
  **~13 tok/s** for an 8B/9B at 4-bit on an M2 base chip — see Appendix A for the corrected numbers.
- ⭐ **MMLU is nearly free locally.** It is four forward passes per item for logprob scoring, not
  generation — 2,000 items runs in minutes. Do this first.
- ⚠️ **GSM8K is the expensive one — in wall-clock, not dollars.** 1,500 items × k=3 × ~250 output
  tokens ≈ 1.1M tokens ≈ **8–12 hours**. Plan it as one resumable overnight run; use `k=3` and
  `max_tokens=320`. **This is the new Phase-2 bottleneck and it is time, not money.**
- For the `L` arm in Track B, reuse RouterBench's GPT-4 per-item scores on overlapping
  GSM8K/MMLU items. Where item ids don't align, report Track B on a local-vs-published-frontier
  basis and say so explicitly.
- Track B runs as a background job **in parallel with Phases 4–6**. It is not on the critical path.

#### The $5–10 upgrade — flag this in the report

If any budget appears, 1,000 GSM8K + 1,000 MMLU items through **Sonnet 4.6 on the Batch API**
costs roughly **$2–3 at k=1** and **$7–10 at k=3** (batch rate $1.50/$7.50 per MTok; GSM8K ≈ 120
in / 250 out per item, MMLU ≈ 200 in / 5 out). That converts Track B into a genuine, current,
Anthropic-model result. It is the highest-value marginal dollar in the entire project.

### Revised phase table

| Phase | Change |
|---|---|
| **2 → 2′** | **Generation → acquisition.** Download and normalize RouterBench, verify per-item score and cost fields, map to the unified schema. ~0.5 day, $0. |
| **3** | Kill-check still applies — but now you get to **choose the pair by headroom**, maximizing `Q_L − Q_S`. That was impossible when you were paying to generate. |
| **4** | **F3 moves from stretch to feasible.** With 405k rows you are no longer data-limited, and DeBERTa-v3-small / ModernBERT-base trains overnight on MPS. Worth one run as the top ablation row. |
| **8** | **Dropped.** Thesis target, no serving requirement. |

### Thesis framing — three defensible contributions

1. A re-analysis of RouterBench under a **free weak arm** and **current 2026 frontier prices**.
2. **Distribution-free threshold selection** (Learn-then-Test) applied to router deployment — the
   field reports point estimates; you report a bound.
3. A **random/oracle-bracketed ablation ladder** (F0→F1→F2→F3, decision rule v1 vs cost-normalized
   v2) across both grouped and leave-one-subject-out splits.

---

## 0. Problem formalization

Let `q` be a query, `S` the small local model, `L` the frontier API model.

- Quality score `s_M(q) ∈ [0,1]` — model `M`'s score on `q` under a fixed grader.
- Cost `c_M(q) ∈ ℝ₊` — dollars to serve `q` with `M`.
- Router `r_τ: q → {S, L}` parameterized by a scalar threshold `τ`.

Realized quality and cost of a routing policy over a query distribution `D`:

```
Q(τ) = E_q[ s_{r_τ(q)}(q) ]
C(τ) = E_q[ c_{r_τ(q)}(q) ]
```

**Objective:**  `minimize C(τ)  subject to  Q(τ) ≥ Q_target`

Two admissible readings of "95% of frontier quality" — they are *very*
different and the project must commit to one up front:

| Reading | Target | Comment |
|---|---|---|
| **Absolute** | `Q ≥ 0.95 · Q_L` | Weak. If `Q_S` is already ≥ 0.95·`Q_L` on a dataset, the constraint is satisfied by routing 100% local. |
| **PGR (recommended)** | `Q ≥ Q_S + 0.95·(Q_L − Q_S)` | Strict. "Recover 95% of the *headroom* the frontier model buys you." This is the standard in the routing literature. |

**Decision: headline on PGR, report absolute as a secondary column.** PGR is the
honest metric and makes results comparable to RouteLLM.

### Router cost constraint

The router must be negligible next to `S`. Budget: **< 15 ms p99 on CPU,
< 100 MB RAM**. This rules out using a 7B model as the router and is why the
feature ladder below tops out at a frozen sentence encoder + MLP head.

---

## 1. Dataset & Feature Engineering Strategy

### 1.1 Dataset selection

Pick datasets that span two regimes: **verifiable** (cheap, objective, noiseless
labels) and **realistic** (actual user traffic distribution, subjective labels).

| # | Dataset | Size used | Grader | Why |
|---|---|---|---|---|
| **D1** | **GSM8K** | 1,500 (train split for router training, test split held out) | Exact match on final numeric answer | Difficulty is genuinely bimodal for 3–8B models. Objective labels. Well-known baseline. |
| **D2** | **MMLU** | 2,000 stratified over 57 subjects | Logprob argmax over A/B/C/D | Domain diversity. Provides the "which topics does S fail at" signal, which is the most learnable routing feature that exists. |
| **D3** | **LMSYS Chatbot Arena Conversations / Arena Human Preference 55k** | 1,500 single-turn English prompts | LLM-as-judge, pairwise with position swap | The **only** source of a realistic query distribution. Production savings are determined by the query mix, not by benchmark accuracy. |

Optional **D4: MBPP or HumanEval** (~400 items, execution-based pass@1) if the
deployment target includes code. Execution grading is the cheapest reliable
signal available — worth including if scope allows.

**Reporting rule: never average across D1–D3.** They have different graders and
different natural cost profiles. Report a Pareto curve per dataset plus a
"blended traffic" curve with explicitly stated mixture weights.

### 1.2 Label construction (do this right — it dominates classifier quality)

**Use soft labels.** A single greedy sample gives a 0/1 Bernoulli label with
enormous variance. Instead:

- **GSM8K / MBPP:** sample `k = 5` from `S` at `T = 0.7`, label
  `s_S = (#correct)/5 ∈ {0, .2, .4, .6, .8, 1}`. Sample `k = 3` from `L`.
  Cuts label variance ~5×; empirically worth more than any feature upgrade.
- **MMLU:** don't sample at all — score the four options by normalized
  log-likelihood and use `s = P(correct option)` as a soft score. Free and
  low-variance. Report hard accuracy separately.
- **Arena:** pairwise judge `S` vs `L` with **both orderings**; `s_S ∈ {0, 0.5, 1}`
  from {loss, tie, win}, `s_L = 1 − s_S`. Discard items where the two orderings
  disagree, or count them as ties — report the swap-disagreement rate as a
  judge-reliability number.

**Freeze everything.** Write a `manifest.json` alongside the corpus recording:
model version strings, decode params, prompt template hash, sampling seeds,
grader version, and total spend. Any change invalidates cross-run comparisons.

### 1.3 Feature ladder

Implement in order. Each tier is the baseline for the next; the *deltas* are the
interesting result for the report.

**F0 — Heuristics (~25 dims).** Token count, char count, sentence count, mean
word length, digit density, punctuation density, question-mark count, presence of
code fences / LaTeX / URLs / tables, imperative-verb count, Flesch-Kincaid grade,
type-token ratio, count of "explain / prove / derive / step by step" cue phrases,
count of numeric literals, max nesting depth of brackets.
→ Logistic regression. Purpose: establish that *any* signal exists, and give a
free interpretable baseline. Token count alone is a surprisingly strong router.

**F1 — TF-IDF + linear model.** Word 1–2 grams (min_df=2, max_features=50k)
**concatenated with char 3–5 grams** (max_features=50k). Char n-grams matter a
lot here — they capture math notation, code syntax, and morphology that word
tokens miss. → LogisticRegression / LinearSVC with L2. Fast, interpretable,
strong. Expect this to get most of the way there on D1/D2.

**F2 — Frozen sentence embeddings + MLP head (recommended default).**
Encoder: `BAAI/bge-base-en-v1.5` (768d) or `intfloat/e5-base-v2`; use
`all-MiniLM-L6-v2` (384d) if the 15 ms budget is tight. Frozen — no fine-tuning.
Head: `[d → 256 → GELU → Dropout(0.1) → 2]`.
**Concatenate F0 onto the embedding** (`[emb ‖ z-scored heuristics]`) — the
combination reliably beats either alone because length/complexity signals are
poorly represented in semantic embedding space.

**F3 — Fine-tuned cross-encoder (stretch).** `ModernBERT-base` or
`DeBERTa-v3-small` fine-tuned end-to-end on the routing target. Highest ceiling,
needs a GPU and ≥10k labeled queries. Only pursue if F2 is clearly data-limited
rather than signal-limited.

**F4 — Small-model self-signals (optional, changes the architecture).**
Mean token logprob, predictive entropy, and max-margin over the **first 32
generated tokens** of `S`, plus a verbalized-confidence probe.
⚠️ This makes routing non-free: you pay a `S` prefix before deciding. Frame it
explicitly as a **cascade** variant (generate-then-verify) rather than a router,
and price the prefix into `C(τ)`. It is usually the single strongest feature —
worth an ablation row even if it doesn't ship.

**Recommended headline config: F2 + F0, `bge-base-en-v1.5`, MLP head.**

---

## 2. Router Classifier Architecture & Strategy

### 2.1 Objective function — three candidates

| Option | Target | Verdict |
|---|---|---|
| **(a) Binary success of S** | `y = 1[s_S(q) ≥ θ]` | ❌ Ignores whether `L` also fails. Escalates queries nobody can solve — pure wasted spend. |
| **(b) Relative gap regression** | `y = s_L(q) − s_S(q) ∈ [−1,1]` | ✅ Directly models what escalation buys. Handles both-fail (gap≈0 → stay local) and both-succeed (gap=0 → stay local). |
| **(c) End-to-end cost-sensitive decision** | `min E[loss] + λ·E[cost]` | ❌ Elegant but bakes `λ` into the weights, so you lose the ability to sweep `τ` post hoc and re-derive the whole Pareto curve for free. Also unstable at this data scale. |

**Chosen: (b), implemented as a two-head model.**

```
x = [ encoder(q) ‖ heuristics(q) ]        # frozen, 768 + 25 dims
h = MLP(x)                                 # 256-d shared trunk
ŝ_S = σ(w_S · h + b_S)                    # predicted quality of small model
ŝ_L = σ(w_L · h + b_L)                    # predicted quality of frontier model
Δ̂(q) = ŝ_L(q) − ŝ_S(q)                    # routing score
```

Loss: `L = BCE(ŝ_S, s_S) + BCE(ŝ_L, s_L) + β · MSE(ŝ_L − ŝ_S, s_L − s_S)`
with `β ≈ 0.5`. (Soft-label BCE is valid for targets in [0,1] and is better
behaved than MSE on bounded scores.) The auxiliary gap term keeps the *difference*
well-fit even when both marginals are easy.

**Why two heads rather than one gap head:** swapping the frontier model requires
retraining only `w_L`; adding a middle tier (Haiku) is one more head; and
`ŝ_S`, `ŝ_L` are individually interpretable and individually calibratable.

### 2.2 Decision rule

**v1 (baseline):**  route to `L` iff `Δ̂(q) > τ`.

**v2 (recommended) — cost-normalized gain:**

```
G(q) = Δ̂(q) / ( ĉ_L(q) − ĉ_S(q) )        route to L iff G(q) > τ
```

Escalating a 4,000-token query costs far more than a 200-token one, so the naive
rule over-spends on long queries. `G` is "expected quality points per dollar,"
and thresholding it is the Lagrangian-optimal rule for the constrained problem.
Expect a visibly better Pareto frontier whenever query lengths are heterogeneous
— which they are in Arena data. **Ship v1 first, then v2 as an ablation; the
delta is a genuine contribution.**

`ĉ_L(q)` needs a predicted *output* length. Simplest workable estimate: a second
tiny regression head predicting `log T_out`, trained on the frozen corpus.

### 2.3 Calibration

Raw head outputs are not calibrated, and `τ` must be transferable across
datasets. After training, fit **isotonic regression** on the calibration split
mapping `ŝ_M → E[s_M | ŝ_M]` for each head independently. Deliverable: reliability
diagrams for both heads + Expected Calibration Error (15 bins) before/after.
Isotonic is preferred over Platt here because the miscalibration is usually
non-sigmoidal at the extremes.

### 2.4 Training setup

| Item | Value |
|---|---|
| Encoder | frozen; embeddings precomputed once → `features.parquet` |
| Head | 768+25 → 256 (GELU, dropout 0.1) → 2 |
| Optimizer | AdamW, lr 1e-3 (head-only) / 2e-5 (if F3 fine-tuning), wd 0.01 |
| Schedule | cosine, 10% warmup |
| Batch / epochs | 64 / 30 with early stopping (patience 5) |
| Model selection | val **Spearman ρ(Δ̂, Δ)** — *not* accuracy. Ranking quality is what the threshold sweep consumes. |
| Imbalance | if `S` succeeds >80% of the time, weight samples by `|Δ|` or use focal loss on the binary variant. Always report per-bucket error. |
| Seeds | 5 seeds; report mean ± std on every headline number |

### 2.5 Split logic — leakage avoidance

This section is the difference between a believable result and a fake one. Four
distinct leakage vectors:

**(1) Near-duplicate leakage.** GSM8K contains templated/paraphrased problems;
Arena contains repeated and near-repeated prompts. A near-duplicate straddling
train/test lets the router memorize an answer key.
→ **Fix:** before splitting, run MinHash LSH on 5-gram shingles (Jaccard ≥ 0.8)
*and* embedding-cosine dedup (≥ 0.95). Build a connected-components graph over
near-duplicates and assign each **component** a single `group_id`. Split by
`group_id`, never by row. Emit a dedup report (# components collapsed) as an
artifact.

**(2) Group / domain leakage.** With a random split, the router can memorize
`subject → difficulty` on MMLU's 57 subjects. That is legitimate signal for
in-domain deployment but badly inflates OOD estimates.
→ **Fix:** report **both** a random grouped split and a **leave-one-subject-out**
split. The gap between them *is* your generalization headroom and is one of the
more interesting results in the report.

**(3) Threshold leakage — the big one.** Choosing `τ` on the same data used to
report the 95% recovery guarantees optimistic savings.
→ **Fix: three-way split, `60 / 20 / 20`.**

```
train (60%)  →  fit encoder head
cal   (20%)  →  isotonic calibration AND τ selection  ← the only data τ ever sees
test  (20%)  →  touched exactly once, at the end, for the reported numbers
```

`n_cal ≥ 1,000` is a hard floor (see §3.4 — the confidence bound width is set by
`n_cal`, not `n_train`).

**(4) Judge / label leakage.** If the LLM judge is the same family as `L`, its
self-preference bias inflates `s_L` and therefore inflates the measured headroom.
→ **Fix:** use a judge from a different family than both routed models where
possible; always position-swap; and report judge–human agreement on the Arena
subset where human preference labels exist. If a cross-family judge isn't
available, state the bias explicitly and treat Arena PGR as an upper bound.

**Small-data protocol.** If total corpus < 5k after dedup, replace the single
split with **nested grouped CV**: 5 outer folds (grouped) for reporting, inner
fold within each training portion for `τ`. Report mean ± std across outer folds.

---

## 3. Metric & Pareto Curve Evaluation Pipeline

### 3.1 Quality metric

Per-dataset grader, all normalized to `[0,1]`:

| Dataset | Grader | Score |
|---|---|---|
| GSM8K | numeric normalization (strip `$`, `,`, trailing `.0`), exact match on final answer | pass@1 over k=5 |
| MMLU | length-normalized logprob argmax over A/B/C/D | `P(correct)` (soft) + hard accuracy |
| Arena / MT-Bench | pairwise LLM judge, both orderings | win=1, tie=0.5, loss=0 |
| MBPP (opt.) | sandboxed execution against unit tests | pass@1 over k=5 |

**Grader validation gate:** hand-check 50 graded items per grader; require ≥95%
agreement with your own judgment before trusting any downstream number. Bad
graders have killed more routing projects than bad classifiers.

Routed quality at threshold `τ`, over `n` test queries:

```
Q̂(τ) = (1/n) Σᵢ [ s_L,i · 1{Gᵢ > τ}  +  s_S,i · 1{Gᵢ ≤ τ} ]
```

### 3.2 Cost metric

```
c_L(q) = ( p_in · T_in(q) + p_out · T_out(q) ) / 1e6
```

**Report two cost models — this is an honesty requirement, not a nicety:**

- **M1 — Pure API spend.** `c_S = 0`. This is the headline "cost savings" number
  and the one the field quotes.
- **M2 — Fully loaded.** `c_S = (GPU $/hr) / (queries/hr throughput)`, plus
  router inference cost, plus (for cascade variants) the wasted `S` prefix on
  escalated queries. **Compute the crossover point explicitly**: at a low enough
  throughput, "free" local inference is more expensive than Haiku at $1/MTok.
  That crossover is a genuinely useful finding for the report.

**Current Anthropic list prices (USD per 1M tokens, Aug 2026):**

| Model | Input | Output |
|---|---|---|
| Claude Opus 5 | $5 | $25 |
| Claude Sonnet 5 | $2 | $10 |
| Claude Sonnet 4.6 / 4.5 | $3 | $15 |
| Claude Haiku 4.5 | $1 | $5 |

⚠️ **Note on your spec:** Claude 3.5 Sonnet is retired. The `$3 / $15` you assumed
is the current **Sonnet 4.6 / 4.5** rate — use Sonnet 4.6 as `L` and your cost
model is unchanged. Sonnet 5 at `$2 / $10` is the cheaper alternative.

⚠️ **Baseline fairness:** the Batch API is **50% off** and prompt-cache reads are
**0.1×** input price. A router that beats a *naive* all-frontier baseline 3× may
only beat a batch-optimized baseline 1.5×. Report savings against both an
unoptimized and a batch-optimized frontier baseline.

Report cost as **$ per 1,000 queries** and as the ratio `C(τ) / C_L`.

### 3.3 Headline metrics

Adopt the RouteLLM metric set so results are comparable to published work:

```
PGR(τ)   = ( Q(τ) − Q_S ) / ( Q_L − Q_S )          # performance gap recovered
CPT(95%) = min{ α : PGR ≥ 0.95 }                    # min frontier-call rate for 95%
APGR     = ∫₀¹ PGR(α) dα                            # single scalar, router-vs-router
Savings  = 1 − C(τ*) / C_L    at PGR = 0.95
```

where `α` = fraction of queries sent to `L`. **`CPT(95%)` is precisely the
quantity your project targets.**

**Mandatory reference curves on every Pareto plot** — a router curve without
these is uninterpretable:
- **Random router** (route a random `α` fraction to `L`) — the straight-line
  floor. Any router below this is worse than a coin flip.
- **Oracle router** (route using true `Δ`, not `Δ̂`) — the achievable ceiling.
  The router-to-oracle gap tells you whether to invest in features or accept
  irreducible noise.

### 3.4 Optimal threshold `τ*` — the math

Work on the **calibration split** `D_cal = {(qᵢ, s_S,i, s_L,i, Gᵢ, c_S,i, c_L,i)}`,
`i = 1..n`.

**Step 1 — exact sweep, O(n log n), no search.**
`Q̂(τ)` and `Ĉ(τ)` are step functions changing only at the `n` observed values of
`G`. Sort by `G` descending, compute prefix sums of `s_L` and suffix sums of
`s_S` (same for costs), and read off `(Q̂, Ĉ)` at every breakpoint exactly. There
are at most `n` distinct candidate thresholds; enumerate all of them.

**Step 2 — monotonize.** Because `Δ̂` is imperfect, `Q̂(τ)` can be locally
non-monotone. Take the **upper-left concave envelope** of the `(Ĉ, Q̂)` point
cloud (a Graham-scan style upper hull). Points strictly inside the hull are
dominated and are never optimal. Equivalently, apply isotonic regression to
`Q̂(·)`. Report the envelope as *the* Pareto frontier.

**Step 3 — naive selection.** Since both `Q̂` and `Ĉ` are non-increasing in `τ`,
the cheapest feasible policy is the *largest* feasible threshold:

```
τ_naive = max{ τ : Q̂(τ) ≥ Q_target },   Q_target = Q_S + 0.95·(Q_L − Q_S)
```

**Step 4 — conservative selection (this is the part that makes the result real).**
`Q̂(τ_naive)` is optimistically biased because `τ` was selected to maximize it.
Replace the point estimate with a one-sided lower confidence bound:

```
τ* = max{ τ : LCB_{1−δ}( Q(τ) ) ≥ Q_target },   δ = 0.05
```

Three ways to compute `LCB`, in increasing order of tightness:

1. **Hoeffding** (`s ∈ [0,1]`): `LCB = Q̂ − √( ln(1/δ) / (2n) )`.
   Distribution-free but loose: at `n = 1500, δ = 0.05` the slack is ≈ **0.032** —
   3.2 quality points, often larger than the entire `Q_L − Q_S` headroom.
2. **Empirical Bernstein / Wilson** — uses the observed variance; typically 2–3×
   tighter when scores are near 0 or 1 (which MMLU and GSM8K are).
3. **Paired bootstrap (recommended).** ⭐ Because `s_L` and `s_S` are measured on
   the *same* queries, evaluate the per-query **difference**
   `dᵢ(τ) = s_{r_τ(qᵢ)},i − s_L,i` and bootstrap the mean of `d`. The paired
   variance is far smaller than the variance of either arm — commonly 3–5× tighter
   than Hoeffding. `B = 2000` resamples, take the 5th percentile.

**Multiplicity correction.** `τ` is selected over `m` breakpoints simultaneously,
so a naive per-`τ` bound doesn't control the actual risk. Two options:
- Union bound: use `δ/m`. Simple, conservative.
- **Fixed sequential testing (Learn-then-Test / RCPS, Angelopoulos et al.).**
  Because the family is *monotone* in `τ`, test thresholds in order from most
  conservative to most aggressive and stop at the first rejection. This controls
  family-wise error at `δ` with **no** Bonferroni penalty. This is the correct
  tool and is a strong differentiator for the writeup.

**Sizing implication:** the bound width is set by `n_cal`. Budget
`n_cal ≥ 1,000–2,000` *per dataset* you intend to quote a guarantee on. If you
can only afford 300 calibration queries, do not claim a guarantee — report the
point estimate with a CI and say so.

**Step 5 — report on `test`, once.** Apply the frozen `τ*` to the untouched test
split. Report `Q_test`, `PGR_test`, `C_test`, savings ratio, and `α_test`
(realized frontier-call rate), each with a paired-bootstrap CI. Expect
`PGR_test < PGR_cal`; the size of that drop is itself a result worth reporting.

### 3.5 Plot spec (`plot_pareto.py`)

- **Panel A:** x = mean cost ($/1k queries, log scale) · y = `PGR`. Curves:
  router (v1), router (v2 cost-normalized), random baseline, oracle ceiling.
  Horizontal dashed line at `PGR = 0.95`; vertical marker at `τ*`; annotate the
  savings multiple at the intersection.
- **Panel B:** x = `α` (% calls to frontier) · y = `PGR` — this is the CPT plot.
- **Panel C:** small multiples, one per dataset (D1/D2/D3), same axes.
- **Panel D:** router score distribution `Δ̂`, split by whether `S` actually
  succeeded — the visual sanity check that the classifier separates at all.
- Shaded bands = paired-bootstrap 95% CI on every curve.

---

## 4. Phased Implementation Roadmap

### 4.1 File structure

```
cost-aware-router/
├── configs/
│   ├── models.yaml            # model ids, price table, decode params
│   ├── datasets.yaml          # dataset ids, sample counts, split seeds
│   └── experiment.yaml        # feature config, τ targets, δ, bootstrap B
├── router/
│   ├── schema.py              # Query / Response / Score dataclasses + parquet IO
│   ├── data.py                # download + normalize D1..D4 → corpus.parquet
│   ├── dedup.py               # MinHash LSH + embedding dedup → group_id
│   ├── splits.py              # grouped 60/20/20, LOSO, nested CV
│   ├── generate.py            # run S and L over corpus (resumable, hash-cached)
│   ├── graders.py             # exact_match, mcq_logprob, llm_judge, exec_test
│   ├── score.py               # apply graders → scores.parquet
│   ├── features.py            # F0 heuristics, F1 tfidf, F2 embeddings, F4 logprobs
│   ├── models.py              # LogReg / TwoHeadMLP / cross-encoder; fit/predict/save
│   ├── calibrate.py           # isotonic per head + reliability diagrams
│   ├── costs.py               # token counting × price table → $ per query
│   ├── threshold.py           # exact sweep, concave envelope, LCB, LTT selection
│   ├── eval.py                # Q, C, PGR, CPT, APGR, paired bootstrap CIs
│   └── serve.py               # FastAPI /route (optional, Phase 8)
├── scripts/
│   ├── 00_build_corpus.py
│   ├── 01_acquire_routerbench.py    # was 01_generate_responses.py (Rev A)
│   ├── 02_score_responses.py
│   ├── 03_train_router.py
│   ├── 04_select_tau.py
│   └── 05_plot_pareto.py
├── plots/plot_pareto.py
├── tests/                     # graders, cost math, τ solver, envelope, split integrity
├── artifacts/                 # *.parquet, checkpoints, manifest.json (gitignored)
├── report/                    # figures + writeup
├── DESIGN.md
└── README.md
```

**Data contracts between stages** (each stage reads one parquet, writes one — no
hidden state, every stage independently re-runnable):

```
corpus.parquet     : qid, source, group_id, split, prompt, reference, meta
responses.parquet  : qid, model, sample_idx, text, n_in, n_out, logprobs, latency_ms
scores.parquet     : qid, model, score, grader, grader_version
features.parquet   : qid, feature_set, vector
predictions.parquet: qid, s_hat_S, s_hat_L, delta_hat, c_hat_L
```

### 4.2 Execution order & phase gates

| Phase | Work | Est. | **Gate — do not proceed until** |
|---|---|---|---|
| **0** | `costs.py` + `schema.py` + tests. Price table from `models.yaml`. Dry-run spend estimator. | 0.5 d | You can print projected `$` for the full generation run *before spending anything*. |
| **1** | `data.py`, `dedup.py`, `splits.py`. **Pilot corpus: 2,000 queries** (800 GSM8K / 800 MMLU / 400 Arena). | 1 d | Dedup report emitted; assert **zero** `group_id` appears in two splits. |
| **2** | `generate.py` — the expensive, irreversible step. Content-hash cache; resumable; k=5 from `S`, k=3 from `L`. **Use the Batch API for `L` (50% off)** — this is exactly the offline use case for it. | 1–2 d | `responses.parquet` complete + `manifest.json` with model versions, seeds, and actual spend. |
| **3** | `graders.py`, `score.py`. Hand-validate 50 items per grader. | 0.5 d | ⛔ **Kill-check:** compute `Q_S` and `Q_L` per dataset. If `Q_L − Q_S < 0.05` on a dataset, **there is no routing headroom — drop it or choose a smaller `S`.** Catch this now, not in Phase 6. |
| **4** | `features.py`, baselines F0 → F1 → F2. Grouped CV, report AUC + Spearman ρ per tier. | 1 d | F2 beats F0 by a margin exceeding seed noise. If not, the signal isn't in the query text — revisit the model pair or add F4. |
| **5** | `models.py` two-head training, `calibrate.py`. 5 seeds. | 1 d | Reliability diagrams + ECE before/after isotonic; val Spearman reported. |
| **6** | `threshold.py`, `eval.py`. Sweep, envelope, paired bootstrap, LTT. | 1 d | `CPT(95%)` and savings-at-PGR-0.95 on the **untouched** test split, with CIs. |
| **7** | `plot_pareto.py`, ablation table, error analysis of the 20 costliest routing mistakes. | 1 d | Pareto figure with random + oracle reference curves. Report drafted. |
| **8** | *(optional)* `serve.py`, shadow-mode logger, p99 latency measurement. | 1 d | Router p99 < 15 ms on CPU. |

**Scale-up decision point:** run Phases 1–7 end-to-end on the 2,000-query pilot
first. Only after the pipeline produces a valid Pareto curve should you expand to
the full ~10k corpus. The pilot costs a few dollars; a broken full run costs a
weekend.

### 4.3 Ablation table to fill in (this is the report's core result)

| Config | AUC | ρ(Δ̂,Δ) | CPT(95%) | Savings @ PGR .95 | APGR |
|---|---|---|---|---|---|
| Random router | 0.50 | 0.00 | 0.95 | ~5% | 0.50 |
| F0 heuristics + LogReg | | | | | |
| F1 TF-IDF + LogReg | | | | | |
| F2 embeddings + MLP | | | | | |
| F2 + F0 (headline) | | | | | |
| + cost-normalized rule (v2) | | | | | |
| + F4 logprob cascade | | | | | |
| Oracle router | 1.00 | 1.00 | | | 1.00 |

---

## 5. Risks & expectation-setting

1. **No headroom.** If `S` and `L` score similarly on a dataset, no router can
   help. → Phase 3 kill-check. Note that a *strong* small model is bad for this
   project on easy benchmarks: Phi-4-mini reportedly scores ~88.6% on GSM8K,
   leaving only ~7 points of headroom vs. a frontier model. MMLU (~67% vs ~88%)
   and open-ended Arena traffic have much more room.
2. **Target ambiguity.** Absolute-95% vs PGR-95% differ enormously. → Fixed in §0.
3. **Overfit `τ`.** → Three-way split + paired-bootstrap LCB + LTT (§3.4).
4. **Unfair baseline.** Batch API and prompt caching already cut frontier spend
   50–90%. → Report savings against both baselines (§3.2).
5. **Distribution shift.** Benchmarks ≠ production traffic. → Arena is the
   realism proxy; never average it with GSM8K/MMLU.
6. **Judge self-preference.** → Cross-family judge + position swap (§2.5).
7. **Local serving economics.** At low throughput, "free" local inference costs
   more than Haiku 4.5 at `$1/$5`. → Compute the crossover in cost model M2.

### Realistic expectations

RouteLLM (GPT-4-Turbo vs Mixtral-8x7B) reported, at 95% performance: **~85%**
cost reduction on MT-Bench, **~45%** on MMLU, **~35%** on GSM8K.

So: **2–5× savings is realistic on open-ended, Arena-like traffic. On GSM8K alone,
expect closer to 1.5×.** Frame the project's headline number on a blended or
Arena-weighted traffic mix, and be explicit about the mixture weights.

### Stretch: 3-tier cascade

`local → Haiku 4.5 ($1/$5) → Sonnet 4.6 ($3/$15)`. Given Haiku's price point, the
two-tier local-vs-Sonnet framing may well be dominated by a three-tier one. The
two-head architecture extends to `k` heads with no redesign — add `ŝ_Haiku` and
solve a 2-threshold version of the same constrained problem. Good "future work"
section, or a genuine result if time allows.

---

## 6. Delivery — merging the two roadmaps, and splitting four ways

**Verdict on `implementation_plan_2.md`: right shape, wrong statistics.** Roughly 60% survives —
keep its scaffolding, stack and scope discipline; replace its labelling, splitting and threshold
logic. It is a good plan for building *a* router and a weak plan for defending *a result*.

### Keep — including two things it gets right that this document did not

| From plan 2 | Why it stays |
|---|---|
| **"Get the data without generating it"** (its Phase 2) | Same conclusion as Rev A, reached independently. Strong instinct — the decision the project hinges on. |
| Dual-stream structural + semantic features | Exactly the F0 + F2 headline config in §1.3. |
| **XGBoost baseline** *(adopt)* | Not in this document, and it should have been. Gradient boosting on fused structural + embedding features is a strong, cheap baseline — it belongs between F1 and F2 in the ablation ladder, and may beat the MLP at this data size. |
| Concrete stack, `requirements.txt`, flat `src/` | Lower ceremony than §4.1; a team can start Monday. Keep the shape, keep this document's module boundaries inside it. |
| **FastAPI demo** (its Phase 6) *(reinstate)* | Rev A dropped serving for a thesis target. With four people it is affordable again, and a live routing demo with a running cost counter is worth a lot at a defence. **Phase 8 returns.** |

### Fix — eight defects, in descending order of cost to you

1. **The label collapses.** `Y = 1 if small ≥ 0.90 × large`. On GSM8K/MMLU per-item scores are 0 or 1,
   so the rule reduces to *"small correct OR large wrong"* — the 0.90 does nothing, and the ratio is
   undefined when the large model scores 0. Worse, binarising at training time discards the magnitude
   the threshold sweep needs. → Regress `Δ = s_L − s_S` with two heads (§2.1). Never binarise.
2. **τ is swept on the test set** (its Phase 5.2). Threshold leakage — the headline savings are
   guaranteed optimistic. → Three-way split, τ on `cal` only (§2.5).
3. **No dedup, no grouped split.** Random 80/20 over Arena-style data leaks near-duplicates.
   → MinHash + embedding dedup, split by component (§2.5).
4. **No reference curves on the Pareto plot.** Without random and oracle brackets you cannot tell
   whether the router beat a coin flip. → §3.3.
5. **No headroom kill-check.** If `Q_L − Q_S` is small nothing works, and you learn it in its Phase 5.
   → §4 Phase 3 gate.
6. **No cost model.** "Calculate Total Cost" with no token accounting or price table. → `costs.py`
   first, with tests (§4 Phase 0).
7. **Ollama + Llama 3 8B on an M2 Air** (~13 tok/s, dated), `openai` with no budget. → Appendices A, B.
8. **The notebook is the home of the headline result.** → `threshold.py` as a tested module; the
   notebook calls it.

Net: its Phases 1–4 survive with fixes 1, 3 and 6. Its Phase 5 is replaced wholesale by §3. Its
Phase 6 returns as Phase 8.

### Splitting four ways

> **The principle: split by artifact, not by phase.** Phases are sequential — four people on
> sequential phases means three idle. The seams are the parquet contracts, already defined in §4.1.

> **The scheduling trick that makes it work.** In week 2, generate a **synthetic `scores.parquet`**
> with a known planted signal — a difficulty variable driving both `s_S` and `s_L` with a
> controllable gap. From week 3 everyone downstream develops against it, before real data exists.
> The evaluation module can be finished and unit-tested *entirely* on synthetic data, where you know
> the right answer. This defuses the classic four-person failure: the statistics person starting last
> and the project jamming in the final fortnight.

| Role | Owns these modules | Owns this contract | Thesis chapter |
|---|---|---|---|
| **A · Data & Corpus** *(most careful person)* | `data.py`, `dedup.py`, `splits.py`, `01_acquire_routerbench.py` | `corpus.parquet`, `scores.parquet` | Datasets, deduplication, split methodology |
| **B · Features & Router** | `features.py`, `models.py`, `calibrate.py`, `graders.py` | `features.parquet`, `predictions.parquet` | Feature engineering and router architecture |
| **C · Evaluation & Statistics** *(strongest stats person)* | `costs.py`, `threshold.py`, `eval.py`, `plot_pareto.py` | the Pareto figure, the ablation table | Cost modelling, threshold selection, distribution-free guarantees |
| **D · Systems & Demo** | `generate.py`, `serve.py`, CI, `manifest.json` | the inference layer, reproducibility | Local inference, serving architecture, live prototype |

**C is the role to staff first.** Everything in §3 — exact sweep, concave envelope, paired bootstrap,
Learn-then-Test — *is* the contribution. A bug in the τ solver silently invalidates every number in
the paper and nobody downstream will catch it. A is second-most consequential: dedup and split bugs
are invisible and fatal.

### 16-week schedule

| Weeks | A · Data | B · Router | C · Statistics | D · Systems |
|---|---|---|---|---|
| 1–2 | contracts | contracts | **cost model** | repo · CI |
| 3–5 | **RouterBench · dedup · splits** | feature ladder (synthetic) | **τ machinery (synthetic)** | inference clients · MLX bench |
| 6–8 | corpus frozen | two heads · calibration | **first Pareto** | generation infra |
| 9–11 | Track B runs | ablation ladder | **bootstrap · LTT · LOSO** | generation infra |
| 12–13 | support | ablation ladder | **bootstrap · LTT · LOSO** | FastAPI demo |
| 14–15 | write-up | write-up | figures | FastAPI demo |
| 16 | write-up | write-up | write-up | defence |

**Bold = critical path.** If a week slips, it must not slip in a bold cell.

### Gates

| Week | Gate | Evidence |
|---|---|---|
| 2 | **Contracts frozen** | All five parquet schemas in `schema.py`; `costs.py` passing tests; synthetic `scores.parquet` with a planted signal committed. Nobody blocked after this. |
| 5 | **Pipeline green on synthetic** | End-to-end run on fake data produces a Pareto curve. It will be nonsense — but plumbing, τ solver and plot code are proven before real data arrives. |
| 8 | **First Pareto on real data** | Pair chosen by headroom; `Q_L − Q_S` reported per dataset; §4 Phase 3 kill-check passed. **If this slips, cut scope, not rigour.** |
| 13 | **Statistics locked** | `CPT(95%)` and savings on the untouched test split, with paired-bootstrap CIs and the Learn-then-Test bound. No number changes after this week. |
| 15 | **Freeze** | Demo running, figures final, draft complete. Week 16 is buffer and rehearsal — not work. |

### Four rules that keep a four-person ML project from failing

1. **Rotate an integrator weekly.** One named person merges to `main` and runs the full pipeline end
   to end every Friday. The role rotates so all four understand the whole system by the defence.
2. **Artifacts are not git.** Parquets and checkpoints are gitignored — use a shared Drive folder or a
   private Hugging Face dataset repo, named by content hash. Never emailed, never hand-renamed.
3. **The frozen-corpus rule is a team rule.** Nobody regenerates responses without bumping the
   manifest version. One silent regeneration and every earlier comparison is meaningless.
4. **Every owner writes their own tests** — and C's matter most. A wrong τ solver produces plausible,
   publishable, entirely false numbers.

## Appendix A — Choosing the local model `S`

> **Correction to the Rev A figure.** The "25–40 tok/s" quoted in Rev A is an M3/M4-class number.
> The **M2 base chip has ~100 GB/s of memory bandwidth**, and decode speed on a bandwidth-bound
> machine is roughly `bandwidth ÷ model bytes`. A 9B at 4-bit therefore runs nearer **~13 tok/s**,
> not 30 — and the fanless Air throttles 15–25% under sustained multi-hour load. Every wall-clock
> estimate below is rebuilt on the corrected number.

Two constraints decide this, and neither is the obvious one.

**Constraint 1 — bandwidth, not memory, is the ceiling.** 16 GB is plenty to *load* a 9B model; it
just runs slowly. The binding resource for a project needing thousands of completions is tokens per
second, set by bandwidth divided by weight bytes. Halving the model roughly doubles throughput.

**Constraint 2 — you want `S` weak, not strong.** The router only has something to learn when
`Q_L − Q_S` is large (§5, risk 1). A small model that is already good at your benchmark destroys the
experiment. Both constraints point the same way: **down the size ladder, not up.**

| Model | 4-bit size | est. tok/s | Track B GSM8K | License | Why this one |
|---|---|---|---|---|---|
| **Qwen3.5-4B** *(primary)* | ~2.5 GB | 35–40 | ~9 h | Apache 2.0 | Modern, permissive, fast enough for one overnight run with memory left for the bge encoder and dedup. The default pick. |
| **Llama 3.2 3B** | ~2.0 GB | 45–50 | ~7 h | Llama CL | **Max headroom, fastest turnaround.** Weakest of the set (~63% MMLU, ~78% GSM8K) → largest `Q_L − Q_S`, cleanest routing signal, best-documented tooling. |
| **Phi-4-mini 3.8B** | ~2.3 GB | 35–40 | ~9 h | MIT | **The contrast case.** ~88.6% GSM8K nearly closes the math gap by itself. Run it *deliberately* to demonstrate risk 1 empirically. Cleanest license of the five. |
| **Gemma 3 4B** | ~2.8 GB | 30–35 | ~10 h | Gemma | **Robustness column.** Different pretraining lineage, 128K context, vision-capable — checks results aren't a Qwen artifact. |
| **Qwen3.5-9B** | ~5.5 GB | ~13 | ~24 h | Apache 2.0 | **Upper bound, only with a cut corpus.** Loads fine in 16 GB but the wall-clock kills a full Track B. Use on a 400-item subset to show the headroom trend continues. |

### Turn risk 1 into a result — the headroom sweep

Run Llama 3.2 3B → Qwen3.5-4B → Phi-4-mini against the same frontier arm and plot `CPT(95%)` against
the measured `Q_L − Q_S`. Three overnight runs, no extra spend, and it converts the project's biggest
risk into a quantified finding: *how much headroom does a router need before it pays for itself?*
That is a better chapter than any single-pair result.

### Two levers that beat any model choice

1. **Batch your generation.** On a bandwidth-bound machine weights are read once per batch, so MLX
   batched decoding at batch 8 gives roughly 3–5× effective throughput. Do this before optimizing
   anything else — worth more than dropping a model size.
2. **Or don't run it locally at all.** Serve the same open weights on a free inference tier
   (Appendix B, row 3) and a 24-hour laptop run becomes ~3 hours. Since `c_S = 0` is a modeling
   choice in Rev A, not a measurement, nothing in the paper changes.

### Measure before planning around these numbers

```bash
mlx_lm.generate --model mlx-community/Qwen3.5-4B-Instruct-4bit \
  --prompt "Solve: 17*23. Think step by step." --max-tokens 256
```

Record the reported tok/s in `manifest.json`. Every wall-clock figure above is derived from memory
bandwidth, not measured on your machine — one command replaces all of them with facts.

## Appendix B — Choosing the API arm

> **The move that makes a zero-budget thesis rigorous.** *The model you run and the model you price
> do not have to be the same model.* Rev A already re-prices logged RouterBench responses at 2026
> rates, so the cost axis is a modeling layer, not a receipt. Run on free tiers; price `c_L` at
> frontier list rates; state the separation plainly in Methods.

| Provider / model | Free tier | List price /1M | Role here |
|---|---|---|---|
| **Google Gemini** — 3 Flash · 3.5 Flash *(primary free arm)* | Yes — AI Studio, ~15 RPM / ~1,500 RPD, no card | $0.50/$3.00 · $1.50/$9.00 | **The frontier arm for Track B.** 1,500 RPD covers a full 1,500-item GSM8K pass in one day, free. Disclose: free-tier prompts may be used to improve Google's products; Pro models are paid-only. |
| **GitHub Models** — GPT-5 / GPT-4o class | Yes — GitHub account, ~10–15 RPM / 50–150 RPD | — | **Quality anchor.** Low daily cap, but genuinely frontier-class at $0. Use a 200-item subset to verify Flash-derived labels track a true frontier model. Student Developer Pack raises limits. |
| **Groq** · Cerebras — llama-3.1-8b, qwen3-32b, gpt-oss | Yes — 14,400 RPD, 6,000 TPM (Cerebras ~30–60k TPM) | — | **Serve the small arm, not the frontier one.** Open weights, so not a frontier model — but 6,000 TPM turns a 24-hour laptop run into ~3 hours, and it is the natural middle tier for the three-tier cascade in §5. |
| **Anthropic** — Sonnet 4.6 · Haiku 4.5 | No | $3/$15 · $1/$5 | **The pricing reference.** Keep Sonnet 4.6 as the `c_L` constant — matches your original spec, credible frontier price. If budget appears, ~$2 buys 1,000 GSM8K items on Batch at k=1. |
| **OpenAI** — GPT-5.6 Terra · GPT-5 nano | No | $2.50/$15 · $0.05/$0.40 | **Cheapest paid escape hatch.** 1,000 GSM8K items ≈ $2 on Terra via Batch, or ~$0.11 on nano. Nano is far too weak to be `L`, but an excellent extra *middle* tier. |

### The recommended stack

| | |
|---|---|
| **Track A — primary, $0** | RouterBench. No API call at any point. |
| **Track B — small arm** | Qwen3.5-4B at 4-bit via MLX, batched. Fall back to Groq free if wall-clock bites. |
| **Track B — frontier arm** | Gemini 3 Flash free tier for volume; GitHub Models on 200 items as the quality anchor. |
| **Cost model** | `c_L` priced at Sonnet 4.6 ($3/$15), `c_S = 0`. Report the run-vs-price separation explicitly. |

> **Read the terms before building on a free tier.** Cohere's trial keys are **non-commercial only**
> (fine for academic work — but say so). Google's free tier permits training on your prompts.
> Cerebras' free catalogue has churned before. None is disqualifying; all belong in a footnote, and
> none should be discovered the week before submission.

## Appendix C — Answered questions

1. **Local compute** — GPU/VRAM or Apple Silicon unified memory? Sets `S` size
   and whether F3 fine-tuning is feasible.
2. **API access & budget** — which Claude model as `L`, dollar budget for the
   one-time generation run, and is the Batch API available?
3. **Deployment target** — academic benchmark report, or must serve real traffic?
   Determines whether Arena is primary or secondary and whether Phase 8 exists.
4. **Judge availability** — is a non-Anthropic judge model available for Arena
   scoring, or do we accept and document self-preference bias?

## Appendix C — References

- Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data*, 2024 — PGR / CPT / APGR metrics, Arena-trained routers.
- Angelopoulos et al., *Learn then Test* / *Risk-Controlling Prediction Sets* — distribution-free threshold selection.
- Chen et al., *FrugalGPT* — cascade framing and the generate-then-verify variant.
- Ding et al., *Hybrid LLM* — quality-gap regression routing.
- *RouterBench* — standardized router benchmark and reporting conventions.
