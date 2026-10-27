# Chapter 2 — Related Work

> **DRAFT for Anirvan Ranjan (task T54).** Citations checked against the papers' arXiv / proceedings
> pages on 2026-10-27; `[@key]` citations refer to `thesis/references.bib`. Read each paper before citing it, rewrite
> in your own words, and check UPES's rules on declaring AI assistance.

Using a cheap model where it suffices and an expensive one only where it is needed is an old idea in
machine learning (classifier cascades); with large language models (LLMs) it has become practical,
because models of very different size and price now answer the same prompts. Work on the problem falls
into three lines: **cascades**, which try the cheap model first and escalate on doubt; **routers**, which
decide from the query alone before any model runs; and **benchmarks** for comparing both. A fourth,
separate literature — **distribution-free risk control** — supplies the tools we use to turn a routing
threshold into a guarantee. We review each and then place this project among them (Table 2.1).

## 2.1 Cascades: try cheap first, escalate on doubt

**FrugalGPT** [@chen2024frugalgpt] framed LLM cost reduction as a systems problem and proposed three
strategies: prompt adaptation (shorter prompts), LLM approximation (caching or substituting a cheaper
model) and the LLM cascade, in which queries pass through a sequence of models and a learned scoring
function decides whether to accept an answer or call the next model. The authors report matching
GPT-4's accuracy with up to 98% lower cost on their tasks, or improving accuracy by 4% at the same cost.
The cascade's decision is made *after* seeing an answer, at the price of paying for every model tried.

**Mixture-of-thought cascades** [@yue2024cascades] showed that a cheap model's *answer consistency* is a
strong difficulty signal for reasoning tasks: sample several answers from the weaker model (GPT-3.5),
accept when they agree, and escalate to the stronger one (GPT-4) when they do not. Mixing two reasoning
formats, chain-of-thought and program-of-thought, makes disagreement more informative. Across six
reasoning benchmarks the cascade performed comparably to GPT-4 at about 40% of its cost. The idea rests
on **self-consistency** [@wang2023selfconsistency], which samples diverse reasoning paths and takes the
majority answer; agreement among samples correlates with correctness.

**AutoMix** [@aggarwal2024automix] replaces sampling with *self-verification*: the small model is
prompted with a few examples to judge whether its own answer is supported by the context, and a router
formulated as a partially observable Markov decision process (POMDP) uses that noisy judgement to decide
whether to escalate. It reports over 50% lower computational cost at comparable performance.

These works establish that the small model's own output carries information a query-only router lacks.
Our exploratory result on RouterBench (Chapter 5) reproduces that finding on a new pair: agreement
between two small models nearly triples certified savings compared with the best query-only router.

## 2.2 Routers: decide before any model runs

A router picks one model per query from the query text alone, so it pays for exactly one model call.

**Hybrid LLM** [@ding2024hybrid] is the closest precursor to our two-tier design. A router — a small
encoder fine-tuned on query text — predicts whether a small model's answer will be close in quality to a
large model's, and routes to the small model when it is. A threshold on the predicted quality gap can be
moved at test time to trade quality for cost. The authors report up to 40% fewer calls to the large
model with no drop in response quality. They also account for the randomness of generated answers by
scoring several sampled responses per query [check §3 for the exact construction], which is the same
reason our labels average over k samples.

**RouteLLM** [@ong2025routellm] trains routers between a strong and a weak model from *preference data*:
Chatbot Arena comparisons, augmented with "golden-labelled" MMLU validation questions and about 120,000
pairs judged by GPT-4. It compares four router architectures — similarity-weighted ranking, matrix
factorisation, a BERT classifier and a causal LLM classifier — and introduces the evaluation metrics we
adopt: **performance gap recovered** (PGR), the share of the strong–weak quality gap a router recovers;
**APGR**, its average over all frontier-call rates; and **CPT(x%)**, the smallest frontier-call rate that
achieves PGR x%. With GPT-4 as the strong model and Mixtral-8x7B as the weak one — the same development
pair we use — it reports over 2× cost reduction on public benchmarks without substantial quality loss.

Two features of RouteLLM matter for interpreting our results. First, its gains on benchmarks such as
MMLU depend on augmenting the training data with in-domain labels [check the exact wording in §5 of
the paper before citing]. Second,
it reports curves on a test set, not thresholds chosen in advance, so its operating points carry no
guarantee about new queries. Our query-only routers, trained in-domain on frozen splits, land in the
same place: well above random, far below the oracle (Chapter 4).

## 2.3 Benchmarks for routing

**RouterBench** [@hu2024routerbench] removed the main obstacle to studying routing on a budget: it
publishes over 405,000 inference outcomes from eleven LLMs on eight datasets (64 tasks), with each
answer's quality score and cost, so that routers can be compared without running any model. It
evaluates routers in cost–quality space and defines **AIQ** (average improvement in quality) as the
normalised area under a router's non-decreasing convex hull. It also proposes the **zero router**,
which mixes models according to their combined convex hull, as the baseline a learned router must beat.

We use RouterBench as the source of our corpus but, as Chapter 3 documents, not as-is: we found its
GSM8K scores do not measure correctness, one of its small models answers multiple choice at chance,
and its split labels are misleading. Our convex-envelope analysis (Chapter 5) is the same idea as AIQ's
hull, applied to a single router's thresholds.

## 2.4 Distribution-free risk control

Every routing paper must choose a threshold. Chosen on test data, the reported quality is optimistic;
chosen on held-out data without correction, it is still optimistic because the threshold was selected to
look good. A line of work in statistics provides finite-sample guarantees for exactly this situation.

**Risk-controlling prediction sets** (RCPS) [@bates2021rcps] calibrate a model's tuning parameter on
held-out data so that, with probability at least 1 − δ, the true risk is below a chosen level. They
build upper confidence bounds on the risk at each candidate parameter and pick the most aggressive one
that is still certified, assuming the risk is monotone in the parameter.

**Learn then Test** (LTT) [@angelopoulos2021ltt] removes the monotonicity requirement by recasting
calibration as multiple hypothesis testing: each candidate parameter is a null hypothesis "the risk
here is too high", tested with a valid p-value, and a family-wise error-rate procedure picks the set of
certified parameters. With **fixed-sequence testing** — testing candidates in a pre-set order and
stopping at the first failure — no Bonferroni penalty is paid, which makes it practical for a routing
threshold with thousands of candidates.

The p-values need concentration bounds. Hoeffding's inequality is valid but loose. **Betting-based
confidence sequences** [@waudbysmith2024betting] estimate the mean of a bounded random variable by
constructing a gambler's wealth process that grows when the hypothesised mean is wrong; they are valid
at any sample size and adapt to the observed variance, and RCPS uses one variant (often called the WSR
bound) as its tightest option.

To our knowledge, routing papers have not used these tools. We apply LTT with the WSR bound to the
routing threshold, on a calibration split separate from training and test, and find two things worth
reporting (Chapter 5): the bootstrap bound the original design recommended undercovers on low-headroom
data, while WSR is valid and nearly as tight.

## 2.5 Where this project sits

**Table 2.1 — Comparison with the closest work.**

| | Decides | Small model | Threshold guarantee | Cost model | Runs on a laptop |
|---|---|---|---|---|---|
| FrugalGPT | after answers (cascade) | API models | no | API price | no |
| Mixture-of-thought cascade | after sampled answers | GPT-3.5 | no | API price | no |
| AutoMix | after self-verification | smaller LLMs [check §4] | no | compute ratio | no |
| Hybrid LLM | before (query only) | smaller open LLMs [check §4] | no | share of large-model calls | no |
| RouteLLM | before (query only) | Mixtral-8x7B | no | API price | no |
| **This project** | **before, and after (cascade, D-12)** | **Qwen3.5-4B, local** | **yes: LTT + WSR on a held-out split** | **API price (M1) and hardware time (M2)** | **yes, live demo** |

The project's contributions follow from the gaps in that table: (i) a re-audited, leakage-checked
routing corpus with frozen splits; (ii) a calibrated two-head router with a cost-normalised decision
rule; (iii) a threshold with a finite-sample guarantee, chosen without touching test data; (iv) evidence
that the small model's own agreement is the most useful routing signal; and (v) the whole system
running on a 16 GB laptop, with costs that include the laptop's own running cost.
