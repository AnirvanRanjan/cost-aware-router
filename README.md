# Cost-Aware Model Router

[![Tests](https://github.com/AnirvanRanjan/cost-aware-router/actions/workflows/tests.yml/badge.svg)](https://github.com/AnirvanRanjan/cost-aware-router/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Apple%20Silicon-lightgrey.svg)]()
[![Target](https://img.shields.io/badge/target-PGR%20%E2%89%A5%200.95-blue.svg)]()

A lightweight query-difficulty router that sends each query either to a **small local model on Apple
Silicon** (Qwen3.5-4B via MLX, no API cost) or to a **frontier API model** (GPT-4 in the RouterBench
experiments; Gemini 3 Flash in the live demo).

The target is **2–5× lower inference cost while recovering ≥ 95% of the frontier model's quality
advantage**, measured as **PGR (performance gap recovered)**: PGR = (Q − Q_S) / (Q_L − Q_S). The
routing threshold τ\* is chosen with a **finite-sample confidence bound** on a held-out calibration
split, not read off the test set.

> **Status (Oct 2026):** all results below are on the **development pair, Mixtral-8x7B vs GPT-4**,
> using RouterBench's stored answers. The headline pair, **Qwen3.5-4B vs GPT-4** (DECISIONS.md D-08),
> needs Qwen's own answers from the Mac (Track B) and is not run yet.

---

## Key Highlights & Results

- **Certified thresholds:** a three-way split (**train / calibration / test**); τ\* is picked on
  calibration only, by fixed-sequence **Learn-then-Test** with the **Waudby-Smith & Ramdas (WSR)
  betting bound** (δ = 0.05), and applied once to the untouched test split. Hoeffding and paired-
  bootstrap bounds are also computed for comparison; the bootstrap undercovers on low-headroom data,
  so WSR is the default (`router/threshold.py`).
- **Two-head gap prediction:** the router estimates both models' chance of being right and routes on
  the predicted gap:
  $$\hat{\Delta}(q) = \hat{s}_L(q) - \hat{s}_S(q)$$
  When both models are likely to fail ($\hat{\Delta} \approx 0$), the query stays on the free local
  model instead of paying for a frontier answer that will not be better.
- **What the router can see matters most** (Mixtral pair, certified PGR ≥ 0.95):
  - **Query-only routing** (heuristics, TF-IDF, sentence embeddings): **4–9%** cost savings.
  - **Cascade, answer + second opinion** (look at the small model's answer and whether a second
    small model, Yi-34B, agrees, before escalating): **26%** savings. For the Mac this becomes Qwen's
    agreement with itself across samples (D-12, proposed; not measured yet).
- **Leakage-checked benchmark splits:** a re-audited, re-graded RouterBench corpus (34,778 questions in
  7 datasets, 104,334 model scores). Near-duplicates are grouped in two passes — **lexical MinHash +
  LSH**, then **template-centred `bge-base-en-v1.5` embeddings** — so no near-copy crosses
  train / cal / test. Splits are frozen (D-11).
- **Runs on a laptop:** the trained router is saved as one file (`artifacts/bundle/router.pkl`) and
  served with a streaming web UI (FastAPI) on macOS. The demo server is built and tested with stand-in
  models; the run with Qwen and Gemini is pending.

---

## System Architecture

```
User Query
    │
    ▼
┌────────────────────────────────────────────────────────┐
│               Feature Extraction Pipeline              │
│  - F0: 23 heuristics (length, readability, digits…)    │
│  - F1: TF-IDF word + character n-grams                 │
│  - F2: BAAI/bge-base-en-v1.5 sentence embeddings       │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│        Two-Head Classifier (isotonic-calibrated)       │
│        ŝ_S(q) (local)          ŝ_L(q) (frontier)       │
│                     \            /                     │
│                      Δ̂ = ŝ_L − ŝ_S                     │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│               Threshold Gate (τ*)                      │
│   chosen on cal: Learn-then-Test + WSR betting bound   │
└────────────────────────────────────────────────────────┘
         │                                    │
    Δ̂ ≤ τ* (keep local)                 Δ̂ > τ* (escalate)
         │                                    │
         ▼                                    ▼
┌──────────────────┐                 ┌──────────────────┐
│   Local Model    │                 │  Frontier Model  │
│ Qwen3.5-4B / MLX │                 │ GPT-4 / Gemini   │
│  $0 API cost (M1)│                 │    Cloud API     │
└──────────────────┘                 └──────────────────┘
```

Rule v2 (cost-normalised) thresholds Δ̂ divided by the predicted frontier cost instead of Δ̂.

---

## Repository Layout

```
├── configs/
│   ├── clients.yaml          # Model clients: local MLX models and API endpoints
│   ├── datasets.yaml         # Dataset loader paths and model pairs
│   ├── demo_set.yaml         # Planned demo questions with the expected routing decision
│   ├── experiment.yaml       # Seed, split fractions, target PGR (0.95), confidence δ
│   └── models.yaml           # 2026 prices, measured local tok/s, cost-model M2 assumptions
├── router/
│   ├── bundle.py             # One-file router bundle: features + heads + calibrators + τ*
│   ├── calibrate.py          # Isotonic calibration per head, reliability diagrams, ECE
│   ├── costs.py              # Cost models M1 (API spend) and M2 (laptop running cost)
│   ├── data.py               # RouterBench loading, GSM8K re-grading, gold answer letters
│   ├── dedup.py              # Lexical MinHash + template-centred embedding dedup
│   ├── eval.py               # Routing curves and metrics (PGR, APGR, CPT, AUC)
│   ├── features.py           # F0 heuristics, F1 TF-IDF, F2 sentence embeddings
│   ├── generate.py           # Model clients (MLX, OpenAI-compatible APIs), cached and resumable
│   ├── graders.py            # GSM8K numeric exact match, multiple-choice grading
│   ├── mcq_mlx.py            # Multiple-choice scoring by option probability (local, MLX)
│   ├── models.py             # Baselines and the two-head MLP router
│   ├── routerbench_audit.py  # Audit of the raw RouterBench file
│   ├── schema.py             # Data contracts between pipeline stages
│   ├── serve.py              # FastAPI demo server with SSE streaming
│   ├── splits.py             # Grouped train/cal/test splits (no group crosses splits)
│   ├── synthetic.py          # Synthetic data for quick smoke tests
│   ├── threshold.py          # Exact sweep, envelope, WSR / bootstrap / Hoeffding, Learn-then-Test
│   ├── trackb.py             # Aligns Qwen's answers with GPT-4's scores (Track B)
│   └── static/               # Demo web page
├── scripts/
│   ├── 00_make_synthetic.py      # Synthetic data for a quick end-to-end check
│   ├── 01_acquire_routerbench.py # Builds the corpus from the raw RouterBench file
│   ├── 02_dedup.py               # Two-pass near-duplicate grouping and re-split
│   ├── 03_extract_embeddings.py  # F2 embeddings for the corpus
│   ├── 04_train_two_head.py      # Two-head router, 5 seeds
│   ├── 05_calibrate.py           # Isotonic calibration on cal
│   ├── 06_run_ablation.py        # Feature ladder ablation (F0, F1, F2)
│   ├── 07_baseline_predictions.py # F0 / F1 baseline routers
│   ├── 08_threshold_report.py    # τ* on cal, one-shot test report, certified savings
│   ├── 09_generate.py            # Generate answers with any client (Track B)
│   ├── 10_build_trackb.py        # Qwen vs GPT-4 scores on the same questions
│   ├── 11_build_bundle.py        # Exports the router bundle for the demo
│   ├── 12_cascade_explore.py     # Cascade / second-opinion routing experiment
│   ├── plot_pareto.py            # Pareto figure, panels A–D
│   ├── bench_batch.py            # MLX batched-decoding speed-up (Mac only)
│   ├── check_demo_set.py         # Checks every planned demo question's routing decision
│   ├── estimate_spend.py         # Projected API spend and local wall-clock before any run
│   ├── serve_demo.py             # Launches the demo web server
│   └── verify_data.py            # Checks local data against the release fingerprints
├── notes/                    # Reports, audit findings, figures
├── thesis/                   # Report chapters (Markdown drafts) and references.bib
├── tests/                    # Unit and integration tests
├── DECISIONS.md              # Team decisions (D-01 …)
├── DESIGN.md                 # Technical design
└── TASKS.md                  # Tasks by lane, and progress
```

---

## Setup & Installation

### Requirements
- macOS with Apple Silicon for local models and the demo (16 GB recommended; the 4-bit Qwen3.5-4B
  needs about 3 GB). Everything else, including the tests, also runs on Linux.
- Python 3.10 or newer (CI uses 3.11)
- Git and the GitHub CLI (`gh`)

### 1. Environment

```bash
git clone https://github.com/AnirvanRanjan/cost-aware-router.git
cd cost-aware-router

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt       # core: data pipeline, statistics, tests
pip install -r requirements-ml.txt    # torch, sentence-transformers, FastAPI, mlx-lm (Mac)
```

### 2. API keys

Keys live in `.env`, which is git-ignored. Never commit them.

```bash
cp .env.example .env
```

```ini
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GITHUB_TOKEN=your_github_token_here
```

### 3. Tests

```bash
pytest
```
*About 375 tests; those needing torch or FastAPI are skipped if those aren't installed.*

---

## Getting the Data

Data is shared as GitHub release assets named by content fingerprint, not stored in git:

```bash
# Frozen corpus and scores (release data-910a584c8279baa7)
gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench \
    -p corpus.parquet -p scores.parquet -p dedup_pairs.parquet -p manifest.json -p SHA256SUMS

# Precomputed F2 embeddings (optional, ~157 MB)
gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router -D artifacts/features \
    -p features.parquet

python scripts/verify_data.py        # prints "Same data as everyone else."
```

---

## Reproducing the Pipeline

### 1. Quick check on synthetic data
```bash
python scripts/00_make_synthetic.py
python scripts/estimate_spend.py
```

### 2. Router training and evaluation (Mixtral pair)
```bash
python scripts/03_extract_embeddings.py   # only if you did not download features.parquet
python scripts/04_train_two_head.py --features artifacts/features/features.parquet
python scripts/05_calibrate.py --features artifacts/features/features.parquet
python scripts/06_run_ablation.py --features artifacts/features/features.parquet
python scripts/07_baseline_predictions.py
python scripts/08_threshold_report.py     # the official, certified savings numbers
python scripts/12_cascade_explore.py      # needs the raw RouterBench file
python scripts/11_build_bundle.py         # writes artifacts/bundle/router.pkl
```

---

## Live Demo

```bash
python scripts/serve_demo.py              # Qwen locally via MLX + Gemini API
python scripts/serve_demo.py --offline    # no network: cached frontier answers only
python scripts/serve_demo.py --fake       # no models: check the UI anywhere
```

Then open **`http://127.0.0.1:8000`**.

- **Routing decision per query:** ŝ_S, ŝ_L and Δ̂ against τ\*, and where the query went.
- **Streaming answers** over server-sent events.
- **Cost and latency:** first-token and total time, this query's estimated cost, and money saved so
  far versus sending everything to the frontier model (priced at list rates even on a free tier).

Before a demo, run `python scripts/check_demo_set.py`. The current interim router sends most short,
everyday questions to the frontier (`notes/demo_set_check.md`); the Qwen-trained / cascade router is
meant to fix that.

---

## Key Experimental Results (interim)

Headline pool (**MMLU, GSM8K, ARC-Challenge, Winogrande**), S = Mixtral-8x7B, L = GPT-4, τ\* chosen
on cal with the WSR bound, target PGR ≥ 0.95, δ = 0.05, test split used once
(`notes/threshold_report.md`, `notes/cascade_report.md`):

| Router | What it sees | Rule | PGR on test [95% CI] | Certified savings [95% CI] |
|:---|:---|:---:|:---:|:---:|
| `f0_logreg` | query: heuristics | v1 | 96.6% [95.3, 97.8] | **3.9%** [3.4, 4.4] |
| `f1_logreg` | query: TF-IDF | v1 | 96.9% [95.5, 98.1] | **2.7%** [2.3, 3.0] |
| `two_head_mlp` † | query: F2 embeddings + F0 | v2 | 96.3% [95.2, 97.5] | **9.4%** [7.8, 10.9] |
| `cascade_answer` | + Mixtral's answer | v1 | 97.2% [96.1, 98.3] | **8.4%** [7.2, 9.7] |
| `cascade_answer+2nd` | + agreement with Yi-34B | v1 | 97.9% [96.8, 98.9] | **26.0%** [24.2, 27.9] |

† τ\* grid taken from cal, because those predictions had no train rows; re-run pending.

> **Takeaway:** routing from the question text alone saves little on this pair; a perfect router would
> send only 22% of questions to GPT-4, the best query-only router about 89%. Letting the router see
> the small model's answer and a second opinion triples the certified saving. The 2–5× target is not
> reached yet; the Qwen pair and Qwen's self-agreement are the next test.

---

## Security & Secrets

- API keys live only in `.env`, which `.gitignore` excludes.
- The repository was checked by hand for keys and personal data before it was made public (D-09).
- Model clients (`router/generate.py`) read keys only from named environment variables
  (`GEMINI_API_KEY`, `GITHUB_TOKEN`, …) and stop with a clear error if one is missing.

---

## Documentation Index

- [DESIGN.md](DESIGN.md) — full technical design.
- [DECISIONS.md](DECISIONS.md) — team decisions, D-01 onwards.
- [TASKS.md](TASKS.md) — tasks by lane and progress.
- [KICKOFF.md](KICKOFF.md) — initial plan and lane contracts.
- [thesis/](thesis/) — report chapter drafts.
- [notes/threshold_report.md](notes/threshold_report.md) — certified thresholds and savings (official numbers).
- [notes/cascade_explore.md](notes/cascade_explore.md) — the cascade experiment, explained.
- [notes/dedup_report.md](notes/dedup_report.md) — near-duplicate grouping, with examples.

---

## Licence & Attribution

- **Code:** [MIT License](LICENSE).
- **Data releases:** derived from [RouterBench](https://github.com/withmartian/routerbench) and
  [GSM8K](https://github.com/openai/grade-school-math), under their own terms.
