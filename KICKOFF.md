# Kickoff — Day 0 through the Week-2 gate

This is the "who does what, starting today" companion to `DESIGN.md`. The design says *what* to
build; this says *how four people start building it without blocking each other*.

---

## What already exists

The Phase-0 foundation is in the repo and green (`pytest`: 40 passed):

| File | What it gives you | Owner from now on |
|---|---|---|
| `router/schema.py` | The five parquet contracts, with `validate`, `read`, `write` and `content_hash`. A malformed file fails at the boundary, not three stages later. | everyone (changes need every reader's review) |
| `router/costs.py` + `configs/models.yaml` | The price table and cost model: M1 API spend, M2 fully loaded, batch/cache discounts, run estimator. | C |
| `router/splits.py` | Grouped train/cal/test split + a leakage check. | A |
| `router/synthetic.py` | Synthetic corpus + scores with a **planted** routing signal. Your test oracle for weeks 1–8. | C (generator), everyone (consumer) |
| `scripts/00_make_synthetic.py` | Writes `artifacts/synthetic/{corpus,scores,truth}.parquet` + `manifest.json`. | — |
| `scripts/estimate_spend.py` | The Phase-0 gate: projected dollars and hours **before** spending anything. | C |
| `.github/workflows/tests.yml` | CI: tests + both scripts on every push and PR. | D |

**One contract amendment vs `DESIGN.md` §4.1, to confirm on Day 0:** `scores.parquet` carries
`mean_n_in` and `mean_n_out`. RouterBench ships per-item costs but no per-sample responses, so the
cost lane must not depend on `responses.parquet`.

---

## Day 0 — the first meeting (90 minutes, all four, laptops open)

1. **Assign roles** (15 min). Staff C first: strongest at statistics and most willing to write
   tests. Then A: the most careful person. B and D by preference.

   | Role | Lane | One-line job |
   |---|---|---|
   | **A** | Data & Corpus | Get real data in, deduplicated and split with no leakage |
   | **B** | Features & Router | Turn prompts into features and features into `ŝ_S`, `ŝ_L` |
   | **C** | Evaluation & Statistics | Turn predictions into a Pareto curve and a *defensible* τ\* |
   | **D** | Systems & Demo | Make models callable, runs reproducible, and the demo live |

2. **Agree three things out loud** (10 min) and write them in the group chat:
   - The headline target is **PGR ≥ 0.95**, not "95% of frontier accuracy" (`DESIGN.md` §0).
   - **Nobody regenerates responses without bumping the manifest version.**
   - **No paid API calls before the week-8 gate.** Free tiers only.
3. **D creates the GitHub repo** (15 min), private, from this folder. Everyone else is invited.
   Turn on branch protection for `main`: PRs only, one review, CI must pass.
4. **Everyone gets green on their own Mac** (30 min) — see the README's setup block. Done means
   `pytest` prints `40 passed` and `python scripts/00_make_synthetic.py` prints the headroom table.
5. **Book the rhythm** (10 min):
   - Async standup Mon / Wed / Fri, three lines in the group chat: *done · next · blocked*.
   - **Friday integration**, 45 minutes. A rotating integrator merges open PRs and runs
     everything end to end. Rotation: **W1 D → W2 A → W3 B → W4 C**, then repeat.
6. **Create the shared artifacts folder** (10 min): one Google Drive folder, `artifacts/`, mirrors
   the repo's gitignored `artifacts/`. Files are named by the content hash in their manifest.

---

## Week 1

### A · Data & Corpus

- [ ] **RouterBench audit.** Load `withmartian/routerbench` from Hugging Face and write
      `notes/routerbench_audit.md` answering five questions. A notebook is fine here; this is
      exploration.
      1. Which datasets, and how many items each?
      2. Is the per-item score binary or continuous, per dataset?
      3. Are token counts present per item and model, or only cost? (If only cost, back out tokens
         from the 2024 prices and document how.)
      4. Can GSM8K and MMLU items be traced to their original dataset IDs? Track B needs this to align
         local-model runs with RouterBench's frontier scores.
      5. Are there duplicate prompts within or across datasets?
- [ ] `router/data.py::load_routerbench()` maps a RouterBench frame onto `CORPUS` + `SCORES`.
      No splits yet.
- [ ] `configs/datasets.yaml` records the model-name mapping (which RouterBench model is `S` and
      which is `L`) and per-dataset sample counts.
- **Done when:** `tests/test_data.py` validates a 200-row RouterBench sample against both contracts,
  and the audit note answers all five questions.

### B · Features & Router

- [ ] `router/features.py::heuristics(prompts) -> (np.ndarray, names)`: the ~25 F0 features from
      `DESIGN.md` §1.3. Pure function, no state.
- [ ] F1: TF-IDF (word 1–2 grams + char 3–5 grams) **fit on the train split only**. Takes a fitted
      vectorizer, returns vectors.
- [ ] Write `features.parquet` for synthetic F0 and F1 through `schema.write`.
- **Done when:** hand-written prompts give hand-computed F0 values in tests, and F0 token count on
  synthetic GSM8K has Spearman ρ > 0.3 with `truth.difficulty`. If it doesn't, your extractor is
  wrong, not the data. The signal is planted.

### C · Evaluation & Statistics

- [ ] `router/eval.py`: `Q(τ)`, `C(τ)`, `PGR`, `APGR`, and the random and oracle reference curves.
- [ ] `router/threshold.py` steps 1 and 2 from `DESIGN.md` §3.4: the exact O(n log n) sweep and
      the concave envelope.
- [ ] Hand-build a 6-row example where you work out τ\* on paper. It becomes a unit test.
- [ ] Property test: the exact sweep agrees with brute force on 200 random small cases.
- **Done when:** on synthetic, the oracle curve dominates random, and the property test passes.
  Note that **oracle PGR can exceed 1.0.** On items where the small model beats the large one,
  routing small is *better* than all-frontier. That is correct, not a bug.

### D · Systems & Demo

- [ ] Repo, CI green on GitHub, branch protection, a PR template with a "contracts touched?"
      checkbox.
- [ ] Install MLX and benchmark **Qwen3.5-4B** and **Llama 3.2 3B** at batch 1 and batch 8. Put the
      measured tok/s into `configs/models.yaml`, replacing the estimates. This single commit corrects
      every wall-clock figure in the design doc.
- [ ] `router/generate.py` skeleton: one `Client` protocol with
      `complete(prompts, n, temperature, max_tokens)`, and two implementations. `MLXClient` for
      local; `OpenAICompatClient(base_url, api_key)` for Gemini, Groq and GitHub Models, which all
      speak the OpenAI API.
- [ ] A content-hash disk cache keyed on `(model, prompt, params, sample_idx)`.
- **Done when:** calling the same request twice hits the cache (tested), and measured tok/s is
  committed.

---

## Week 2

| Role | Tasks | Done when |
|---|---|---|
| **A** | `router/dedup.py`: MinHash LSH on 5-gram shingles (Jaccard ≥ 0.8) plus embedding cosine (≥ 0.95), then connected components to `group_id`. Extend `splits.py` with leave-one-subject-out. | On synthetic, recall of the 100 planted near-duplicate pairs is ≥ 0.9, and the dedup report is written as an artifact. |
| **B** | F2 embeddings (`bge-base-en-v1.5`; `all-MiniLM-L6-v2` if CPU time hurts) cached to `features.parquet`. First baseline: logistic regression on F0 and on F1, predicting the realized gap. | Validation Spearman ρ(Δ̂, Δ) reported for F0 and F1 on synthetic. F1 ≥ F0. |
| **C** | Lower confidence bounds (Hoeffding + paired bootstrap), τ\* selection, `plot_pareto.py` panels A and B. Also test with a **low-headroom** synthetic config (raise `a_small`) so the bound's behavior is checked where it matters. | The first Pareto figure exists, on fake data, with random and oracle curves and a marked τ\*. |
| **D** | `generate.py` end to end on 20 real GSM8K items: local MLX **and** one free-tier API, both through the cache, both writing `responses.parquet` + `manifest.json`. | Re-running the script makes zero model calls. |

---

## The Week-2 gate (Friday, whole team)

- [ ] All five contracts agreed and frozen in `schema.py`. `CONTRACT_VERSION` stays `1.0`.
- [ ] `costs.py` passing, with the measured tok/s in `models.yaml`.
- [ ] Synthetic `scores.parquet` on the shared drive, named by its content hash.
- [ ] Each lane has one PR merged into `main` with tests.
- [ ] `python scripts/estimate_spend.py` re-run with measured numbers and the output pasted in the
      group chat. That is the Phase-0 gate from `DESIGN.md` §4.2.

After this Friday, **nobody is blocked on anybody** until real data lands in weeks 6–8.

---

## How you work together

- **Branches:** `a/dedup`, `b/f0-heuristics`, `c/exact-sweep`, `d/mlx-client`. The prefix is the
  lane.
- **Reviews:** every PR is reviewed by someone **from a different lane**. That spreads knowledge of
  the whole system, and by the defence all four of you can explain every stage. A PR that touches
  `schema.py` needs a review from every lane that reads the changed contract.
- **Definition of done for any module:** it writes its output through `schema.write`, it has tests,
  and it gets one line in the README.
- **Artifacts** never go in git, never go over email, and are never renamed by hand. They go in
  the shared folder, named by content hash, next to their `manifest.json`.

## Not in weeks 1–2

- **No paid API calls.** Free tiers only, and only D, only for the 20-item smoke test.
- **No tuning on synthetic data.** It exists to test plumbing. Hyperparameters picked on it are
  meaningless.
- **No notebook as the source of a number.** Notebooks explore; tested modules produce results.
- **No slide shows a synthetic number** unless it is labelled SYNTHETIC in the title.
