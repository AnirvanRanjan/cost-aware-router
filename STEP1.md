# Step 1 — Ani on the M1, then hand off to the M2

Step 1 is the light work: set up the repo and find out what's actually inside RouterBench.
None of it runs a language model, so the M1 is fine. The M2 takes over for anything that
does, because local-model speed has to be measured on the machine that will run the models.

---

## Part 1 · Ani (M1) · about 2–3 hours

### 1. Get green (15 min)

```bash
cd ~/Downloads/"Major Project"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # expect: 40 passed
python scripts/00_make_synthetic.py     # prints a headroom table
python scripts/estimate_spend.py        # prints dollars and hours
```

If `pytest` doesn't say `40 passed`, stop and fix that first. Everything else assumes it.

### 2. Git and GitHub (15 min)

```bash
git init -b main
git add .
git status                              # check: no .venv/, no artifacts/, no .env
git commit -m "Phase 0 foundation: contracts, cost model, synthetic data, tests"
```

Then on github.com: create a **private** repo (empty, no README), and push:

```bash
git remote add origin git@github.com:<you>/cost-aware-router.git
git push -u origin main
```

Invite the other three as collaborators. Check the **Actions** tab — the test workflow
should go green on its own.

### 3. RouterBench audit (30 min) — Person A's Week-1 task

The audit is scripted, on the branch `a/routerbench-audit` (already committed):

```bash
git switch a/routerbench-audit
pytest                                  # expect: 44 passed on this branch
```

Download the 0-shot file only (about 100 MB). Skip `routerbench_raw.pkl`, which is 1.2 GB.

```bash
hf download withmartian/routerbench routerbench_0shot.pkl \
  --repo-type dataset --local-dir artifacts/raw/routerbench
# older CLI: huggingface-cli download (same arguments)
```

It lands in `artifacts/`, which git ignores. That's on purpose.

> **Pickle files run code when loaded.** Only load `.pkl` files from the official
> `withmartian/routerbench` repo, never from a link someone sends you.

Then:

```bash
python scripts/audit_routerbench.py     # writes notes/routerbench_audit.md
```

The script answers the five questions straight from the file, and adds a headroom preview for
Mistral-7B vs GPT-4 (the Phase-3 kill-check, early):

1. Which datasets, and how many items each?
2. Is the per-item score 0/1 or continuous, per dataset?
3. Are token counts stored, or only cost?
4. Can GSM8K and MMLU items be matched to their original dataset IDs?
5. Are there duplicate prompts, within a dataset or across datasets?

It doesn't assume RouterBench's layout. If the file isn't shaped the way the paper describes,
it prints a **WARNING** and lists every column instead of guessing. Read the output, fill in
each **Interpretation** line by hand, then:

```bash
git add notes/routerbench_audit.md
git commit -m "RouterBench audit: datasets, score types, costs, IDs, duplicates"
git push -u origin a/routerbench-audit
```

Open a pull request. CI must pass.

**Part 1 is done when:** the repo is on GitHub with green CI, all three teammates are
invited, and the audit PR answers all five questions.

---

## Part 2 · The M2 teammate · about 1–2 hours

### 1. Get green (15 min)

```bash
git clone git@github.com:<ani>/cost-aware-router.git
cd cost-aware-router
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                  # expect: 40 passed
```

### 2. Measure local-model speed (45 min)

The design's hour estimates are guesses for an M2. This replaces them with facts.

```bash
pip install mlx-lm
```

On huggingface.co, search the `mlx-community` organisation for the 4-bit instruct builds of
**Qwen3.5-4B** and **Llama 3.2 3B**. Write down the exact repo names. Then, for each:

```bash
mlx_lm.generate --model mlx-community/<exact-repo-name> \
  --prompt "Solve: 17*23. Think step by step." --max-tokens 256
```

Run it three times and keep the median **Generation: … tokens-per-sec** figure. Run it plugged
in, with nothing else open. (The first run downloads the model, so don't count it.)

### 3. Commit the numbers (15 min)

```bash
git switch -c d/measured-tok-s
```

In `configs/models.yaml`, replace `tokens_per_sec` for `qwen3.5-4b` and `llama-3.2-3b` with
your medians, and change the comment above them to say **measured** with the date and the
exact model repo names. Then:

```bash
pytest                                  # still 40 passed
python scripts/estimate_spend.py        # paste this output in the group chat
git commit -am "Measured tok/s on M2 for Qwen3.5-4B and Llama 3.2 3B"
git push -u origin d/measured-tok-s
```

Open a PR. Ani reviews it.

**Part 2 is done when:** measured tok/s is merged into `main`, and the new
`estimate_spend.py` output is in the group chat. That's the Phase-0 gate from `DESIGN.md`.
