# Runbook: Qwen runs on the M2 (lane D, Track B)

Everything below runs on the 16 GB M2, from the repo root, in order. Each long step can be stopped
(Ctrl-C, lid closed, crash) and restarted with **the same command**: finished work is cached and is
never redone. Paste the last lines of each step's output into the team chat.

## 0. Set up (once, ~15 min)

```bash
git clone https://github.com/AnirvanRanjan/cost-aware-router.git   # or: git pull, if you have it
cd cost-aware-router
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ml.txt            # includes mlx-lm on Apple Silicon

gh release download data-910a584c8279baa7 -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench \
  -p corpus.parquet -p scores.parquet -p dedup_pairs.parquet -p manifest.json -p SHA256SUMS
python scripts/verify_data.py                 # must print "Same data as everyone else."
pytest -q                                     # should pass
```

## 1. Gemini key (T40, 2 min)

```bash
cp .env.example .env            # then put your key after GEMINI_API_KEY=  (never commit .env)
python scripts/09_generate.py --list-models gemini-3-flash | grep -i flash
python scripts/09_generate.py --client gemini-3-flash --sources gsm8k --limit 3
```
If the first command doesn't list `gemini-3-flash-preview`, change `model_id` in
`configs/clients.yaml` to the flash model it does list.

## 2. Smoke test Qwen (5 min; downloads ~3 GB the first time)

```bash
python scripts/09_generate.py --client qwen3.5-4b --sources gsm8k --splits test --limit 5
python scripts/14_score_mcq.py --sources mmlu --limit 50
```
Check: the GSM8K answers end with "The answer is N."; the MCQ step prints accuracy well above 0.25
and no single letter above ~40%. If either looks wrong, stop and post the output.

## 3. Measure batch speed (T31, ~10 min)

```bash
python scripts/bench_batch.py --client qwen3.5-4b --batches 1 2 4 8 --n 16
```
Put the best batch size in `configs/clients.yaml` (`max_batch`) and commit `notes/batch_speedup.json`.

## 4. Multiple choice: MMLU, ARC, Winogrande (~1–2 h)

```bash
caffeinate -i python scripts/14_score_mcq.py
```
16,130 questions, one forward pass each. Writes `artifacts/trackb/qwen_mcq_scores.parquet` and
`qwen_mcq_options.parquet`.

## 5. GSM8K, 3 samples per question (overnight, in this order)

```bash
caffeinate -i python scripts/09_generate.py --client qwen3.5-4b --sources gsm8k --splits test cal \
  --k 3 --temperature 0.7 --out artifacts/trackb/qwen_gsm8k_testcal.parquet
caffeinate -i python scripts/09_generate.py --client qwen3.5-4b --sources gsm8k --splits train \
  --k 3 --temperature 0.7 --out artifacts/trackb/qwen_gsm8k_train.parquet
```
Test + cal is 2,980 questions × 3 answers; train is 4,470 × 3. Step 3's speed tells us how long each
takes (roughly 1.5 M and 2.3 M output tokens). Test + cal comes first because the guarantee needs it;
train is for retraining the router on Qwen and can be cut short if time runs out.

## 6. Hand over

```bash
python - <<'EOF'
import pandas as pd
parts = [pd.read_parquet(p) for p in ("artifacts/trackb/qwen_gsm8k_testcal.parquet",
                                      "artifacts/trackb/qwen_gsm8k_train.parquet")]
pd.concat(parts).to_parquet("artifacts/trackb/qwen_responses.parquet", index=False)
EOF
python scripts/10_build_trackb.py --responses artifacts/trackb/qwen_responses.parquet \
                                  --mcq artifacts/trackb/qwen_mcq_scores.parquet
```
Post the coverage and kill-check lines it prints, then upload `artifacts/trackb/` (or tell Ani and
he publishes it as a data release). Partial results are fine: step 6 works on whatever is finished.
