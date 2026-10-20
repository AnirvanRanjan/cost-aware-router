# Cost-Aware Model Router

A lightweight classifier that predicts query difficulty and routes each query to either a small
local model or a frontier API model. The target is **2–5× lower cost at ≥ 95% of the frontier
model's quality (PGR)**, with a decision threshold backed by a confidence bound, not a point
estimate.

- **`DESIGN.md`**: the full technical design (the dark-mode PDF next to it is the same document).
- **`KICKOFF.md`**: who does what, from Day 0 to the Week-2 gate.

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # weeks 1-2 core, about a minute
pytest                                   # expect: all tests pass
python scripts/00_make_synthetic.py      # -> artifacts/synthetic/ + a headroom table
python scripts/estimate_spend.py         # the Phase-0 gate: dollars and hours before any spend
```

Install `requirements-ml.txt` when your lane needs torch, sentence-transformers, MLX or FastAPI.
Copy `.env.example` to `.env` for API keys. `.env` is gitignored.

## Getting the data

Generated data isn't in git. Each version is a GitHub release named after its fingerprint:

```bash
gh release download data-3f774421cc116201 -R AnirvanRanjan/cost-aware-router -D artifacts/routerbench
python scripts/verify_data.py            # must print "Same data as everyone else."
```

To make a new version instead: `scripts/01_acquire_routerbench.py`, then `scripts/02_dedup.py`
(needs `routerbench_0shot.pkl` and the GSM8K answers file; see `configs/datasets.yaml`).

## Layout

```
configs/      models.yaml (prices, local tok/s)   experiment.yaml (splits, target, tau settings)
router/       schema.py  costs.py  splits.py  synthetic.py      <- week 0, done
              data.py  dedup.py                                  <- A
              features.py  models.py  calibrate.py               <- B
              eval.py  threshold.py                              <- C
              generate.py  serve.py                              <- D
scripts/      00_make_synthetic.py  estimate_spend.py
tests/        one test file per module
artifacts/    gitignored. Parquets and manifests live in the shared drive, named by content hash.
```

## Four rules

1. **A rotating integrator merges and runs the full pipeline every Friday.**
2. **Artifacts are not git.** Shared drive, content-hash names, never emailed.
3. **Nobody regenerates responses without bumping the manifest version.**
4. **Every module owner writes their own tests**, and the τ solver's tests matter most.

## Licence

Code: [MIT](LICENSE). The data releases are derived from RouterBench (withmartian/routerbench)
and GSM8K (openai/gsm8k) and stay under those datasets' own terms.
