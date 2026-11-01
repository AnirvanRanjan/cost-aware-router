# Track B on the UPES GPU (H100)

One job produces every Qwen3.5-4B answer Track B needs. On an H100 it should take minutes rather than
the days it takes on a laptop (vLLM batches thousands of questions at once).

## What to send / what the cluster needs

Nothing to copy by hand: the code is on GitHub and the data is a public release.
The cluster needs: one NVIDIA GPU with ≥ 16 GB (an H100 is plenty), Linux, Python 3.10–3.12, CUDA
drivers, and internet access to github.com and huggingface.co (if compute nodes have no internet, see
"Offline nodes" below).

## Steps (on the cluster)

```bash
git clone https://github.com/AnirvanRanjan/cost-aware-router.git
cd cost-aware-router
python3 -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install -r gpu/requirements-gpu.txt

bash gpu/run_trackb.sh --limit 20     # smoke test (~5 min incl. model download): check the printed accuracy
bash gpu/run_trackb.sh                # full run
```
With SLURM instead: edit the partition / account lines in `gpu/run_trackb.sbatch`, then
`sbatch gpu/run_trackb.sbatch --limit 20`, check `trackb_*.log`, then `sbatch gpu/run_trackb.sbatch`.

Bring back `trackb_results.tar.gz` and unpack it in the project folder on the Mac:
```bash
tar xzf trackb_results.tar.gz
python scripts/10_build_trackb.py --responses artifacts/trackb/qwen_responses.parquet \
                                  --mcq artifacts/trackb/qwen_mcq_scores.parquet
```

## Offline nodes

Download the model on a node that has internet (often the login node), then point the job at it:
```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B
QWEN_MODEL=models/Qwen3.5-4B bash gpu/run_trackb.sh
```

## What it runs

`scripts/gpu_trackb_vllm.py`: Qwen3.5-4B in bf16, thinking mode off, the same GSM8K prompt as the Mac
(`router.generate.build_prompt`), 3 samples per question (temperature 0.7, top-p 0.8, top-k 20, up to
1,024 tokens), and one-token option scoring for multiple choice. Settings and timings are written to
`artifacts/trackb/gpu_manifest.json`.

The Mac demo runs the 4-bit MLX build of the same model. Report the difference: run ~200 test
questions on the Mac too and compare accuracy.
