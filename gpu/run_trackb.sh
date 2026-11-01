#!/usr/bin/env bash
# Track B on one GPU: Qwen3.5-4B answers for GSM8K (k=3) and MMLU/ARC/Winogrande.
#   bash gpu/run_trackb.sh            # full run
#   bash gpu/run_trackb.sh --limit 20 # smoke test first
# Run from the repo root. Safe to re-run: finished GSM8K chunks are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=data-910a584c8279baa7
URL=https://github.com/AnirvanRanjan/cost-aware-router/releases/download/$TAG
mkdir -p artifacts/routerbench
for f in corpus.parquet scores.parquet manifest.json; do
  [ -f artifacts/routerbench/$f ] || curl -fL -o artifacts/routerbench/$f "$URL/$f"
done
python scripts/verify_data.py

nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/gpu_trackb_vllm.py --model "${QWEN_MODEL:-Qwen/Qwen3.5-4B}" "$@"

tar czf trackb_results.tar.gz artifacts/trackb/qwen_responses.parquet artifacts/trackb/qwen_mcq_scores.parquet \
    artifacts/trackb/qwen_mcq_options.parquet artifacts/trackb/gpu_manifest.json 2>/dev/null || true
echo "Send back: trackb_results.tar.gz"
