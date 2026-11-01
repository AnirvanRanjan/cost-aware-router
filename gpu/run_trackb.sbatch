#!/usr/bin/env bash
# SLURM version of gpu/run_trackb.sh. Edit the partition / account lines to match the UPES cluster
# (ask the admin, or run `sinfo` to see partition names), then: sbatch gpu/run_trackb.sbatch
#SBATCH --job-name=trackb-qwen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=trackb_%j.log
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

source .venv-gpu/bin/activate
bash gpu/run_trackb.sh "$@"
