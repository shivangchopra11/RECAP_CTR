#!/bin/bash
# Inference-time route aggregation of a trained RecursiveFCN checkpoint (trained with --keep_model):
#   sbatch -J rrt_routes slurm/route_fusion.sbatch <expid> [depth=12]
#SBATCH --job-name=route_fusion
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=hal            # degraded node (acpi_pad throttling, 2026-09-27)
#SBATCH --requeue

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
echo "host=$(hostname) job=$SLURM_JOB_ID expid=$1 start=$(date)"
python scripts/route_fusion.py --expid "$1" --depth "${2:-12}" --gpu 0
echo "end=$(date)"
