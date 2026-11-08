#!/bin/bash
# Train + test one expid on one GPU. Submit from the repo root:
#   sbatch -J fcnv2 slurm/train.sbatch FCNv2_Criteo_x4_10 [extra run_expid.py args]
#SBATCH --job-name=fcn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=hal            # degraded node (acpi_pad throttling, 2026-09-27)
#SBATCH --requeue

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
EXPID=$1; shift
echo "host=$(hostname) job=$SLURM_JOB_ID expid=$EXPID start=$(date)"
python run_expid.py --expid "$EXPID" --gpu 0 "$@"
echo "end=$(date)"
