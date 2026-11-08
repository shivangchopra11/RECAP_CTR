#!/bin/bash
# One array task = one expid of a tuner sweep. Submitted by scripts/sweep.sh and scripts/run_suite.sh.
#SBATCH --job-name=sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
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
CONFIG_DIR=$1; EXPID_LIST=$2
EXPID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPID_LIST")
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} expid=$EXPID start=$(date)"
python run_expid.py --config "$CONFIG_DIR" --expid "$EXPID" --gpu 0
echo "end=$(date)"
