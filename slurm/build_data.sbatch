#!/bin/bash
# Build feature_map.json + parquet for the dataset of an expid (CPU only, once per dataset):
#   sbatch slurm/build_data.sbatch FCNv2_Criteo_x4_10
#SBATCH --job-name=build_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --exclude=hal            # degraded node (acpi_pad throttling, 2026-09-27)

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date)"
python run_expid.py --expid "$1" --build_data --gpu -1
echo "end=$(date)"
