#!/bin/bash
# Build a distillation teacher (mean logit of an ensemble's saved predictions), CPU only:
#   sbatch slurm/make_teacher.sbatch <teacher_name> "<expid glob>" [num_models] [dataset_id=Criteo_x4_10]
#SBATCH --job-name=make_teacher
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=kira-lab
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --exclude=hal            # degraded node (acpi_pad throttling, 2026-09-27)

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
python scripts/make_teacher.py --name "$1" --expids "$2" ${3:+--num_models $3} --dataset_id "${4:-Criteo_x4_10}"
python scripts/ensemble.py --expids "$2" --dataset_id "${4:-Criteo_x4_10}"
