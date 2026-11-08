#!/bin/bash
# Expand a FuxiCTR tuner config and submit one Slurm array task per expid.
#   scripts/sweep.sh config/tuner/FCNv2_Criteo_x4_10.yaml [max_concurrent=8] [extra sbatch args...]
set -eo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
TUNER=$1; MAX=${2:-8}
shift $(( $# >= 2 ? 2 : 1 ))
CONFIG_DIR=$(python -c "from fuxictr import autotuner; print(autotuner.enumerate_params('$TUNER'))" | tail -n 1)
EXPID_LIST=$CONFIG_DIR/expids.txt
python -c "from fuxictr.autotuner import load_experiment_ids; print('\n'.join(load_experiment_ids('$CONFIG_DIR')))" > "$EXPID_LIST"
N=$(wc -l < "$EXPID_LIST")
echo "$N expids in $CONFIG_DIR"
sbatch --job-name="$(basename "$CONFIG_DIR")" --array=1-"$N"%"$MAX" "$@" slurm/sweep.sbatch "$CONFIG_DIR" "$EXPID_LIST"
