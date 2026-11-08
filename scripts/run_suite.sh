#!/bin/bash
# The full experiment suite for one dataset, as a chain of Slurm jobs (submit from anywhere):
#   scripts/run_suite.sh <dataset_id>        e.g. Avazu_x4, KKBox_x1, ML_1M_7_x1, KDD12_x10
# Expects in config/: FCNv2_<ds>[_EMA|_EMA_d9999|_KD|_KDrank|_KDrank_EMA],
# RecursiveFCN_<ds>[_EMA|_KD|_KD_EMA] and config/tuner/FCNv2_<ds>_seeds.yaml.
#
#   build data -> FCN_sp 5 seeds (teacher) -> teacher (mean logit) -> FCN_sp KD x3, RRT KD, RRT KD+EMA -> route fusion
#              -> FCN_sp EMA 0.9995 / 0.9999, RRT, RRT+EMA
set -eo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
DS=$1
TEACHER=teacher_fcnsp_5seed
RRT_TIME=4-00:00:00     # the recursive model is ~10x slower than FCN_sp per epoch
sub() { sbatch --parsable "$@"; }

# 1) data (skipped when already built)
if [ -f "../data/$DS/test.npz" ]; then DEP=""; echo "data: ../data/$DS already built"
else B=$(sub -J "build_$DS" slurm/build_data.sbatch "FCNv2_$DS"); DEP="--dependency=afterok:$B"; echo "build=$B"; fi

# 2) FCN_sp: 5 seeds (the teacher ensemble; row-aligned train predictions) and the EMA runs
CONFIG_DIR=$(python -c "from fuxictr import autotuner; print(autotuner.enumerate_params('config/tuner/FCNv2_${DS}_seeds.yaml'))" | tail -n 1)
python -c "from fuxictr.autotuner import load_experiment_ids; print('\n'.join(load_experiment_ids('$CONFIG_DIR')))" > "$CONFIG_DIR/expids.txt"
S=$(sub -J "seeds_$DS" --array=1-5%5 $DEP slurm/sweep.sbatch "$CONFIG_DIR" "$CONFIG_DIR/expids.txt"); echo "seeds=$S"
for v in EMA EMA_d9999; do echo "FCNv2_${DS}_$v=$(sub -J "fcn_${DS}_$v" $DEP slurm/train.sbatch "FCNv2_${DS}_$v")"; done

# 3) recursive model without teacher
for v in "" _EMA; do echo "RecursiveFCN_$DS$v=$(sub -J "rrt_$DS$v" --time=$RRT_TIME $DEP slurm/train.sbatch "RecursiveFCN_$DS$v" --keep_model)"; done

# 4) teacher = mean logit of the 5 seeds, then the distillation runs
T=$(sub -J "teacher_$DS" --dependency=afterok:$S slurm/make_teacher.sbatch $TEACHER "FCNv2_${DS}_00*" 5 "$DS"); echo "teacher=$T"
for v in KD KDrank KDrank_EMA; do echo "FCNv2_${DS}_$v=$(sub -J "fcn_${DS}_$v" --dependency=afterok:$T slurm/train.sbatch "FCNv2_${DS}_$v")"; done
echo "RecursiveFCN_${DS}_KD=$(sub -J "rrt_${DS}_KD" --time=$RRT_TIME --dependency=afterok:$T slurm/train.sbatch "RecursiveFCN_${DS}_KD" --keep_model)"
R=$(sub -J "rrt_${DS}_KD_EMA" --time=$RRT_TIME --dependency=afterok:$T slurm/train.sbatch "RecursiveFCN_${DS}_KD_EMA" --keep_model)
echo "RecursiveFCN_${DS}_KD_EMA=$R"

# 5) route aggregation of the RRT + EMA + KD model
echo "routes=$(sub -J "routes_$DS" --dependency=afterok:$R slurm/route_fusion.sbatch "RecursiveFCN_${DS}_KD_EMA" 12)"
