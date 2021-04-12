#!/usr/bin/env bash

#SBATCH -J LiH-dissoc-curve
#SBATCH -p normal_q
#SBATCH -A jvandyke_alloc
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --array=0-9
#SBATCH -t 120:00:00

source activate chem

python BeH2.py 1.0 1.9 $SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_TASK_ID

wait
exit 0
