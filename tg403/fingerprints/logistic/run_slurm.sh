#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
##
#SBATCH --job-name=dt
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##

hostname
date

module add CUDA/11.2.2
module add ANACONDA/2020.11

python /home1/ok69531/LC50/ordinal/logistic/mgl_logistic.py