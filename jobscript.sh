#!/bin/bash

#SBATCH --job-name=example
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --chdir=directory path
#SBATCH --output=filepath
#SBATCH --err=filepath
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

srun -n1 python script.py -cf LMU.cfg
