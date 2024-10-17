#!/bin/bash

#SBATCH --job-name=example
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --chdir=~/development/GalaxyFlow
#SBATCH --output=/project/ls-gruen/users/patrick.gebhardt/output/gaNdalF_paper/plot_flow_out.txt
#SBATCH --err=/project/ls-gruen/users/patrick.gebhardt/output/gaNdalF_paper/plot_flow_err.txt
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

module load python/3.12-2024.06

srun -n1 python plot_flow.py
