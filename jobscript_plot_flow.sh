#!/bin/bash
#SBATCH --job-name=plot_flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --chdir=~/development/GalaxyFlow
#SBATCH --output=/project/ls-gruen/users/patrick.gebhardt/output/gaNdalF_paper/plot_flow_out.log
#SBATCH --err=/project/ls-gruen/users/patrick.gebhardt/output/gaNdalF_paper/plot_flow_err.log
#SBATCH --partition=cip
#SBATCH --gres=gpu:a40:0
#SBATCH --cpus-per-task=8

srun -n1 python plot_flow.py
