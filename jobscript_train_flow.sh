#!/bin/bash
#SBATCH --job-name=cpu_check_optuna_train_flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF_train_flow/cpu_check_gandalf_train_flow_%A.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF_train_flow/cpu_check_gaNdalF_train_flow_err_%A.log
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:0
#SBATCH --cpus-per-task=8
#SBATCH --signal=B:TERM@300

source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

export RUN_ID="cpu_check_optuna_2025w32"

srun -n1 python train_gandalf_flow.py -cf LMU_train_flow.cfg