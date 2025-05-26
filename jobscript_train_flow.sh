#!/bin/bash
#SBATCH --job-name=train_flow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF_train_flow/gandalf_train_flow.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF_train_flow/gaNdalF_train_flow_err.log
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

##module load python/3.11-2024.06
source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

srun -n1 python train_gandalf_flow.py -cf LMU_train_flow.cfg