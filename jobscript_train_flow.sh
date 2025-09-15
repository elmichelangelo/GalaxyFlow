#!/bin/bash
#SBATCH --job-name=single_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF_train_flow/single_run_%A.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF_train_flow/single_run_err_%A.log
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:0
#SBATCH --cpus-per-task=4
#SBATCH --signal=B:TERM@300

source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

srun -n1 python train_gandalf_flow.py -cf LMU_train_flow.cfg