#!/bin/bash
#SBATCH --job-name=classifier_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF_train_classifer/gandalf_train_classifer_new_data.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF_train_classifer/gandalf_train_classifer_new_data_err.log
#SBATCH --partition=cluster
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8

##module load python/3.11-2024.06
source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

srun -n1 python train_gandalf_classifier.py -cf LMU_train_classifier.cfg