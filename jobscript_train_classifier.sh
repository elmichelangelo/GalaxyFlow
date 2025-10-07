#!/bin/bash
#SBATCH --job-name=cf_test_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF_classifier_training/cf_test_training_%A.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF_classifier_training/cf_test_training_err_%A.log
#SBATCH --partition=inter
#SBATCH --gres=gpu:a40:0
#SBATCH --cpus-per-task=12

##module load python/3.11-2024.06
source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

srun -n1 python train_gandalf_classifier.py -cf LMU_train_classifier.cfg