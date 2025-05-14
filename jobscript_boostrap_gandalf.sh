#!/bin/bash
#SBATCH --job-name=boostrap_gandalf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF/bootstrap_gaNdalF_%A_%a.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF/bootstrap_gaNdalF_err_%A_%a.log
##SBATCH --partition=cip
#SBATCH --partition=inter
##SBATCH --gres=gpu:a40:0
#SBATCH --cpus-per-task=8
#SBATCH --array=1-2%1   # Adjust to the number of bootstrap iterations

##module load python/3.11-2024.06
source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

srun -n1 python run_gaNdalF.py --bootstrap --run_number $SLURM_ARRAY_TASK_ID
