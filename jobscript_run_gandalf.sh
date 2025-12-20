#!/bin/bash
#SBATCH --job-name=run_gandalf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --chdir=/home/p/P.Gebhardt/development/GalaxyFlow/
#SBATCH -o /home/p/P.Gebhardt/Output/gaNdalF/20251220_run_gaNdalF.log
#SBATCH -e /home/p/P.Gebhardt/Output/gaNdalF/20251220_run_gaNdalF_err.log
##SBATCH --partition=cip
#SBATCH --partition=inter
##SBATCH --gres=gpu:a40:0
#SBATCH --cpus-per-task=8
##SBATCH --array=1-10%5  # Adjust the range to the number of bootstrap iterations

##module load python/3.11-2024.06
source /project/ls-gruen/users/patrick.gebhardt/envs/gaNdalF/bin/activate

srun -n1 python run_gaNdalF.py -cf LMU_run_gaNdalF.cfg
