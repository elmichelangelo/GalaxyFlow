#!/bin/bash
#SBATCH --job-name=cf_train_final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -o /dss/dsshome1/04/di97tac/logs/cf_train_final_%A.log
#SBATCH -e /dss/dsshome1/04/di97tac/logs/cf_train_final_err_%A.log
#SBATCH --qos=mcml
#SBATCH --partition=mcml-hgx-h100-94x4
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=3

srun --container-image='/dss/dssfs02/lwp-dss-0001/pn76fa/pn76fa-dss-0000/di97tac/gandalf_container.sqsh' --container-mounts=/dss/dssfs02/lwp-dss-0001/pn76fa/pn76fa-dss-0000/di97tac/:/mnt/project -n1 python /dss/dsshome1/04/di97tac/development/GalaxyFlow/train_gandalf_flow.py -cf LRZ_train_flow.cfg