#!/bin/bash
#SBATCH --job-name=run_gandalf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH -o /dss/dsshome1/04/di97tac/logs/run_gandalf_%A.log
#SBATCH -e /dss/dsshome1/04/di97tac/logs/run_gandalf_err_%A.log
#SBATCH --qos=mcml
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1

srun --container-image='/dss/dssfs02/lwp-dss-0001/pn76fa/pn76fa-dss-0000/di97tac/gandalf_container.sqsh' --container-mounts=/dss/dssfs02/lwp-dss-0001/pn76fa/pn76fa-dss-0000/di97tac/:/mnt/project -n1 python /dss/dsshome1/04/di97tac/development/GalaxyFlow/run_gaNdalF.py -cf LRZ_run_gandalf.cfg