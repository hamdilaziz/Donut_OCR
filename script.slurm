#!/bin/bash
#SBATCH --job-name=metrics
##SBATCH -C v100-32g 
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=8          # number de coeurs CPU alloués à la tâche
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=020:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=metrics_result/output.out
#SBATCH --error=metrics_result/error.out     
#SBATCH -A jqv@v100
#SBATCH -C v100                
set -x
module load cpuarch/amd 
module load pytorch-gpu/py3/1.13.0 
export PYTHONWARNINGS="ignore" 
 
python3 compute_metrics_rimes.py
