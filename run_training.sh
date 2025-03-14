#!/bin/bash
#SBATCH -A cs175_class_gpu ## Account to charge
#SBATCH --time=04:00:00 ## Maximum running time of program
#SBATCH --nodes=1 ## Number of nodes.
## Set to 1 if you are using GPU.
#SBATCH --partition=free-gpu ## Partition name
#SBATCH --mem=8GB ## Allocated Memory
#SBATCH --cpus-per-task=16 ## Number of CPU cores
#SBATCH --gres=gpu:V100:1 ## Type and the number of GPUs
## Don't change the GPU numbers.
## Follow https://rcic.uci.edu/hpc3/specs.html#specs
## to see all available GPUs.

Xvfb $DISPLAY -screen 0 1024x768x24 &
export DISPLAY=:44

# Make sure ObstacleTower executable has correct permissions
chmod +x ObstacleTower/obstacletower.x86_64

python -m src.train --log_dir logs_hpc --num_steps 5000000 
