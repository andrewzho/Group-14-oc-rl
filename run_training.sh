#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=04:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=30GB            ## Allocated Memory
#SBATCH --cpus-per-task 8    ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs

# Display job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# # Load modules (adjust based on your HPC setup)
# # You may need to change these based on your specific HPC system
# module load python/3.8
# module load cuda/11.7

# # Activate virtual environment (if you've created one)
# source ~/obstacle_venv/bin/activate

# # Navigate to project directory (change to your project path)
# cd ~/obstacle-tower-project

# Check for existing checkpoints and use the latest one if available
CHECKPOINT=""
LATEST_CHECKPOINT=$(ls -t logs_hpc/checkpoints/step_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
    CHECKPOINT="--checkpoint $LATEST_CHECKPOINT"
    echo "Found checkpoint: $LATEST_CHECKPOINT"
fi

# Create directories
mkdir -p logs_hpc/checkpoints

# Make sure ObstacleTower executable has correct permissions
chmod +x ./ObstacleTower/obstacletower

# Run training script
echo "Starting training at $(date)"
python src/train.py --log_dir logs_hpc --num_steps 5000000 $CHECKPOINT

# Print completion information
echo "Job completed at $(date)"