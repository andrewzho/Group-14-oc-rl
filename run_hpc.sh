#!/bin/bash
#SBATCH --job-name=obstacle_tower
#SBATCH --output=obstacle_tower_%j.out
#SBATCH --error=obstacle_tower_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Set up virtual display properly
export DISPLAY=:1
Xvfb $DISPLAY -screen 0 1024x768x24 &
sleep 5  # Give Xvfb time to start

# Make sure the executable is, well, executable
echo "Checking for executable"
ls -la ./ObstacleTower/
chmod +x ./ObstacleTower/obstacletower.x86_64
chmod +x ./ObstacleTower/obstacletower_Data/Plugins/x86_64/libgrpc_csharp_ext.so

# Check that the virtual display is working
echo "Testing X server"
xdpyinfo -display $DISPLAY > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "X server is working"
else
    echo "ERROR: X server is not working"
    exit 1
fi

# Run the test script first to verify environment works
echo "Running test environment script"
python test_env.py

# If the test succeeds, run the actual training
if [ $? -eq 0 ]; then
    echo "Test successful, starting training at $(date)"
    python -m src.train --log_dir logs_hpc --num_steps 5000000
else
    echo "Test failed, not starting training"
    exit 1
fi

# Kill the virtual display
pkill Xvfb
echo "Job completed at $(date)" 