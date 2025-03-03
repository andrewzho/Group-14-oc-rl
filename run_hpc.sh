#!/bin/bash
#SBATCH -A cs175_class_gpu    ## Account to charge
#SBATCH --time=04:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=gpu       ## Partition name
#SBATCH --mem=30GB            ## Allocated Memory
#SBATCH --cpus-per-task 8    ## Number of CPU cores
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs

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