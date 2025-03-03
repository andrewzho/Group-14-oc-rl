#!/bin/bash
#SBATCH --job-name=obstacle_debug
#SBATCH --output=obstacle_debug_%j.out
#SBATCH --error=obstacle_debug_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Set up virtual display
export DISPLAY=:1
Xvfb $DISPLAY -screen 0 1024x768x24 &

# Run your test script with verbose output
python test_env.py > env_test_detailed.log 2>&1

# Kill the virtual display
pkill Xvfb 