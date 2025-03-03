import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
import time
import logging
import os
import sys
import subprocess

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Print system information
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Contents of ./ObstacleTower: {os.listdir('./ObstacleTower') if os.path.exists('./ObstacleTower') else 'Directory not found'}")

# Check for the display
logger.info(f"DISPLAY environment variable: {os.environ.get('DISPLAY', 'Not set')}")
try:
    # Try to get information about the X display
    result = subprocess.run(["xdpyinfo"], capture_output=True, text=True)
    logger.info(f"xdpyinfo output: {result.stdout[:200]}...")  # Print first 200 chars
except Exception as e:
    logger.error(f"Error running xdpyinfo: {e}")

logger.info("Starting environment test")
try:
    executable_path = './ObstacleTower/obstacletower.x86_64'
    logger.info(f"Creating environment with executable: {executable_path}")
    logger.info(f"Executable exists: {os.path.exists(executable_path)}")
    
    env = ObstacleTowerEnv(
        executable_path, 
        retro=False, 
        realtime_mode=False,
        timeout_wait=300,  # Longer timeout
        worker_id=int(time.time()) % 10000  # Random worker ID to avoid conflicts
    )
    logger.info("Environment created successfully")
    
    logger.info("Resetting environment")
    start_time = time.time()
    obs = env.reset()
    logger.info(f"Reset completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"Observation shape: {obs[0].shape if obs is not None else 'None'}")
    
    logger.info(f"Taking 3 random actions")
    for i in range(3):
        action = env.action_space.sample()
        logger.info(f"Action {i+1}: {action}")
        start_time = time.time()
        obs, reward, done, info = env.step(action)
        logger.info(f"Step {i+1} completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Reward: {reward}, Done: {done}, Info: {info}")
        
    logger.info("Closing environment")
    env.close()
    logger.info("Test completed successfully")
except Exception as e:
    logger.error(f"Error during test: {e}", exc_info=True) 