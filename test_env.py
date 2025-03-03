import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
import time
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

logger.info("Starting environment test")
try:
    logger.info("Creating environment")
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower.x86_64', retro=False, realtime_mode=False, timeout_wait=300)
    logger.info("Environment created successfully")
    
    logger.info("Resetting environment")
    start_time = time.time()
    obs = env.reset()
    logger.info(f"Reset completed in {time.time() - start_time:.2f} seconds")
    
    logger.info(f"Taking 10 random actions")
    for i in range(10):
        action = env.action_space.sample()
        logger.info(f"Action {i+1}: {action}")
        start_time = time.time()
        obs, reward, done, info = env.step(action)
        logger.info(f"Step {i+1} completed in {time.time() - start_time:.2f} seconds. Reward: {reward}")
        
    logger.info("Closing environment")
    env.close()
    logger.info("Test completed successfully")
except Exception as e:
    logger.error(f"Error during test: {e}", exc_info=True) 