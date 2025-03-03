from obstacle_tower_env import ObstacleTowerEnv
import os
import time

def create_obstacle_tower_env(executable_path='./ObstacleTower/obstacletower.x86_64', 
                            realtime_mode=False, 
                            timeout=300,
                            no_graphics=True):
    """Create an Obstacle Tower environment with appropriate settings for HPC."""
    # Set environment variables
    os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
    
    # If on HPC, force no graphics
    if no_graphics:
        os.environ['OBSTACLE_TOWER_DISABLE_GRAPHICS'] = '1'
    
    # Create environment with longer timeout
    worker_id = int(time.time()) % 10000  # Random worker ID
    env = ObstacleTowerEnv(
        executable_path,
        retro=False,
        realtime_mode=realtime_mode,
        timeout_wait=timeout,
        worker_id=worker_id
    )
    
    return env 