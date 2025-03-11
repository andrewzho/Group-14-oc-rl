from obstacle_tower_env import ObstacleTowerEnv
import os
import time
import gym
import numpy as np
from collections import deque

class PositionTrackingWrapper(gym.Wrapper):
    """Wrapper that tracks agent position for reward shaping purposes"""
    def __init__(self, env):
        super().__init__(env)
        self._previous_position = None
        self._previous_keys = None
        self._visited_positions = {}  # Track visited positions
        self._position_resolution = 1.0  # Resolution for position discretization
        
    def step(self, action):
        """
        Wrapper around env.step that handles both single actions and batched actions from vectorized environments.
        """
        # Convert to numpy array if it's not already
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # Handle single action vs batched actions from vectorized environments
        if len(action.shape) == 1:
            # Single action - reshape to expected format
            obs, reward, done, info = self.env.step(action.reshape([1, -1]))
            return obs, reward, done, info
        else:
            # Batch of actions - reshape each one and step environment
            obs, reward, done, info = self.env.step(action.reshape([action.shape[0], -1]))
            return obs, reward, done, info
    
    def _get_position_key(self, info):
        """Create a discrete position key for the current position"""
        if 'x_pos' not in info or 'z_pos' not in info:
            return None
            
        # Add floor to make positions unique across floors
        floor = info.get('current_floor', 0)
        # Discretize position to reduce number of unique positions
        x = round(info['x_pos'] / self._position_resolution) * self._position_resolution
        z = round(info['z_pos'] / self._position_resolution) * self._position_resolution
        
        return f"{floor}_{x}_{z}"
    
    def reset(self):
        obs = self.env.reset()
        self._previous_position = None
        self._previous_keys = None
        self._prev_frame = None
        # Don't reset visited positions to maintain exploration knowledge across episodes
        # But do limit the size to prevent memory growth
        if len(self._visited_positions) > 10000:
            # Keep only the most visited positions
            self._visited_positions = dict(sorted(self._visited_positions.items(), 
                                                 key=lambda item: item[1], 
                                                 reverse=True)[:5000])
        return obs

def create_obstacle_tower_env(executable_path='./ObstacleTower/obstacletower.x86_64', 
                            realtime_mode=False, 
                            timeout=300,
                            no_graphics=True,
                            config=None,
                            worker_id=None):
    """Create an Obstacle Tower environment with appropriate settings for HPC."""
    # Set environment variables
    os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
    
    # If on HPC, force no graphics
    if no_graphics:
        os.environ['OBSTACLE_TOWER_DISABLE_GRAPHICS'] = '1'
    
    # Create environment with longer timeout
    if worker_id is None:
        worker_id = int(time.time()) % 10000  # Random worker ID
        
    env = ObstacleTowerEnv(
        executable_path,
        retro=False,
        realtime_mode=realtime_mode,
        timeout_wait=timeout,
        worker_id=worker_id,
        config=config  # Pass config to the environment
    )
    
    env = PositionTrackingWrapper(env)
    
    return env 