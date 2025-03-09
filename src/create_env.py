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
        obs, reward, done, info = self.env.step(action)
        
        # Add position information to info
        # These are estimates as the exact position is not directly available
        # We'll use the visual observation to estimate movement
        if not hasattr(self, '_prev_frame') or self._prev_frame is None:
            self._prev_frame = obs[0] if isinstance(obs, tuple) else obs
            info['x_pos'] = 0
            info['y_pos'] = 0
            info['z_pos'] = 0
        else:
            # Estimate movement from frame difference
            # This is a simple approach - actual position would require environment access
            current_frame = obs[0] if isinstance(obs, tuple) else obs
            frame_diff = np.mean(np.abs(current_frame - self._prev_frame))
            
            # If there's significant difference, assume movement
            if frame_diff > 0.1:
                # Direction is based on action
                if isinstance(action, (list, np.ndarray)) and len(action) >= 3:
                    move_idx, rot_idx = action[0], action[1]
                    
                    # Update position estimates based on action
                    # This is simplified - actual position would be better
                    if move_idx == 1:  # Forward
                        info['z_pos'] = info.get('z_pos', 0) + 1
                    elif move_idx == 2:  # Backward
                        info['z_pos'] = info.get('z_pos', 0) - 1
                        
                    if rot_idx == 1:  # Right rotation
                        info['x_pos'] = info.get('x_pos', 0) + 0.5
                    elif rot_idx == 2:  # Left rotation
                        info['x_pos'] = info.get('x_pos', 0) - 0.5
            
            self._prev_frame = current_frame
            
            # Track visited positions and add exploration bonus
            # Discretize position to a grid
            pos_key = self._get_position_key(info)
            if pos_key is not None:
                visit_count = self._visited_positions.get(pos_key, 0)
                self._visited_positions[pos_key] = visit_count + 1
                
                # Add to info for use in reward shaping
                info['visit_count'] = visit_count
                
                # Reset visited positions if floor changes to encourage exploration on new floors
                if hasattr(self, '_last_floor') and info.get('current_floor', 0) > self._last_floor:
                    self._visited_positions = {}
                self._last_floor = info.get('current_floor', 0)
        
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
                            config=None):
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
        worker_id=worker_id,
        config=config  # Pass config to the environment
    )
    
    env = PositionTrackingWrapper(env)
    
    return env 