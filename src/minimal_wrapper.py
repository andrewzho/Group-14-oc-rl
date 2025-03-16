"""
Minimal wrapper for Obstacle Tower to work with Stable Baselines 3
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from gym import spaces
import gym


class MinimalStackedObservation(gym.ObservationWrapper):
    """
    Stack frames without using vectorized environments.
    This wrapper stacks frames and returns them as a PyTorch tensor directly.
    """
    def __init__(self, env, n_frames=4):
        """Initialize wrapper"""
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = []
        
        # Calculate stacked shape
        original_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(original_shape[2] * n_frames, original_shape[0], original_shape[1]),
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        """Reset environment and initialize frame stack"""
        observation = self.env.reset(**kwargs)
        
        # Fill buffer with the same initial frame
        self.frames = [observation] * self.n_frames
        
        return self._get_observation()
    
    def observation(self, observation):
        """Process observation from environment"""
        self.frames.append(observation)
        if len(self.frames) > self.n_frames:
            self.frames.pop(0)
        
        return self._get_observation()
    
    def _get_observation(self):
        """Convert stacked frames to PyTorch tensor"""
        # Stack frames
        stacked = np.concatenate(self.frames, axis=2)
        
        # Normalize to [0, 1]
        stacked = stacked / 255.0
        
        # Convert to float32
        stacked = stacked.astype(np.float32)
        
        # Transpose from HWC to CHW
        stacked = np.transpose(stacked, (2, 0, 1))
        
        return stacked


def create_wrapped_env(env_path, worker_id=1, seed=None, n_frames=4):
    """
    Create Obstacle Tower environment with minimal wrappers.
    
    Args:
        env_path: Path to Obstacle Tower executable
        worker_id: Worker ID
        seed: Random seed
        n_frames: Number of frames to stack
        
    Returns:
        Wrapped environment
    """
    # Import necessary modules
    import os
    
    # Set environment variables
    os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
    os.environ['OBSTACLE_TOWER_DISABLE_GRAPHICS'] = '1'
    
    # Import obstacle tower environment
    try:
        from obstacle_tower_env import ObstacleTowerEnv
    except ImportError:
        raise ImportError(
            "Could not import obstacle_tower_env. Make sure it's in your Python path."
        )
    
    # Create environment
    env = ObstacleTowerEnv(
        environment_filename=env_path,
        worker_id=worker_id,
        retro=True,
        realtime_mode=False,
        timeout_wait=60
    )
    
    # Set seed if provided
    if seed is not None:
        env.seed(seed)
    
    # Wrap environment
    env = MinimalStackedObservation(env, n_frames=n_frames)
    
    return env


if __name__ == "__main__":
    # Test wrapper
    env = create_wrapped_env("ObstacleTower/ObstacleTower.exe", seed=42)
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Test step
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}, Reward: {reward}, Done: {done}")
        
        if done:
            print("Episode finished early!")
            break
    
    # Clean up
    env.close()