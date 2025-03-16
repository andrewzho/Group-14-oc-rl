"""
Memory-efficient wrapper for Obstacle Tower environment
"""
import os
import numpy as np
import gym
from gym import spaces
from collections import deque

# Apply NumPy patch for mlagents_envs
if not hasattr(np, 'bool'):
    setattr(np, 'bool', np.bool_)


class MemoryEfficientWrapper(gym.Env):
    """
    Memory-efficient wrapper for Obstacle Tower that:
    1. Uses lower resolution (42x42 instead of 84x84)
    2. Converts to grayscale to reduce channels
    3. Still provides continuous action space for SAC
    """
    def __init__(self, env_path, worker_id=1, stack_frames=4, seed=None):
        """Initialize wrapper"""
        # Set environment variables
        os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
        os.environ['OBSTACLE_TOWER_DISABLE_GRAPHICS'] = '1'
        
        # Import and create Obstacle Tower environment
        from obstacle_tower_env import ObstacleTowerEnv
        self.env = ObstacleTowerEnv(
            environment_filename=env_path,
            worker_id=worker_id,
            retro=True,
            realtime_mode=False,
            timeout_wait=60
        )
        
        # Apply seed if provided
        if seed is not None:
            self.env.seed(seed)
        
        # Set up frame stacking
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        
        # Get number of discrete actions
        self.num_actions = self.env.action_space.n
        
        # Set up observation space (grayscale, 42x42, stacked)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, 42, 42),  # Grayscale, smaller resolution
            dtype=np.float32
        )
        
        # Create continuous action space for SAC
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    
    def reset(self):
        """Reset environment and stack frames"""
        obs = self.env.reset()
        
        # Process observation
        processed_obs = self._process_observation(obs)
        
        # Initialize frames
        for _ in range(self.stack_frames):
            self.frames.append(processed_obs)
        
        return self._get_observation()
    
    def step(self, action):
        """Take step with continuous action converted to discrete"""
        # Convert continuous action to discrete
        discrete_action = int((action[0] + 1.0001) / 2 * self.num_actions)
        discrete_action = max(0, min(discrete_action, self.num_actions - 1))
        
        # Take step
        obs, reward, done, info = self.env.step(discrete_action)
        
        # Process observation
        processed_obs = self._process_observation(obs)
        
        # Update frame stack
        self.frames.append(processed_obs)
        
        return self._get_observation(), reward, done, info
    
    def _process_observation(self, obs):
        """
        Process observation:
        1. Convert to grayscale
        2. Resize to 42x42
        3. Normalize to [0, 1]
        """
        # Convert to grayscale by taking mean across channels
        gray = np.mean(obs, axis=2, keepdims=False).astype(np.float32)
        
        # Resize to 42x42
        resized = self._resize_image(gray, (42, 42))
        
        # Normalize
        normalized = resized / 255.0
        
        return normalized
    
    def _resize_image(self, image, size):
        """Resize image using simple downsampling"""
        import cv2
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    def _get_observation(self):
        """Stack frames along first dimension"""
        return np.array(self.frames)
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def seed(self, seed=None):
        """Set seed"""
        return self.env.seed(seed)


def create_memory_efficient_env(env_path, worker_id=1, stack_frames=4, seed=None):
    """Create memory-efficient environment"""
    return MemoryEfficientWrapper(
        env_path=env_path,
        worker_id=worker_id,
        stack_frames=stack_frames,
        seed=seed
    )


if __name__ == "__main__":
    """Test wrapper"""
    env = create_memory_efficient_env("ObstacleTower/ObstacleTower.exe")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}, reward={reward}, done={done}")
    
    env.close()