"""
Obstacle Tower wrapper with action space conversion for SAC compatibility
"""
import os
import numpy as np
import gym
from gym import spaces
from collections import deque

# Apply NumPy patch for mlagents_envs
if not hasattr(np, 'bool'):
    setattr(np, 'bool', np.bool_)


class ObstacleTowerSACWrapper(gym.Env):
    """
    Gym wrapper for Obstacle Tower that includes:
    - Frame stacking
    - Channel-first (PyTorch) observation format
    - Normalization
    - Conversion from discrete to continuous action space for SAC
    
    This wrapper is directly compatible with Stable-Baselines3 SAC.
    """
    def __init__(self, env_path, worker_id=1, stack_frames=4, seed=None,
                 grayscale=False, normalize=True):
        """
        Initialize wrapper.
        
        Args:
            env_path: Path to Obstacle Tower executable
            worker_id: Worker ID for Unity environment
            stack_frames: Number of frames to stack
            seed: Random seed
            grayscale: Whether to convert observations to grayscale
            normalize: Whether to normalize observations to [0, 1]
        """
        # Set environment variables
        os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
        os.environ['OBSTACLE_TOWER_DISABLE_GRAPHICS'] = '1'
        
        # Import and create Obstacle Tower environment
        try:
            from obstacle_tower_env import ObstacleTowerEnv
        except ImportError:
            raise ImportError(
                "Could not import obstacle_tower_env. Make sure it's in your Python path."
            )
        
        self.env = ObstacleTowerEnv(
            environment_filename=env_path,
            worker_id=worker_id,
            retro=True,  # Use retro for simpler observation space
            realtime_mode=False,
            timeout_wait=60
        )
        
        # Apply seed if provided
        if seed is not None:
            self.env.seed(seed)
        
        # Set up frame stacking
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        
        # Get original observation shape
        orig_obs_space = self.env.observation_space
        self.obs_shape = orig_obs_space.shape
        
        # Other preprocessing options
        self.grayscale = grayscale
        self.normalize = normalize
        
        # Calculate output shape
        if grayscale:
            self.processed_obs_shape = (stack_frames, self.obs_shape[0], self.obs_shape[1])
        else:
            self.processed_obs_shape = (self.obs_shape[2] * stack_frames, self.obs_shape[0], self.obs_shape[1])
            
        # Set up observation space
        if normalize:
            low, high = 0.0, 1.0
            dtype = np.float32
        else:
            low, high = 0, 255
            dtype = np.uint8
            
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=self.processed_obs_shape,
            dtype=dtype
        )
        
        # Get original discrete action space
        self.original_action_space = self.env.action_space
        self.num_actions = self.original_action_space.n
        
        # Create continuous action space for SAC
        # We'll use a Box space with shape (1,) where the continuous action
        # will be mapped to the closest discrete action
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
    def reset(self):
        """Reset environment and initialize frame stack"""
        obs = self.env.reset()
        
        # Apply preprocessing
        if self.grayscale:
            obs = self._to_grayscale(obs)
            
        # Initialize frames with copies of initial observation
        for _ in range(self.stack_frames):
            self.frames.append(obs.copy())
            
        # Return stacked observation
        return self._get_observation()
        
    def step(self, action):
        """
        Take step with continuous action converted to discrete.
        
        Args:
            action: Continuous action from SAC (shape [1,])
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Convert continuous action to discrete
        # Map from [-1, 1] to [0, num_actions-1]
        # Add small epsilon to avoid edge case at exactly 1.0
        discrete_action = int((action[0] + 1.0001) / 2 * self.num_actions) % self.num_actions
        
        # Take step with discrete action
        obs, reward, done, info = self.env.step(discrete_action)
        
        # Apply preprocessing
        if self.grayscale:
            obs = self._to_grayscale(obs)
            
        # Update frame stack
        self.frames.append(obs)
        
        # Return stacked observation and other values
        return self._get_observation(), reward, done, info
        
    def _to_grayscale(self, obs):
        """Convert RGB observation to grayscale"""
        gray = np.mean(obs, axis=2, keepdims=True)
        return gray
        
    def _get_observation(self):
        """Stack frames and transpose for PyTorch"""
        if self.grayscale:
            # For grayscale, stack along first dimension
            stacked = np.stack([f.squeeze() for f in self.frames], axis=0)
        else:
            # For RGB, concatenate along channel dimension, then transpose
            stacked = np.concatenate(list(self.frames), axis=2)
            stacked = np.transpose(stacked, (2, 0, 1))
            
        # Normalize if requested
        if self.normalize:
            stacked = stacked.astype(np.float32) / 255.0
            
        return stacked
        
    def close(self):
        """Close environment"""
        self.env.close()
        
    def seed(self, seed=None):
        """Set random seed"""
        return self.env.seed(seed)


def create_obstacle_tower_env_for_sac(env_path, worker_id=1, stack_frames=4, seed=None):
    """
    Create Obstacle Tower environment wrapped for SAC.
    
    Args:
        env_path: Path to Obstacle Tower executable
        worker_id: Worker ID
        stack_frames: Number of frames to stack
        seed: Random seed
        
    Returns:
        ObstacleTowerSACWrapper: Wrapped environment
    """
    return ObstacleTowerSACWrapper(
        env_path=env_path, 
        worker_id=worker_id,
        stack_frames=stack_frames,
        seed=seed,
        grayscale=False,  # RGB input
        normalize=True  # Normalize to [0, 1]
    )


if __name__ == "__main__":
    # Test wrapper
    env = create_obstacle_tower_env_for_sac("ObstacleTower/ObstacleTower.exe", seed=42)
    print(f"Created environment: {type(env).__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test reset
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation min/max: {obs.min():.4f}/{obs.max():.4f}")
    
    # Test step with continuous action
    for i in range(5):
        action = env.action_space.sample()  # Random continuous action
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}, Action: {action}, Reward: {reward}, Done: {done}")
    
    env.close()