"""
Simple vectorization wrapper for Obstacle Tower environment.
This avoids the compatibility issues with stable-baselines3's vectorization.
"""
import numpy as np
from gym import spaces

class SimpleVecEnv:
    """Simple vectorized environment wrapper"""
    
    def __init__(self, env):
        """Initialize wrapper with environment"""
        self.env = env
        self.num_envs = 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self):
        """Reset environment and return batch of observations"""
        obs = self.env.reset()
        # Add batch dimension
        return np.expand_dims(obs, axis=0)
        
    def step(self, actions):
        """Take step with first action in batch"""
        # Extract scalar action from batch
        action = actions[0] if isinstance(actions, (list, np.ndarray)) else actions
        
        # Step environment
        obs, reward, done, info = self.env.step(action)
        
        # Add batch dimensions
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward]),
            np.array([done]),
            [info]
        )
        
    def close(self):
        """Close environment"""
        self.env.close()


class FrameStack:
    """Simple frame stacking wrapper"""
    
    def __init__(self, env, n_frames=4):
        """Initialize frame stacking"""
        self.env = env
        self.n_frames = n_frames
        self.frames = None
        
        # Get shape of single observation space
        obs_shape = env.observation_space.shape
        
        # Debug shape information - remove in production
        # print(f"Original observation space: {env.observation_space}")
        # print(f"Original observation shape: {obs_shape}")
        
        # Handle both vectorized and non-vectorized observations
        if len(obs_shape) == 4:  # (N, H, W, C)
            self.batch_dim = True
            h, w, c = obs_shape[1], obs_shape[2], obs_shape[3]
        else:  # (H, W, C)
            self.batch_dim = False
            h, w, c = obs_shape[0], obs_shape[1], obs_shape[2]
        
        # Create new observation space with stacked channels
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c * n_frames),
            dtype=np.uint8
        )
        
        # Pass through action space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        
    def reset(self):
        """Reset environment and initialize frame stack"""
        obs = self.env.reset()
        
        # Initialize frames with copies of initial observation
        self.frames = [obs.copy() for _ in range(self.n_frames)]
        
        # Stack frames
        return self._get_obs()
        
    def step(self, action):
        """Take step and update frame stack"""
        obs, reward, done, info = self.env.step(action)
        
        # Update frame stack
        self.frames.pop(0)
        self.frames.append(obs)
        
        return self._get_obs(), reward, done, info
        
    def _get_obs(self):
        """Stack frames along channel dimension"""
        # For (N, H, W, C) observations, stack on C dimension
        stacked = np.concatenate(self.frames, axis=-1)
        return stacked
        
    def close(self):
        """Close environment"""
        self.env.close()


class VecTransposeImage:
    """Transpose images for PyTorch's channel-first format"""
    
    def __init__(self, env):
        """Initialize transpose wrapper"""
        self.env = env
        self.num_envs = getattr(env, 'num_envs', 1)
        
        # Get shape of observation
        obs_shape = env.observation_space.shape
        
        # Create transposed observation space
        if len(obs_shape) == 3:  # (H, W, C)
            h, w, c = obs_shape
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(c, h, w),
                dtype=np.uint8
            )
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")
        
        # Pass through action space
        self.action_space = env.action_space
        
    def reset(self):
        """Reset and transpose observation"""
        obs = self.env.reset()
        return self._transpose_obs(obs)
        
    def step(self, action):
        """Step and transpose observation"""
        obs, reward, done, info = self.env.step(action)
        return self._transpose_obs(obs), reward, done, info
        
    def _transpose_obs(self, obs):
        """Transpose observation from NHWC to NCHW format"""
        if len(obs.shape) == 4:  # (N, H, W, C)
            return np.transpose(obs, (0, 3, 1, 2))
        else:  # (H, W, C)
            transposed = np.transpose(obs, (2, 0, 1))
            return np.expand_dims(transposed, axis=0)
        
    def close(self):
        """Close environment"""
        self.env.close()


def create_simple_vec_env(env, stack_frames=4):
    """Create vectorized environment with frame stacking"""
    vec_env = SimpleVecEnv(env)
    
    if stack_frames > 1:
        vec_env = FrameStack(vec_env, n_frames=stack_frames)
    
    vec_env = VecTransposeImage(vec_env)
    
    return vec_env