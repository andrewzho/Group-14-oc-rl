"""
Custom environment wrappers for the Obstacle Tower Challenge.
These wrappers enhance the environment with features like:
- Observation preprocessing
- Reward shaping
- Action space modifications
- Episode termination conditions
"""
import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, List, Any, Optional, Union

import cv2
from collections import deque

import gym
import numpy as np

class StableBaselinesFix(gym.Wrapper):
    """
    Ensures observations are compatible with Stable-Baselines3 vectorized environments.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Handle multiple return values (newer gym API)
        if isinstance(obs, tuple):
            obs = obs[0]
            
        # Ensure observation matches expected shape exactly
        if isinstance(obs, np.ndarray) and obs.shape == self.observation_space.shape:
            # Already matches, no change needed
            return obs
        else:
            # Force correct shape
            print(f"Warning: Reshaping observation from {obs.shape if hasattr(obs, 'shape') else 'unknown'} to {self.observation_space.shape}")
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
    
    def step(self, action):
        result = self.env.step(action)
        
        # Handle different gym API versions
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            # Newer gym API
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            
        # Ensure observation matches expected shape exactly
        if isinstance(obs, np.ndarray) and obs.shape == self.observation_space.shape:
            # Already matches, no change needed
            return obs, reward, done, info
        else:
            # Force correct shape
            print(f"Warning: Reshaping observation from {obs.shape if hasattr(obs, 'shape') else 'unknown'} to {self.observation_space.shape}")
            return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), reward, done, info

class GymVersionBridge(gym.Wrapper):
    """Compatibility wrapper to handle different Gym API versions"""
    
    def reset(self, **kwargs):
        """Bridge between old and new Gym reset API"""
        try:
            # Try new API (returns obs, info)
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                return result
            else:
                # Old API returned just obs
                return result, {}
        except ValueError as e:
            if "too many values to unpack" in str(e):
                # Handle case where env uses new API but caller expects old
                obs = self.env.reset(**kwargs)[0]
                return obs
            raise

class GrayscaleWrapper(gym.ObservationWrapper):
    """
    Converts RGB observations to grayscale to reduce dimensionality.
    """
    def __init__(self, env):
        super().__init__(env)
        
        # Handle various observation formats
        obs_shape = self.observation_space.shape
        
        # Check for existing frame stacking or unusual shapes
        if len(obs_shape) == 3:
            if obs_shape[2] > 4:  # Likely already processed/stacked
                # Just pass through if already processed
                self.process = False
                return
            else:
                self.process = True
                # Update observation space for normal case
                self.observation_space = spaces.Box(
                    low=0, high=255, 
                    shape=(obs_shape[0], obs_shape[1], 1), 
                    dtype=np.uint8
                )
        else:
            # Unsupported shape, just pass through
            self.process = False
    
    def observation(self, observation):
        """Convert to grayscale if needed"""
        if not self.process:
            return observation
            
        if observation.shape[2] == 1:
            return observation  # Already grayscale
        
        # Convert to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # Keep the same dimension but with 1 channel
        return np.expand_dims(gray, -1)


class ResizeObservationWrapper(gym.ObservationWrapper):
    """
    Resizes observations to a specified size.
    """
    def __init__(self, env, size=(84, 84)):
        super().__init__(env)
        self.size = size
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(size[0], size[1], self.observation_space.shape[2]), 
            dtype=np.uint8
        )
    
    def observation(self, observation):
        """Resize observation"""
        return cv2.resize(
            observation, self.size, 
            interpolation=cv2.INTER_AREA
        )


class FrameStackWrapper(gym.Wrapper):
    """
    Stacks the last n frames together for temporal information.
    """
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        
        # Update observation space
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(old_shape[0], old_shape[1], old_shape[2] * n_frames),
            dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        """Reset environment and frame stack"""
        result = self.env.reset(**kwargs)
    
        # Handle both old and new Gym API
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        observation = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(observation.copy())
        return self._get_observation()
    
    def step(self, action):
        """Step environment and update frame stack"""
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Concatenate frames along channel dimension"""
        return np.concatenate(list(self.frames), axis=2)


class NormalizeObservationWrapper(gym.ObservationWrapper):
    """
    Normalizes pixel values to [0, 1] range.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, observation):
        """Normalize observation to [0, 1]"""
        return np.array(observation, dtype=np.float32) / 255.0


class RewardShapingWrapper(gym.RewardWrapper):
    """
    Enhances sparse rewards in Obstacle Tower with shaped rewards.
    """
    def __init__(self, env, floor_bonus=10.0, key_bonus=5.0, door_bonus=2.0, 
                 time_penalty=-0.01, movement_bonus=0.001):
        super().__init__(env)
        self.floor_bonus = floor_bonus
        self.key_bonus = key_bonus
        self.door_bonus = door_bonus
        self.time_penalty = time_penalty
        self.movement_bonus = movement_bonus
        
        # Track state for reward shaping
        self.last_floor = 0
        self.last_keys = 0
        self.last_doors = 0
        self.last_position = None
    
    def reset(self, **kwargs):
        
        """Reset tracked state"""
        result = self.env.reset(**kwargs)
    
        # Handle both old and new Gym API
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self.last_floor = 0
        self.last_keys = 0
        self.last_doors = 0
        self.last_position = None
        return obs
    
    def step(self, action):
        """Step with shaped rewards"""
        obs, reward, done, info = self.env.step(action)
        
        # Get current state
        curr_floor = info.get("floor", 0)
        curr_keys = info.get("keys_collected", 0)
        curr_doors = info.get("doors_opened", 0)
        curr_position = info.get("agent_position", None)
        
        # Calculate shaped reward
        shaped_reward = reward
        
        # Floor progression bonus
        if curr_floor > self.last_floor:
            shaped_reward += self.floor_bonus * (curr_floor - self.last_floor)
            self.last_floor = curr_floor
        
        # Key collection bonus
        if curr_keys > self.last_keys:
            shaped_reward += self.key_bonus * (curr_keys - self.last_keys)
            self.last_keys = curr_keys
        
        # Door opening bonus
        if curr_doors > self.last_doors:
            shaped_reward += self.door_bonus * (curr_doors - self.last_doors)
            self.last_doors = curr_doors
        
        # Movement bonus (exploring new areas)
        if curr_position is not None and self.last_position is not None:
            # Calculate distance moved
            distance = np.linalg.norm(
                np.array(curr_position) - np.array(self.last_position)
            )
            if distance > 0.5:  # Only reward significant movement
                shaped_reward += self.movement_bonus * distance
        
        self.last_position = curr_position
        
        # Time penalty to encourage efficiency
        shaped_reward += self.time_penalty
        
        return obs, shaped_reward, done, info
    
    def reward(self, reward):
        """This method is required by gym.RewardWrapper but we override step instead"""
        return reward


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeats actions and returns the last observation and accumulated reward.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        """Repeat action and accumulate reward"""
        total_reward = 0.0
        done = False
        info = {}
        
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, info


class InfoLoggerWrapper(gym.Wrapper):
    """
    Logs useful information about the environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_counter = 0
        self.step_counter = 0
        self.episode_rewards = []
        self.episode_floors = []
        self.episode_keys = []
        self.episode_doors = []
        
        # Current episode tracking
        self.current_reward = 0
        self.max_floor = 0
        self.max_keys = 0
        self.max_doors = 0
        
    def reset(self, **kwargs):
        """Reset counters for new episode"""
        result = self.env.reset(**kwargs)
    
        # Handle both old and new Gym API
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        
        # Log completed episode info
        if self.step_counter > 0:
            self.episode_rewards.append(self.current_reward)
            self.episode_floors.append(self.max_floor)
            self.episode_keys.append(self.max_keys)
            self.episode_doors.append(self.max_doors)
            
            # Print episode summary
            if self.episode_counter % 10 == 0:
                print(f"Episode {self.episode_counter}: "
                      f"Reward={self.current_reward:.2f}, "
                      f"Floor={self.max_floor}, "
                      f"Keys={self.max_keys}, "
                      f"Doors={self.max_doors}")
        
        # Reset episode tracking
        self.episode_counter += 1
        self.current_reward = 0
        self.max_floor = 0
        self.max_keys = 0
        self.max_doors = 0
        
        return obs
    
    def step(self, action):
        """Track episode progress"""
        obs, reward, done, info = self.env.step(action)
        
        # Update counters
        self.step_counter += 1
        self.current_reward += reward
        
        # Update episode stats
        if "floor" in info:
            self.max_floor = max(self.max_floor, info["floor"])
            info["max_floor"] = self.max_floor
        
        if "keys_collected" in info:
            self.max_keys = max(self.max_keys, info["keys_collected"])
            info["max_keys"] = self.max_keys
            
        if "doors_opened" in info:
            self.max_doors = max(self.max_doors, info["doors_opened"])
            info["max_doors"] = self.max_doors
        
        # Add episode info
        info["episode_num"] = self.episode_counter
        info["step_num"] = self.step_counter
        info["episode_reward"] = self.current_reward
        
        return obs, reward, done, info


class TimeLimit(gym.Wrapper):
    """
    Ends episodes after a specified number of steps.
    """
    def __init__(self, env, max_steps=1000):
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0
        
    def reset(self, **kwargs):
        """Reset step counter"""
        self.step_count = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Count steps and terminate if limit reached"""
        obs, reward, done, info = self.env.step(action)
        self.step_count += 1
        
        # End episode if step limit reached
        if self.step_count >= self.max_steps:
            done = True
            info["timeout"] = True
        else:
            info["timeout"] = False
            
        return obs, reward, done, info


def make_obstacle_tower_env(env, grayscale=True, resize=(84, 84), normalize=True, 
                           frame_stack=4, frame_skip=4, reward_shaping=True, 
                           time_limit=1000, log_info=True):
    """
    Creates a fully preprocessed Obstacle Tower environment with all wrappers.
    
    Args:
        env: Raw Obstacle Tower environment
        grayscale: Whether to convert observations to grayscale
        resize: Size to resize observations to
        normalize: Whether to normalize observations to [0, 1]
        frame_stack: Number of frames to stack
        frame_skip: Number of frames to skip (action repeat)
        reward_shaping: Whether to enable reward shaping
        time_limit: Maximum steps per episode
        log_info: Whether to log episode information
        
    Returns:
        Wrapped environment
    """
    # Apply observation preprocessing
    if resize is not None:
        env = ResizeObservationWrapper(env, size=resize)
    
    if grayscale:
        env = GrayscaleWrapper(env)
    
    # Frame manipulation
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)
    
    if frame_stack > 1:
        env = FrameStackWrapper(env, n_frames=frame_stack)
    
    # Reward and episode modifications
    if reward_shaping:
        env = RewardShapingWrapper(env)
    
    if time_limit > 0:
        env = TimeLimit(env, max_steps=time_limit)
    
    # Observation normalization (keep last)
    if normalize:
        env = NormalizeObservationWrapper(env)
    
    # Logging (very last)
    if log_info:
        env = InfoLoggerWrapper(env)
    
    return env