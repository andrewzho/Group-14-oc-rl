# New file: stable_baselines_train.py
import os
import numpy as np
import gym
import cv2  # Add import for cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from obstacle_tower_env import ObstacleTowerEnv
from src.create_env import create_obstacle_tower_env

# Add ActionFlattener class
class ActionFlattener:
    """Converts a MultiDiscrete action space to a flattened Discrete action space"""
    
    def __init__(self, nvec):
        """Initialize with MultiDiscrete nvec parameter."""
        self.nvec = nvec
        self.action_space = gym.spaces.Discrete(np.prod(nvec))
        
        # Create mapping from flattened to multi actions
        self.actions = []
        for i in range(self.action_space.n):
            self.actions.append(self._unflatten_action(i))
    
    def lookup_action(self, action):
        """Convert flattened action index to multidiscrete action."""
        return self.actions[action]
    
    def _unflatten_action(self, action):
        """Convert a flattened action index to multidiscrete action."""
        multi_action = []
        for dim_size in self.nvec:
            multi_action.append(action % dim_size)
            action = action // dim_size
        return multi_action

class ObstacleTowerWrapper(gym.Wrapper):
    """Wrapper to make Obstacle Tower compatible with Stable Baselines"""
    
    def __init__(self, env):
        """Initialize the wrapper."""
        super().__init__(env)
        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        
        # Define action space (flattened to Discrete)
        if hasattr(env.action_space, 'n'):
            self.action_space = env.action_space
        else:
            # Flatten the multi-discrete action space
            self.action_flattener = ActionFlattener(env.action_space.nvec)
            self.action_space = self.action_flattener.action_space
    
    def reset(self):
        """Reset the environment."""
        obs = self.env.reset()
        # Process observation to match observation space
        if isinstance(obs, tuple):
            obs = obs[0]  # Get visual observation
        # Resize to 84x84
        obs = cv2.resize(obs, (84, 84))
        return obs
    
    def step(self, action):
        """Step the environment."""
        # Convert action to the format expected by Obstacle Tower
        if hasattr(self, 'action_flattener'):
            action = self.action_flattener.lookup_action(action)
            action = np.array(action)
        
        # Step the environment
        obs, reward, done, info = self.env.step(action)
        
        # Process observation
        if isinstance(obs, tuple):
            obs = obs[0]  # Get visual observation
        # Resize to 84x84
        obs = cv2.resize(obs, (84, 84))
        
        return obs, reward, done, info

def make_obstacle_tower_env(env_path, seed=0, frame_skip=4):
    """Create a vectorized Obstacle Tower environment."""
    def _init():
        env = create_obstacle_tower_env(
            executable_path=env_path,
            realtime_mode=False,
            timeout=300,
            no_graphics=True
        )
        env.seed(seed)
        env = ObstacleTowerWrapper(env)
        # Skip frames to speed up training
        env = MaxAndSkipEnv(env, skip=frame_skip)
        # Monitor for logging
        env = Monitor(env)
        return env
    
    return _init

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", type=str, default="./ObstacleTower/obstacletower.x86_64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--log_dir", type=str, default="logs/sb3")
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_obstacle_tower_env(args.env_path, args.seed)])
    env = VecTransposeImage(env)  # Transpose image to channel-first format expected by SB3
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="obstacle_tower_ppo",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Create PPO agent
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        device="auto"
    )
    
    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO",
    )
    
    # Save the final model
    model.save(os.path.join(args.log_dir, "final_model"))