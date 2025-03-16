"""
Final PPO implementation for Obstacle Tower Challenge
"""
import os
import argparse
import numpy as np
import torch
import gym
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Apply NumPy patch
if not hasattr(np, 'bool'):
    setattr(np, 'bool', np.bool_)


class ObstacleTowerPPOWrapper(gym.Wrapper):
    """Simple wrapper for Obstacle Tower compatible with PPO"""
    def __init__(self, env_path, worker_id=1, seed=None):
        # Set environment variables
        os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':1')
        os.environ['OBSTACLE_TOWER_DISABLE_GRAPHICS'] = '1'
        
        # Create environment
        from obstacle_tower_env import ObstacleTowerEnv
        env = ObstacleTowerEnv(
            environment_filename=env_path,
            worker_id=worker_id,
            retro=True,
            realtime_mode=False
        )
        
        # Initialize wrapper
        super().__init__(env)
        
        # Set seed
        if seed is not None:
            self.env.seed(seed)


def main():
    """Train PPO agent on Obstacle Tower"""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-path", type=str, required=True, help="Path to Obstacle Tower executable")
    parser.add_argument("--total-timesteps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()
    
    # Create vectorized environment
    def make_env():
        # Create environment
        env = ObstacleTowerPPOWrapper(args.env_path, 1, 42)
        # Add Monitor for episode stats
        env = Monitor(env)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Create PPO agent
    print("Creating PPO agent...")
    model = PPO(
        policy="CnnPolicy",
        env=env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=3e-4,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    # Train agent
    print(f"Training for {args.total_timesteps} timesteps...")
    model.learn(total_timesteps=args.total_timesteps)
    
    # Save the trained model
    model.save("obstacle_tower_ppo")
    
    # Clean up
    env.close()
    
    print("Training complete!")


if __name__ == "__main__":
    main()