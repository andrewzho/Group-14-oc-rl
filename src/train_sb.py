import gym
from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np
from PIL import Image
import argparse
import torch
import torch.nn as nn
import os
from typing import Dict, List, Tuple, Type, Optional, Callable, Union
import cv2
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('obstacle_tower_sb3')

# Custom feature extractor for Obstacle Tower
class ObstacleTowerFeaturesExtractor(BaseFeaturesExtractor):
    """Custom CNN feature extractor for Obstacle Tower with attention mechanism."""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(ObstacleTowerFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # Assume stacked frames with shape (n_channels * n_stack, height, width)
        n_input_channels = observation_space.shape[0]  # This will be 12 for 4 stacked frames with 3 channels each
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
            
        # Final layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# Custom wrapper for reward shaping
class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RewardShapingWrapper, self).__init__(env)
        self._previous_keys = None
        self._previous_floor = 0
        self._previous_position = None
        self._last_action = None
        self._cumulative_reward = 0
        
    def reset(self, **kwargs):
        """Handle reset with compatibility for both gym and gymnasium interfaces"""
        result = self.env.reset(**kwargs)
        
        # Initialize state tracking variables
        self._previous_keys = 0
        self._previous_floor = 0
        self._previous_position = None
        self._last_action = None
        self._cumulative_reward = 0
        
        # Check if result is a tuple (new gym/gymnasium style)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs, info
        else:
            # Old-style reset (just returns observation)
            return result
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward  # Start with the original reward
        
        # Get current state information
        current_floor = info.get("current_floor", 0)
        current_keys = info.get("total_keys", 0)
        current_position = (
            info.get("x_pos", 0),
            info.get("y_pos", 0),
            info.get("z_pos", 0)
        )
        
        # Floor progression bonus
        if current_floor > self._previous_floor:
            floor_bonus = 2.0 * (1 + 0.5 * (current_floor - self._previous_floor))
            shaped_reward += floor_bonus
            logger.info(f"Floor bonus: reached floor {current_floor}, +{floor_bonus}")
        
        # Key collection bonus
        if current_keys > self._previous_keys:
            key_bonus = 1.0
            shaped_reward += key_bonus
            logger.info(f"Key collected! +{key_bonus}")
        # Door opening bonus
        elif self._previous_keys is not None and current_keys < self._previous_keys:
            door_bonus = 1.5
            shaped_reward += door_bonus
            logger.info(f"Door opened! +{door_bonus}")
        
        # Movement bonus (encourage exploration)
        if self._previous_position is not None:
            distance = sum((current_position[i] - self._previous_position[i])**2 for i in range(3))**0.5
            if distance > 0.5:  # Significant movement
                movement_bonus = 0.01 * distance
                shaped_reward += movement_bonus
        
        # Small time penalty to encourage faster completion
        time_penalty = -0.0001
        shaped_reward += time_penalty
        
        # Update state tracking
        self._previous_keys = current_keys
        self._previous_floor = current_floor
        self._previous_position = current_position
        self._last_action = action
        self._cumulative_reward += shaped_reward
        
        # Add cumulative reward to info
        info["cumulative_reward"] = self._cumulative_reward
        
        return obs, shaped_reward, done, info

# Custom wrapper to preprocess observations
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        
        # New observation space for processed image
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, 84, 84), dtype=np.float32
        )
    
    def observation(self, observation):
        # Extract visual observation and process it
        visual_obs = observation[0]  # Extract visual component
        
        if visual_obs.shape != (84, 84, 3):
            # Resize to 84x84 if necessary
            visual_obs = cv2.resize(visual_obs, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] if necessary
        if visual_obs.dtype == np.uint8:
            visual_obs = visual_obs.astype(np.float32) / 255.0
        
        # Convert to CHW format for PyTorch (channels first)
        visual_obs = np.transpose(visual_obs, (2, 0, 1))
        
        return visual_obs
        
    def reset(self, **kwargs):
        """Handle reset with compatibility for both gym and gymnasium interfaces"""
        try:
            # Try new-style reset (returns observation and info)
            result = self.env.reset(**kwargs)
            
            # Check if result is a tuple (new gym/gymnasium style)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
                return self.observation(obs), info
            else:
                # Old-style reset (just returns observation)
                return self.observation(result)
        except TypeError:
            # If kwargs caused an error, try old-style reset without kwargs
            result = self.env.reset()
            return self.observation(result)

# Custom wrapper for curriculum learning
class CurriculumWrapper(gym.Wrapper):
    def __init__(self, env, starting_floor=0, success_threshold=0.7):
        super(CurriculumWrapper, self).__init__(env)
        self.current_max_floor = starting_floor
        self.success_threshold = success_threshold
        self.floor_successes = {}  # Track success rate per floor
        self.current_floor = starting_floor
        self.episode_count = 0
        
    def reset(self, **kwargs):
        """Handle reset with compatibility for both gym and gymnasium interfaces"""
        # Set floor before reset
        if hasattr(self.env.unwrapped, 'floor'):
            try:
                self.env.unwrapped.floor(self.current_floor)
            except Exception as e:
                logger.warning(f"Failed to set floor: {e}")
        
        # Track episodes
        self.episode_count += 1
        
        # Reset environment
        result = self.env.reset(**kwargs)
        
        # Check if result is a tuple (new gym/gymnasium style)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            # Add curriculum info
            info["curriculum_floor"] = self.current_floor
            info["curriculum_max_floor"] = self.current_max_floor
            return obs, info
        else:
            # Old-style reset (just returns observation)
            return result
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Update floor statistics when episode ends
        if done:
            floor_reached = info.get("current_floor", 0)
            
            # Initialize floor stats if needed
            if floor_reached not in self.floor_successes:
                self.floor_successes[floor_reached] = {"attempts": 0, "successes": 0}
                
            self.floor_successes[floor_reached]["attempts"] += 1
            
            # Consider it a success if agent reached or exceeded current floor
            if floor_reached >= self.current_floor:
                self.floor_successes[floor_reached]["successes"] += 1
                
                # Calculate success rate
                success_rate = (self.floor_successes[floor_reached]["successes"] / 
                               self.floor_successes[floor_reached]["attempts"])
                
                # Update curriculum if success rate is high enough and we have enough samples
                if (success_rate >= self.success_threshold and 
                    self.floor_successes[floor_reached]["attempts"] >= 5):
                    if self.current_max_floor < 10:  # Cap at floor 10
                        self.current_max_floor += 1
                        self.current_floor = self.current_max_floor
                        logger.info(f"Curriculum update: advancing to floor {self.current_floor}")
            
            # Add curriculum info
            info["curriculum_floor"] = self.current_floor
            info["curriculum_max_floor"] = self.current_max_floor
            
        return obs, reward, done, info

def make_env(env_id, rank, seed=0, realtime_mode=False, reward_shaping=True, curriculum=False):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :return: (Callable)
    """
    def _init():
        # Use rank to create unique worker_id and base_port
        worker_id = rank + 1  # Ensure worker_id is at least 1 to avoid conflicts
        base_port = 5005 + (worker_id * 10)  # Different port for each worker
        
        env = ObstacleTowerEnv(
            env_id, 
            retro=False, 
            realtime_mode=realtime_mode,
            worker_id=worker_id,
            timeout_wait=60  # Increased timeout for environment startup
        )
        env.seed(seed + rank)
        env = ObservationWrapper(env)
        
        if reward_shaping:
            env = RewardShapingWrapper(env)
            
        if curriculum:
            env = CurriculumWrapper(env)
            
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Obstacle Tower with Stable-Baselines3")
    parser.add_argument('--env_path', type=str, default="./ObstacleTower/obstacletower.x86_64", 
                        help="Path to Obstacle Tower executable")
    parser.add_argument('--realtime', action='store_true', help="Enable real-time mode for visualization")
    parser.add_argument('--num_envs', type=int, default=1, help="Number of parallel environments")
    parser.add_argument('--n_steps', type=int, default=128, help="Number of steps per update")
    parser.add_argument('--batch_size', type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument('--n_epochs', type=int, default=4, help="Number of epochs for PPO updates")
    parser.add_argument('--total_timesteps', type=int, default=10000000, help="Total timesteps for training")
    parser.add_argument('--reward_shaping', action='store_true', help="Enable reward shaping")
    parser.add_argument('--curriculum', action='store_true', help="Enable curriculum learning")
    parser.add_argument('--frame_stack', type=int, default=4, help="Number of frames to stack")
    parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--log_dir', type=str, default="./logs", help="Directory for logs")
    parser.add_argument('--eval_freq', type=int, default=50000, help="Evaluation frequency in timesteps")
    parser.add_argument('--force_kill', action='store_true', help="Force kill Unity processes before starting")
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Force kill any lingering Unity processes if requested
    if args.force_kill:
        try:
            import psutil
            for proc in psutil.process_iter():
                if "obstacletower" in proc.name().lower() or "unity" in proc.name().lower():
                    logger.info(f"Killing process: {proc.name()} (PID: {proc.pid})")
                    proc.kill()
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to kill Unity processes: {e}")
    
    env = None
    eval_env = None
    
    try:
        # Create vectorized environment
        logger.info(f"Creating {args.num_envs} environments with worker IDs starting from 1")
        if args.num_envs == 1:
            env = DummyVecEnv([make_env(args.env_path, 0, args.seed, args.realtime, args.reward_shaping, args.curriculum)])
        else:
            try:
                env = SubprocVecEnv([make_env(args.env_path, i, args.seed, args.realtime, args.reward_shaping, args.curriculum) 
                                   for i in range(args.num_envs)], 
                                   start_method='spawn')  # Use 'spawn' for better Windows compatibility
            except Exception as e:
                logger.warning(f"Failed to create SubprocVecEnv: {e}. Falling back to DummyVecEnv.")
                # Fall back to DummyVecEnv if SubprocVecEnv fails
                env = DummyVecEnv([make_env(args.env_path, i, args.seed, args.realtime, args.reward_shaping, args.curriculum) 
                                 for i in range(args.num_envs)])
        
        # Apply frame stacking
        logger.info(f"Applying frame stacking with {args.frame_stack} frames")
        env = VecFrameStack(env, n_stack=args.frame_stack)
        
        # Create separate env for evaluation with completely different worker ID range (100+)
        logger.info("Creating evaluation environment with worker ID 100")
        try:
            eval_env = DummyVecEnv([make_env(args.env_path, 100, args.seed + 100, False, args.reward_shaping, False)])
            eval_env = VecFrameStack(eval_env, n_stack=args.frame_stack)
            
            # Initialize evaluation callback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(args.log_dir, "best_model"),
                log_path=os.path.join(args.log_dir, "eval_results"),
                eval_freq=args.eval_freq,
                n_eval_episodes=5,
                deterministic=True,
                render=False
            )
            callbacks = [eval_callback]
        except Exception as e:
            logger.warning(f"Failed to create evaluation environment: {e}. Training without evaluation.")
            eval_env = None
            callbacks = []
        
        # Initialize checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=args.eval_freq,
            save_path=os.path.join(args.log_dir, "checkpoints"),
            name_prefix="ppo_obstacle_tower"
        )
        callbacks.append(checkpoint_callback)
        
        # Define policy kwargs for custom feature extractor
        policy_kwargs = {
            "features_extractor_class": ObstacleTowerFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
        }
        
        # Learning rate schedule
        def linear_schedule(initial_value):
            def func(progress_remaining):
                return progress_remaining * initial_value
            return func
        
        # PPO setup
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.01,  # Encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=os.path.join(args.log_dir, "tensorboard")
        )
        
        # Log training start
        logger.info("Starting training with the following parameters:")
        logger.info(f"Environments: {args.num_envs}")
        logger.info(f"Frame stack: {args.frame_stack}")
        logger.info(f"Reward shaping: {args.reward_shaping}")
        logger.info(f"Curriculum learning: {args.curriculum}")
        logger.info(f"Total timesteps: {args.total_timesteps}")
        logger.info(f"Device: {model.device}")
        
        # Training
        start_time = time.time()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO",
            reset_num_timesteps=True
        )
        
        # Save final model
        model.save(os.path.join(args.log_dir, "final_model"))
        
        # Log training end
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final model saved to {os.path.join(args.log_dir, 'final_model')}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Ensure environments are properly closed
        if env is not None:
            logger.info("Closing training environment")
            env.close()
        if eval_env is not None:
            logger.info("Closing evaluation environment")
            eval_env.close()
            
        # Optional: Force close any remaining Unity processes
        try:
            import psutil
            for proc in psutil.process_iter():
                if "obstacletower" in proc.name().lower():
                    logger.info(f"Closing Unity process: {proc.name()} (PID: {proc.pid})")
                    proc.kill()
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to close Unity processes: {e}")

if __name__ == "__main__":
    main()