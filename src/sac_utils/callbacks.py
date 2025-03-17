"""
Custom callbacks for monitoring and enhancing the training process.
"""
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from src.sac_utils .demo_buffer import DemonstrationBuffer, DemonstrationRecorder


class TensorboardCallback(BaseCallback):
    """
    Custom callback for detailed Tensorboard logging during training.
    Logs metrics beyond what Stable Baselines 3 logs by default.
    Fixed to avoid attribute error.
    """
    def __init__(self, verbose=0):
        """Initialize callback"""
        super().__init__(verbose)
        # Don't set training_env here
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_floor_reached = []
        self.episode_times = []
        self.episode_keys_collected = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_floor = 0
        self.current_episode_keys = 0

    def _on_training_start(self) -> None:
        """
        Called at the start of training
        """
        # Only set training_env here when it's safe to do so
        # self.training_env = self.model.env
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """
        Called at each step of training
        """
        # Assuming single environment (unwrap if vectorized)
        info = self.locals.get('infos', [{}])[0]
        reward = self.locals.get('rewards', [0])[0]
        done = self.locals.get('dones', [False])[0]
        
        # Update episode stats
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Extract Obstacle Tower specific info
        if "floor" in info:
            self.current_episode_floor = max(self.current_episode_floor, info["floor"])
        if "keys_collected" in info:
            self.current_episode_keys = max(self.current_episode_keys, info["keys_collected"])
        
        # If episode is done, record stats
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_floor_reached.append(self.current_episode_floor)
            self.episode_keys_collected.append(self.current_episode_keys)
            
            # Log to tensorboard
            if len(self.episode_rewards) % 10 == 0:  # Log every 10 episodes
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                mean_floor = np.mean(self.episode_floor_reached[-10:])
                mean_keys = np.mean(self.episode_keys_collected[-10:])
                
                self.logger.record("rollout/ep_rew_mean", mean_reward)
                self.logger.record("rollout/ep_len_mean", mean_length)
                self.logger.record("metrics/floor_reached", mean_floor)
                self.logger.record("metrics/keys_collected", mean_keys)
                
                # Also log agent-specific metrics if available
                if hasattr(self.model, "custom_logger"):
                    for key, values in self.model.custom_logger.items():
                        if values:
                            self.logger.record(f"agent/{key}", np.mean(values[-10:]))
                            # Clear the buffer after logging
                            if len(values) > 100:
                                self.model.custom_logger[key] = values[-100:]
                
                self.logger.dump(self.num_timesteps)
            
            # Reset episode stats
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_floor = 0
            self.current_episode_keys = 0
            
        return True


class DemonstrationRecorderCallback(BaseCallback):
    """
    Records successful episodes as demonstrations for future training.
    """
    def __init__(
        self, 
        demo_buffer: DemonstrationBuffer,
        min_reward_percentile: float = 80.0,
        min_floor: int = 1,
        verbose: int = 0
    ):
        """
        Initialize callback.
        
        Args:
            demo_buffer: Buffer to store demonstrations
            min_reward_percentile: Minimum reward percentile to record (0-100)
            min_floor: Minimum floor reached to consider recording
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.demo_buffer = demo_buffer
        self.min_reward_percentile = min_reward_percentile
        self.min_floor = min_floor
        self.recorder = DemonstrationRecorder(demo_buffer)
        
        # Tracking variables
        self.current_obs = None
        self.episode_obs = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_next_obs = []
        self.episode_dones = []
        self.episode_infos = []
        self.recording = False
        
        # Statistics
        self.episode_rewards_history = []
        self.recorded_episodes = 0
        
    def _on_training_start(self) -> None:
        """Called at start of training"""
        pass
        
    def _on_rollout_start(self) -> None:
        """Called at start of rollout"""
        if not self.recording:
            self.episode_obs = []
            self.episode_actions = []
            self.episode_rewards = []
            self.episode_next_obs = []
            self.episode_dones = []
            self.episode_infos = []
            
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        Records the current transition for potential demonstration.
        """
        # Extract current step info
        obs = self.locals.get('new_obs', None)
        if obs is None or isinstance(obs, tuple):  # Handle edge cases
            return True
            
        # For vectorized environments, take the first observation
        import numpy as np
        if isinstance(obs, np.ndarray) and len(obs.shape) > 3:
            obs = obs[0]
            
        action = self.locals.get('action', None)
        if isinstance(action, np.ndarray) and len(action.shape) > 1:
            action = action[0]
            
        reward = self.locals.get('reward', [0])[0]
        done = self.locals.get('done', [False])[0]
        info = self.locals.get('info', [{}])[0]
        
        # Record transition for current episode
        if self.current_obs is not None:
            self.episode_obs.append(self.current_obs.copy())
            if action is not None:
                self.episode_actions.append(action.copy())
            else:
                # Handle None action case with a default placeholder
                import numpy as np
                self.episode_actions.append(np.zeros(1, dtype=np.float32))
            self.episode_rewards.append(reward)
            
        self.current_obs = obs
        
        # If episode is done, check if it's worth recording
        if done:
            episode_reward = sum(self.episode_rewards)
            self.episode_rewards_history.append(episode_reward)
            
            # Only consider recording if episode reached minimum floor
            floor_reached = max([info.get("floor", 0) for info in self.episode_infos], default=0)
            
            if floor_reached >= self.min_floor:
                # Calculate reward percentile
                if len(self.episode_rewards_history) >= 10:
                    reward_threshold = np.percentile(
                        self.episode_rewards_history, self.min_reward_percentile
                    )
                    
                    # Record if reward exceeds threshold
                    if episode_reward >= reward_threshold:
                        self._record_episode()
                        self.recorded_episodes += 1
                        
                        if self.verbose > 0:
                            print(f"Recorded demonstration episode with reward {episode_reward:.2f}")
            
            # Reset tracking
            self.current_obs = None
            self._on_rollout_start()
            
        return True
    
    def _record_episode(self) -> None:
        """Add current episode to demonstration buffer"""
        if not self.episode_obs:
            return
            
        # Verify episode data integrity
        if (len(self.episode_obs) != len(self.episode_actions) or
            len(self.episode_obs) != len(self.episode_rewards) or
            len(self.episode_obs) != len(self.episode_next_obs) or
            len(self.episode_obs) != len(self.episode_dones)):
            if self.verbose > 0:
                print("Episode data integrity check failed, not recording")
            return
            
        # Add episode to buffer
        self.demo_buffer.add_episode(
            self.episode_obs,
            self.episode_actions,
            self.episode_rewards,
            self.episode_next_obs,
            self.episode_dones
        )


class ObstacleTowerEvalCallback(EvalCallback):
    """
    Extended evaluation callback with Obstacle Tower-specific metrics.
    """
    def __init__(
        self,
        eval_env: Union[VecEnv, str],
        callback_on_new_best=None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        """Initialize callback with custom metrics"""
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        
    def _on_step(self) -> bool:
        """
        Evaluate agent for a given number of episodes and log additional metrics.
        """
        # Only evaluate at specified frequency
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and evaluation environments
            sync_envs_normalization(self.training_env, self.eval_env)
            
            # Collect episode rewards and additional metrics
            episode_rewards, episode_lengths = [], []
            floors_reached, keys_collected, doors_opened = [], [], []
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done, state = False, None
                episode_reward, episode_length = 0.0, 0
                floor, keys, doors = 0, 0, 0
                
                while not done:
                    action, state = self.model.predict(
                        obs, state=state, deterministic=self.deterministic
                    )
                    obs, reward, done, info = self.eval_env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Extract Obstacle Tower specific metrics
                    if "floor" in info:
                        floor = max(floor, info["floor"])
                    if "keys_collected" in info:
                        keys = max(keys, info["keys_collected"])
                    if "doors_opened" in info:
                        doors = max(doors, info["doors_opened"])
                    
                    if self.render:
                        self.eval_env.render()
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                floors_reached.append(floor)
                keys_collected.append(keys)
                doors_opened.append(doors)
            
            # Compute mean metrics
            mean_reward = np.mean(episode_rewards)
            mean_length = np.mean(episode_lengths)
            mean_floor = np.mean(floors_reached)
            mean_keys = np.mean(keys_collected)
            mean_doors = np.mean(doors_opened)
            
            # Log metrics
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_episode_length", mean_length)
            self.logger.record("eval/mean_floor_reached", mean_floor)
            self.logger.record("eval/mean_keys_collected", mean_keys)
            self.logger.record("eval/mean_doors_opened", mean_doors)
            
            # Check if new best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                
                if self.verbose > 0:
                    print(f"New best model with mean reward: {mean_reward:.2f}")
                    print(f"  Floors: {mean_floor:.2f}, Keys: {mean_keys:.2f}, Doors: {mean_doors:.2f}")
                
                # Save best model
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                
                # Run callback if provided
                if self.callback_on_new_best is not None:
                    self._on_event()
            
            # Always save latest model for recovery
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "latest_model"))
                
            # Reset model if needed
            if self.training_env is not None:
                self.training_env.reset()
                
            # Log episode metrics
            self.logger.dump(self.n_calls)
            
        return True


class SaveCheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints periodically.
    """
    def __init__(
        self,
        save_path: str,
        save_freq: int = 100000,
        save_demo_buffer: bool = True,
        save_rnd_model: bool = True,
        verbose: int = 0
    ):
        """
        Initialize callback.
        
        Args:
            save_path: Directory to save checkpoints
            save_freq: Frequency of checkpoints in timesteps
            save_demo_buffer: Whether to save demonstration buffer
            save_rnd_model: Whether to save RND model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_demo_buffer = save_demo_buffer
        self.save_rnd_model = save_rnd_model
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        """
        Save checkpoint if it's time.
        """
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.save_path, f"checkpoint_{self.n_calls}"
            )
            
            # Check if we have a DemoSAC model with custom save method
            if hasattr(self.model, "demo_buffer") and hasattr(self.model, "rnd_model"):
                # Use DemoSAC's save method
                include_demonstrations = self.save_demo_buffer and self.model.demo_buffer is not None
                self.model.save(checkpoint_path, include_demonstrations=include_demonstrations)
                
                if self.verbose > 0:
                    print(f"Saved checkpoint to {checkpoint_path}")
            else:
                # Fallback to standard save method
                self.model.save(checkpoint_path)
                
                if self.verbose > 0:
                    print(f"Saved checkpoint to {checkpoint_path}")
                    
        return True