"""
Callbacks for SAC training with Obstacle Tower.

This file contains custom callbacks for training with stable-baselines3:
- EarlyStoppingCallback: Stops training early if evaluation performance plateaus
- TrainingProgressCallback: Monitors and logs detailed training progress
- TensorboardCallback: Logs additional metrics to TensorBoard
- DemonstrationRecorderCallback: Records high-quality demonstrations
- ObstacleTowerEvalCallback: Evaluation for Obstacle Tower
- SaveCheckpointCallback: Saves model checkpoints
"""
import os
import time
import pickle
import numpy as np
from typing import Optional, Dict, Union, List, Callable, Any

import gym
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log additional metrics every step
        if hasattr(self.model, "demo_buffer") and self.model.demo_buffer is not None:
            # Log demo buffer size
            demo_size = self.model.demo_buffer.total_transitions
            self.logger.record("demo_buffer/size", demo_size)
            
            # Log demo loss if available
            if hasattr(self.model, "demo_loss") and self.model.demo_loss is not None:
                self.logger.record("demo/loss", self.model.demo_loss.item())
                
        # Log RND metrics if available
        if hasattr(self.model, "rnd_model") and self.model.rnd_model is not None:
            if hasattr(self.model.rnd_model, "last_intrinsic_reward"):
                self.logger.record("rnd/intrinsic_reward", np.mean(self.model.rnd_model.last_intrinsic_reward))
            if hasattr(self.model.rnd_model, "last_prediction_error"):
                self.logger.record("rnd/prediction_error", self.model.rnd_model.last_prediction_error.item())
        
        # Log more detailed learning stats
        if hasattr(self.model, "actor_loss") and self.model.actor_loss is not None:
            self.logger.record("losses/actor_loss", self.model.actor_loss.item())
        if hasattr(self.model, "critic_loss") and self.model.critic_loss is not None:
            self.logger.record("losses/critic_loss", self.model.critic_loss.item())
        if hasattr(self.model, "ent_coef_loss") and self.model.ent_coef_loss is not None:
            self.logger.record("losses/ent_coef_loss", self.model.ent_coef_loss.item())
            
        return True


class DemonstrationRecorderCallback(BaseCallback):
    """
    Callback for recording demonstrations from the agent.
    
    Records transitions to the demonstration buffer when the agent
    achieves high rewards or reaches later floors in the tower.
    """
    def __init__(
        self, 
        demo_buffer, 
        min_reward_percentile=90.0, 
        min_floor=1,
        verbose=0
    ):
        super(DemonstrationRecorderCallback, self).__init__(verbose)
        self.demo_buffer = demo_buffer
        self.min_reward_percentile = min_reward_percentile
        self.min_floor = min_floor
        self.episode_reward = 0
        self.highest_floor = 0
        self.current_obs = None
        self.current_info = None
        self.episode_transitions = []
        self.episode_rewards = []
        self.reward_threshold = 0
    
    def _on_training_start(self) -> None:
        # Initialize the reward threshold
        self.reward_threshold = 0
    
    def _on_step(self) -> bool:
        # Get current info and obs
        if self.locals.get("new_obs") is not None:
            self.current_obs = self.locals["new_obs"]
        
        info = self.locals.get("infos")[0] if self.locals.get("infos") else None
        if info is not None:
            self.current_info = info
            # Track highest floor reached in this episode
            if "floor" in info and info["floor"] > self.highest_floor:
                self.highest_floor = info["floor"]
        
        # Update episode reward
        if self.locals.get("rewards") is not None:
            reward = self.locals["rewards"][0]
            self.episode_reward += reward
        
        # Record transition
        if self.locals.get("dones") is not None and not self.locals["dones"][0]:
            # Not done, record the transition
            if (self.locals.get("obs") is not None and 
                self.locals.get("actions") is not None and 
                self.locals.get("rewards") is not None and 
                self.locals.get("new_obs") is not None):
                
                transition = {
                    "obs": self.locals["obs"][0],
                    "action": self.locals["actions"][0],
                    "reward": self.locals["rewards"][0],
                    "next_obs": self.locals["new_obs"][0],
                    "done": False,
                    "info": self.current_info if self.current_info else {}
                }
                self.episode_transitions.append(transition)
        
        # End of episode
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            # Record final transition
            if (self.locals.get("obs") is not None and 
                self.locals.get("actions") is not None and 
                self.locals.get("rewards") is not None and 
                self.locals.get("new_obs") is not None):
                
                transition = {
                    "obs": self.locals["obs"][0],
                    "action": self.locals["actions"][0],
                    "reward": self.locals["rewards"][0],
                    "next_obs": self.locals["new_obs"][0],
                    "done": True,
                    "info": self.current_info if self.current_info else {}
                }
                self.episode_transitions.append(transition)
            
            # Update reward statistics
            self.episode_rewards.append(self.episode_reward)
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
            
            # Update reward threshold
            if len(self.episode_rewards) >= 10:
                self.reward_threshold = np.percentile(
                    self.episode_rewards, 
                    self.min_reward_percentile
                )
            
            # Check if this episode qualifies for demonstration recording
            should_record = False
            if self.highest_floor >= self.min_floor:
                should_record = True
                if self.verbose > 0:
                    print(f"Recording demonstration - reached floor {self.highest_floor}")
            elif self.episode_reward >= self.reward_threshold and self.reward_threshold > 0:
                should_record = True
                if self.verbose > 0:
                    print(f"Recording demonstration - reward {self.episode_reward:.2f} > threshold {self.reward_threshold:.2f}")
            
            # Add transitions to demo buffer if qualified
            if should_record and len(self.episode_transitions) > 0:
                for trans in self.episode_transitions:
                    self.demo_buffer.add(**trans)
                
                if self.verbose > 0:
                    print(f"Added {len(self.episode_transitions)} transitions to demo buffer")
                    print(f"Demo buffer now contains {self.demo_buffer.total_transitions} transitions")
            
            # Reset for next episode
            self.episode_reward = 0
            self.highest_floor = 0
            self.episode_transitions = []
            self.current_obs = None
            self.current_info = None
        
        return True


class ObstacleTowerEvalCallback(EvalCallback):
    """
    Evaluation callback for Obstacle Tower.
    
    Extends EvalCallback to handle Obstacle Tower-specific metrics like
    highest floor reached.
    """
    def __init__(
        self,
        eval_env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        eval_freq=10000,
        log_path=None,
        best_model_save_path=None,
        deterministic=True,
        render=False,
        verbose=1,
        warn=True,
    ):
        super(ObstacleTowerEvalCallback, self).__init__(
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
        self.highest_floors = []
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval environments if needed
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except Exception:
                    if self.verbose > 0:
                        print("Warning: environment synchronization failed")
            
            # Reset highest floors list
            self.highest_floors = []
            
            # Run evaluation
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success,
            )
            
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            
            # Log stats
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                if self.highest_floors:
                    mean_floor = np.mean(self.highest_floors)
                    max_floor = np.max(self.highest_floors)
                    print(f"Highest floor: mean={mean_floor:.1f}, max={max_floor}")
            
            # Log to tensorboard
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            
            if self.highest_floors:
                self.logger.record("eval/mean_floor", np.mean(self.highest_floors))
                self.logger.record("eval/max_floor", np.max(self.highest_floors))
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f} (previous: {self.best_mean_reward:.2f})")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                
                # Save demo buffer if available
                if hasattr(self.model, "demo_buffer") and self.model.demo_buffer is not None:
                    demo_path = os.path.join(self.best_model_save_path, "best_demonstrations.pkl")
                    self.model.demo_buffer.save(demo_path)
                    if self.verbose > 0:
                        print(f"Saved best demonstration buffer to {demo_path}")
                
                # Save RND model if available
                if hasattr(self.model, "rnd_model") and self.model.rnd_model is not None:
                    rnd_path = os.path.join(self.best_model_save_path, "best_rnd_model.pt")
                    torch.save(self.model.rnd_model.state_dict(), rnd_path)
                    if self.verbose > 0:
                        print(f"Saved best RND model to {rnd_path}")
            
        return True
    
    def _log_success(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback to log episode success and floor information.
        """
        info = locals_.get("info")
        if info and isinstance(info, dict) and "floor" in info:
            self.highest_floors.append(info["floor"])


class SaveCheckpointCallback(BaseCallback):
    """
    Callback to save model checkpoints at regular intervals.
    """
    def __init__(
        self, 
        save_path, 
        save_freq=10000, 
        save_demo_buffer=True, 
        save_rnd_model=True,
        verbose=0
    ):
        super(SaveCheckpointCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_demo_buffer = save_demo_buffer
        self.save_rnd_model = save_rnd_model
        
        # Create directories
        os.makedirs(self.save_path, exist_ok=True)
        self.checkpoints_path = os.path.join(self.save_path, "checkpoints")
        os.makedirs(self.checkpoints_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoints_path, 
                f"checkpoint_{self.num_timesteps}"
            )
            
            # Save model
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saved model checkpoint to {checkpoint_path}")
            
            # Save demo buffer if available
            if (self.save_demo_buffer and 
                hasattr(self.model, "demo_buffer") and 
                self.model.demo_buffer is not None):
                
                demo_path = f"{checkpoint_path}_demos.pkl"
                self.model.demo_buffer.save(demo_path)
                if self.verbose > 0:
                    print(f"Saved demo buffer to {demo_path}")
            
            # Save RND model if available
            if (self.save_rnd_model and 
                hasattr(self.model, "rnd_model") and 
                self.model.rnd_model is not None):
                
                rnd_path = f"{checkpoint_path}_rnd.pt"
                torch.save(self.model.rnd_model.state_dict(), rnd_path)
                if self.verbose > 0:
                    print(f"Saved RND model to {rnd_path}")
        
        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping when evaluation reward doesn't improve.
    """
    def __init__(
        self, 
        eval_env, 
        eval_freq=10000, 
        n_eval_episodes=5,
        patience=5, 
        min_improvement=0.1, 
        verbose=0
    ):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.patience_counter = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
            
        # Evaluate policy
        episode_rewards = []
        episode_floors = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            highest_floor = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                next_obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward
                
                # Track highest floor
                if isinstance(info, dict) and "floor" in info and info["floor"] > highest_floor:
                    highest_floor = info["floor"]
                
                obs = next_obs
            
            episode_rewards.append(episode_reward)
            episode_floors.append(highest_floor)
            
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        mean_floor = np.mean(episode_floors)
        max_floor = np.max(episode_floors)
        
        # Log metrics
        self.logger.record("early_stopping/mean_reward", mean_reward)
        self.logger.record("early_stopping/mean_floor", mean_floor)
        self.logger.record("early_stopping/max_floor", max_floor)
        
        # Check if reward improved
        if mean_reward - self.best_mean_reward > self.min_improvement:
            self.best_mean_reward = mean_reward
            self.patience_counter = 0
            if self.verbose > 0:
                print(f"New best mean reward: {mean_reward:.2f}")
                print(f"Mean floor: {mean_floor:.1f}, Max floor: {max_floor}")
        else:
            self.patience_counter += 1
            if self.verbose > 0:
                print(f"No improvement for {self.patience_counter} evaluations")
                print(f"Current reward: {mean_reward:.2f}, Best: {self.best_mean_reward:.2f}")
                print(f"Mean floor: {mean_floor:.1f}, Max floor: {max_floor}")
            
        # Stop training if patience exceeded
        if self.patience_counter >= self.patience:
            if self.verbose > 0:
                print(f"Early stopping triggered after {self.num_timesteps} steps")
                print(f"Best mean reward: {self.best_mean_reward:.2f}")
            return False
            
        return True


class TrainingProgressCallback(BaseCallback):
    """
    Callback to monitor training progress and provide detailed updates.
    """
    def __init__(self, log_interval=1000, verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_floors = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_floor = 0
        
    def _on_step(self) -> bool:
        # Update current episode stats
        if self.locals.get("rewards") is not None:
            self.current_episode_reward += self.locals["rewards"][0]
            self.current_episode_length += 1
        
        # Track floor information
        info = self.locals.get("infos")[0] if self.locals.get("infos") else None
        if info is not None and isinstance(info, dict) and "floor" in info:
            if info["floor"] > self.current_episode_floor:
                self.current_episode_floor = info["floor"]
        
        # Check if episode ended
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_floors.append(self.current_episode_floor)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_floor = 0
            
            # Limit stored episodes to prevent memory issues
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
                self.episode_lengths.pop(0)
                self.episode_floors.pop(0)
        
        # Log stats at regular intervals
        if self.num_timesteps % self.log_interval == 0:
            elapsed_time = time.time() - self.start_time
            fps = int(self.num_timesteps / elapsed_time)
            
            # Calculate episode statistics
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
                mean_floor = np.mean(self.episode_floors[-10:]) if len(self.episode_floors) >= 10 else np.mean(self.episode_floors)
                max_floor = np.max(self.episode_floors) if self.episode_floors else 0
            else:
                mean_reward = 0
                mean_length = 0
                mean_floor = 0
                max_floor = 0
            
            # Log to console
            if self.verbose > 0:
                print(f"\nTimestep: {self.num_timesteps}")
                print(f"FPS: {fps}")
                print(f"Mean reward (last 10 episodes): {mean_reward:.2f}")
                print(f"Mean episode length (last 10 episodes): {mean_length:.1f}")
                print(f"Mean floor (last 10 episodes): {mean_floor:.1f}")
                print(f"Max floor reached: {max_floor}")
                print(f"Time elapsed: {int(elapsed_time // 3600)}h {int((elapsed_time % 3600) // 60)}m {int(elapsed_time % 60)}s")
                
                # Print buffer statistics if available
                if hasattr(self.model, "replay_buffer"):
                    buffer_size = len(self.model.replay_buffer)
                    print(f"Replay buffer size: {buffer_size}")
                
                if hasattr(self.model, "demo_buffer") and self.model.demo_buffer is not None:
                    demo_size = self.model.demo_buffer.total_transitions
                    print(f"Demo buffer size: {demo_size}")
            
            # Log to TensorBoard
            self.logger.record("time/fps", fps)
            self.logger.record("rollout/mean_reward", mean_reward)
            self.logger.record("rollout/mean_episode_length", mean_length)
            self.logger.record("rollout/mean_floor", mean_floor)
            self.logger.record("rollout/max_floor", max_floor)
            
        return True