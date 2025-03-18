"""
Demonstration-augmented Soft Actor-Critic (SAC) implementation.
Extends Stable Baselines3 SAC with demonstration learning capabilities.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.sac.policies import SACPolicy

from src.sac_utils .demo_buffer import DemonstrationBuffer


class DemoSAC(SAC):
    """
    Soft Actor-Critic (SAC) with demonstration learning capabilities.
    
    This implementation extends the stable-baselines3 SAC implementation by:
    1. Adding demonstration buffer integration
    2. Adding behavior cloning loss for learning from demonstrations
    3. Supporting a pretrain phase that learns from demonstrations before RL training
    4. Adding integration with intrinsic rewards from RND
    """
    
    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 500000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        # Demonstration learning parameters
        demo_buffer: Optional[DemonstrationBuffer] = None,
        demo_batch_size: int = 64,
        bc_loss_coef: float = 0.5,
        pretrain_steps: int = 0,
        rnd_model = None,
        intrinsic_reward_coef: float = 0.1,
        **kwargs  # Accept extra kwargs and ignore them
    ):
        """
        Initialize DemoSAC.

        Args:
            All SAC parameters from stable-baselines3, plus:
            demo_buffer: Buffer containing demonstration data
            demo_batch_size: Batch size for demonstration sampling
            bc_loss_coef: Weight of behavior cloning loss
            pretrain_steps: Number of pretraining steps on demonstrations
            rnd_model: Random Network Distillation model for intrinsic rewards
            intrinsic_reward_coef: Weight for intrinsic rewards
        """
        # Filter out kwargs that might not be supported by the installed version
        filtered_kwargs = {}
        # You can add specific kwargs here if needed
        
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            ent_coef=ent_coef,
            target_update_interval=target_update_interval,
            target_entropy=target_entropy,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            **filtered_kwargs
        )
        
        # Demonstration learning attributes
        self.demo_buffer = demo_buffer
        self.use_demonstrations = demo_buffer is not None and not demo_buffer.is_empty()
        self.demo_batch_size = demo_batch_size
        self.bc_loss_coef = bc_loss_coef
        self.pretrain_steps = pretrain_steps
        self.pretrain_completed = False
        
        # RND model for intrinsic rewards
        self.rnd_model = rnd_model
        self.use_rnd = rnd_model is not None
        self.intrinsic_reward_coef = intrinsic_reward_coef
        
        # Set up logging dictionary for additional metrics
        self.custom_logger = {
            "bc_loss": [],
            "pretrain_actor_loss": [],
            "pretrain_critic_loss": [],
            "pretrain_ent_coef_loss": [],
            "intrinsic_rewards": [],
        }
    
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer and add intrinsic rewards if applicable.
        
        Args:
            replay_buffer: Replay buffer to store transition
            buffer_action: Action to store in the buffer
            new_obs: Next observation
            reward: External reward
            done: Done flag
            infos: Additional information
        """
        # Calculate intrinsic rewards if RND model is available
        if self.use_rnd:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(new_obs).to(self.device)
            if len(obs_tensor.shape) == 3:  # Add batch dimension if needed
                obs_tensor = obs_tensor.unsqueeze(0)
                
            # Calculate intrinsic reward
            with torch.no_grad():
                intrinsic_reward = self.rnd_model.calculate_intrinsic_reward(obs_tensor)
                intrinsic_reward = intrinsic_reward.cpu().numpy()
                
            # Log intrinsic rewards
            self.custom_logger["intrinsic_rewards"].append(float(np.mean(intrinsic_reward)))
                
            # Add intrinsic reward to external reward
            reward = reward + self.intrinsic_reward_coef * intrinsic_reward
        
        # Store transition in replay buffer
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)
    
    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        Train the policy network (actor) and value network (critic).
        Includes both standard SAC training and demonstration-based learning.
        
        Args:
            gradient_steps: Number of gradient steps to take
            batch_size: Batch size for sampling from replay buffer
        """
        # Run pretraining if not yet completed
        if self.use_demonstrations and not self.pretrain_completed and self.pretrain_steps > 0:
            self._pretrain_from_demonstrations()
            self.pretrain_completed = True
            
        # Regular SAC training 
        sac_losses = super().train(gradient_steps, batch_size)
        
        # Additional training on demonstrations if available
        if self.use_demonstrations and self.demo_buffer is not None:
            demo_losses = self._train_on_demonstrations()
            
            # Combine losses for logging
            if isinstance(sac_losses, dict):
                sac_losses.update(demo_losses)
            
        return sac_losses
    
    def _train_on_demonstrations(self) -> Dict[str, float]:
        """
        Train on demonstration data by adding behavior cloning loss.
        
        Returns:
            dict: Dictionary with training metrics
        """
        if not self.use_demonstrations or self.demo_buffer is None:
            return {}

        # Skip if not enough demonstrations
        if self.demo_buffer.total_transitions < self.demo_batch_size:
            return {"bc_loss": 0.0}

        # Sample batch from demonstrations
        demo_batch = self.demo_buffer.sample(self.demo_batch_size)
        
        # Convert to policy input format if needed
        demo_obs = demo_batch["observations"]
        demo_actions = demo_batch["actions"]
        
        # Get current policy's action predictions
        with torch.no_grad():
            mean_actions, log_std, _ = self.actor.get_action_dist_params(demo_obs)
            
        # Calculate behavior cloning loss (MSE)
        bc_loss = F.mse_loss(mean_actions, demo_actions)
        
        # Update actor using behavior cloning loss
        self.actor.optimizer.zero_grad()
        (self.bc_loss_coef * bc_loss).backward()
        self.actor.optimizer.step()
        
        # Store loss for logging
        bc_loss_value = bc_loss.item()
        self.custom_logger["bc_loss"].append(bc_loss_value)
        
        return {"bc_loss": bc_loss_value}
    
    def _pretrain_from_demonstrations(self) -> None:
        """Pretrain actor and critic networks using demonstration data"""
        if not self.use_demonstrations or self.demo_buffer is None:
            return
            
        if self.demo_buffer.total_transitions < self.demo_batch_size:
            print(f"Not enough demonstrations for pretraining: {self.demo_buffer.total_transitions} < {self.demo_batch_size}")
            return
            
        print(f"Pretraining for {self.pretrain_steps} steps on demonstration data...")
        
        import numpy as np
        
        # Pretraining loop
        for step in range(1, self.pretrain_steps + 1):
            # Sample demonstrations
            demo_batch = self.demo_buffer.sample(self.demo_batch_size)
            
            # Get tensors
            demo_obs = demo_batch["observations"]
            demo_actions = demo_batch["actions"]
            demo_rewards = demo_batch["rewards"]
            demo_next_obs = demo_batch["next_observations"]
            demo_dones = demo_batch["dones"]
            
            # Handle channel format mismatch
            if isinstance(demo_obs, np.ndarray):
                if len(demo_obs.shape) == 4 and demo_obs.shape[-1] == 3:  # NHWC format
                    # Convert to NCHW format with 4 channels
                    # Stack the same frame 4 times to match expected input
                    demo_obs = np.transpose(demo_obs, (0, 3, 1, 2))  # NHWC -> NCHW
                    demo_next_obs = np.transpose(demo_next_obs, (0, 3, 1, 2))
                    
                    # Repeat first channel to get 4 channels (assumes model expects 4 channels)
                    # This is a simple approach - ideally you'd properly stack sequential frames
                    if demo_obs.shape[1] == 3:  # If RGB (3 channels)
                        demo_obs = np.concatenate([demo_obs[:,:1], demo_obs], axis=1)[:,:4]  # Use first channel + RGB to get 4
                        demo_next_obs = np.concatenate([demo_next_obs[:,:1], demo_next_obs], axis=1)[:,:4]
            elif isinstance(demo_obs, torch.Tensor):
                if len(demo_obs.shape) == 4 and demo_obs.shape[-1] == 3:  # NHWC format
                    demo_obs = demo_obs.permute(0, 3, 1, 2)  # NHWC -> NCHW
                    demo_next_obs = demo_next_obs.permute(0, 3, 1, 2)
                    
                    # Add an extra channel
                    if demo_obs.shape[1] == 3:
                        demo_obs = torch.cat([demo_obs[:,:1], demo_obs], dim=1)[:,:4]
                        demo_next_obs = torch.cat([demo_next_obs[:,:1], demo_next_obs], dim=1)[:,:4]
            
            # Rest of the function remains the same...
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(demo_next_obs)
                current_q_values = self.critic(demo_obs, demo_actions)
                next_q_values = self.critic_target(demo_next_obs, next_actions)
                next_q_values = next_q_values - self.ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = demo_rewards.reshape(-1, 1) + (1 - demo_dones.reshape(-1, 1)) * self.gamma * next_q_values
            
            current_q1, current_q2 = self.critic(demo_obs, demo_actions)
            critic_loss = 0.5 * (F.mse_loss(current_q1, target_q_values) + F.mse_loss(current_q2, target_q_values))
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            actions_pi, log_prob = self.actor.action_log_prob(demo_obs)
            q_values_pi = torch.cat(self.critic(demo_obs, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            
            actor_loss = (self.ent_coef * log_prob - min_qf_pi).mean()
            bc_loss = F.mse_loss(actions_pi, demo_actions)
            total_actor_loss = actor_loss + self.bc_loss_coef * bc_loss
            
            self.actor.optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor.optimizer.step()
            
            if self.ent_coef_optimizer is not None:
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                self.ent_coef = torch.exp(self.log_ent_coef.detach())
            
            if step % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            self.custom_logger["pretrain_actor_loss"].append(actor_loss.item())
            self.custom_logger["pretrain_critic_loss"].append(critic_loss.item())
            if self.ent_coef_optimizer is not None:
                self.custom_logger["pretrain_ent_coef_loss"].append(ent_coef_loss.item())
            
            if step % (self.pretrain_steps // 10) == 0 or step == 1:
                print(f"Pretraining step {step}/{self.pretrain_steps}, "
                    f"actor_loss: {actor_loss.item():.4f}, "
                    f"critic_loss: {critic_loss.item():.4f}, "
                    f"bc_loss: {bc_loss.item():.4f}")
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,  # We'll keep this parameter but not use it
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DemoSAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "DemoSAC":
        """
        Extended learn method with demonstration pretraining and RND updates.
        
        Args:
            Standard SAC learn() parameters
            
        Returns:
            self: The trained model
        """
        # Regular SAC learning - REMOVE the eval_env parameter
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            # Remove these parameters as they're not supported in your version
            # eval_env=eval_env,
            # eval_freq=eval_freq,
            # n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            # eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
    
    def save(self, path: str, include_demonstrations: bool = True) -> None:
        """
        Save model, including demonstration buffer if requested.
        
        Args:
            path: Path to save the model
            include_demonstrations: Whether to save demonstration buffer
        """
        # Save the model
        super().save(path)
        
        # Save additional components
        if include_demonstrations and self.demo_buffer is not None:
            demo_path = f"{path}_demos.pkl"
            self.demo_buffer.save(demo_path)
        
        if self.use_rnd and self.rnd_model is not None:
            rnd_path = f"{path}_rnd.pt"
            torch.save(self.rnd_model.state_dict(), rnd_path)
    
    @classmethod
    def load(
        cls,
        path: str,
        env: Optional[GymEnv] = None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        load_demonstrations: bool = True,
        load_rnd: bool = True,
        **kwargs,
    ) -> "DemoSAC":
        """
        Load model with demonstration buffer and RND model.
        
        Args:
            path: Path to the saved model
            env: Environment to use
            device: Device to use
            custom_objects: Custom objects to load
            load_demonstrations: Whether to load demonstration buffer
            load_rnd: Whether to load RND model
            
        Returns:
            DemoSAC: Loaded model
        """
        # Load the model
        model = super().load(path, env, device, custom_objects, **kwargs)
        
        # Load demonstrations if requested
        if load_demonstrations:
            demo_path = f"{path}_demos.pkl"
            if os.path.exists(demo_path):
                from src.sac_utils .demo_buffer import DemonstrationBuffer
                demo_buffer = DemonstrationBuffer(device=device)
                demo_buffer.load(demo_path)
                model.demo_buffer = demo_buffer
                model.use_demonstrations = not demo_buffer.is_empty()
        
        # Load RND model if requested
        if load_rnd:
            rnd_path = f"{path}_rnd.pt"
            if os.path.exists(rnd_path) and "rnd_model" in kwargs:
                rnd_model = kwargs["rnd_model"]
                rnd_model.load_state_dict(torch.load(rnd_path, map_location=device))
                model.rnd_model = rnd_model
                model.use_rnd = True
        
        return model


class RNDUpdateCallback(BaseCallback):
    """
    Callback to handle RND model updates during training.
    """
    
    def __init__(
        self, 
        rnd_model,
        update_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Initialize the callback.
        
        Args:
            rnd_model: RND model to update
            update_freq: Frequency of RND model updates
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.rnd_model = rnd_model
        self.update_freq = update_freq
        self.rnd_losses = []
    
    def _on_step(self) -> bool:
        """
        Check if we should update the RND model based on frequency
        """
        if self.n_calls % self.update_freq == 0 and self.model.rnd_model is not None:
            # Get a batch of observations from replay buffer
            replay_data = self.model.replay_buffer.sample(self.model.batch_size)
            obs = replay_data.observations
            
            # Convert observation to tensor and ensure it's on the correct device
            if isinstance(obs, torch.Tensor):
                # If already a tensor, just ensure it's on the right device
                obs_tensor = obs.to(self.model.device)
            else:
                # If not a tensor, convert it to one
                obs_tensor = torch.FloatTensor(obs).to(self.model.device)
            
            # Update RND model
            update_info = self.model.rnd_model.update(obs_tensor)
            
            # Log update metrics
            if self.verbose > 0 and hasattr(self.model, "logger"):
                for k, v in update_info.items():
                    self.model.logger.record(f"rnd/{k}", v)
                    
        return True