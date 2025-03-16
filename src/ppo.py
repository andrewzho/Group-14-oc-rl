import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import torch.optim as optim
import math
import os

# Add global debug flag that respects verbosity setting
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'

def debug_print(*args, **kwargs):
    """Print only when debug mode is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)

class PPO:
    def __init__(
        self,
        model,
        lr=2.5e-4,
        clip_eps=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        epochs=4,
        batch_size=256,  # Increased batch size for more stable updates
        vf_coef=0.5,
        ent_reg=0.005,  # FIXED: Reduced from 0.01 to encourage less randomness initially
        max_grad_norm=0.5,
        target_kl=0.05,
        lr_scheduler='linear',  # Change default to linear like SB3
        adaptive_entropy=True,  # Use adaptive entropy for better exploration
        min_entropy=0.01,     # Minimum entropy level to maintain exploration
        entropy_decay_factor=0.9999,  # Slow decay for entropy
        update_adv_batch_norm=True,   # Added parameter for advantage batch normalization
        entropy_boost_threshold=0.001, # Threshold for boosting entropy when progress stalls
        lr_reset_interval=50,          # Reset optimizer occasionally to escape plateaus
        use_icm=False,                 # Whether to use Intrinsic Curiosity Module
        icm_lr=1e-4,                   # Learning rate for ICM module
        icm_reward_scale=0.01,         # Scale factor for intrinsic rewards from ICM
        icm_forward_weight=0.2,   # Weight for ICM forward model loss
        icm_inverse_weight=0.8,   # Weight for ICM inverse model loss
        icm_feature_dim=256,           # Feature dimension for ICM module
        use_recurrent=False,           # Whether to use recurrent policy (LSTM)
        recurrent_seq_len=8,           # Sequence length for recurrent training
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Add device parameter with default value
    ):
        """
        Proximal Policy Optimization algorithm with adaptive entropy and enhanced exploration.
        
        Args:
            model: Policy network
            lr: Learning rate
            clip_eps: PPO clipping parameter
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            epochs: Number of epochs to train on each batch of data
            batch_size: Mini-batch size for training
            vf_coef: Value function loss coefficient
            ent_reg: Entropy regularization coefficient
            max_grad_norm: Maximum gradient norm for gradient clipping
            target_kl: Target KL divergence threshold
            lr_scheduler: Learning rate scheduler type (None, 'linear', 'cosine')
            adaptive_entropy: Whether to use adaptive entropy regularization
            min_entropy: Minimum entropy regularization coefficient
            entropy_decay_factor: Factor to decay entropy by each update
            update_adv_batch_norm: Whether to normalize advantages per batch
            entropy_boost_threshold: Policy loss threshold for boosting entropy
            lr_reset_interval: Reset optimizer every N updates to escape plateaus
            use_icm: Whether to use Intrinsic Curiosity Module
            icm_lr: Learning rate for ICM module
            icm_reward_scale: Scale factor for intrinsic rewards from ICM
            icm_forward_weight: Weight for ICM forward model loss
            icm_inverse_weight: Weight for ICM inverse model loss
            icm_feature_dim: Feature dimension for ICM module
            use_recurrent: Whether to use recurrent policy with LSTM
            recurrent_seq_len: Sequence length for recurrent training
            device: Device to use for tensor operations (e.g., 'cpu', 'cuda')
        """
        self.model = model
        self.lr = lr
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_reg = ent_reg
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.lr_scheduler = lr_scheduler
        self.adaptive_entropy = adaptive_entropy
        self.min_entropy = min_entropy
        self.entropy_decay_factor = entropy_decay_factor
        self.update_adv_batch_norm = update_adv_batch_norm
        self.entropy_boost_threshold = entropy_boost_threshold
        self.lr_reset_interval = lr_reset_interval
        self.device = device  # Store device
        
        # Set up ICM
        self.use_icm = use_icm
        self.icm_reward_scale = icm_reward_scale
        self.icm_lr = icm_lr
        self.icm_forward_weight = icm_forward_weight
        self.icm_inverse_weight = icm_inverse_weight
        self.icm_feature_dim = icm_feature_dim
        self.icm = None

        # Recurrent policy settings
        self.use_recurrent = use_recurrent
        self.recurrent_seq_len = recurrent_seq_len
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Learning rate scheduler
        if self.lr_scheduler == 'linear':
            # Linear scheduler that decreases LR over time - similar to SB3
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: max(1.0 - step / 10000, 0.1)  # Decrease to 10% of initial LR
            )
        elif self.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10000, eta_min=lr * 0.1
            )
        else:
            self.scheduler = None
            
        # Training metrics
        self.metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'approx_kl': [],
            'explained_variance': [],
            'learning_rate': [],
            'clip_fraction': [],
            'stop_iteration': []
        }
        
        # Performance tracking for adaptive exploration
        self.recent_rewards = []
        self.max_reward = -float('inf')
        self.stagnation_counter = 0
        self.updates_since_lr_reset = 0
        
        # For recurrent policies, we need to track LSTM states during rollouts
        if self.use_recurrent:
            self.lstm_states = None
            self.reset_lstm_state()
            
    def reset_lstm_state(self, batch_size=1):
        """Reset LSTM hidden states to zeros."""
        if self.use_recurrent and hasattr(self.model, 'init_lstm_state'):
            self.lstm_states = self.model.init_lstm_state(batch_size, device=self.device)
        else:
            self.lstm_states = None
        
    def compute_gae(self, rewards: List[float], values: List[float], next_value: float, dones: List[bool]) -> List[float]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            next_value: Value estimate for the state after the last one
            dones: List of done flags
            
        Returns:
            List of advantage estimates
        """
        advantages = []
        gae = 0
        
        # Compute advantages in reverse order
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                # For the last state, use next_value
                next_val = next_value
            else:
                # For other states, use the next value in the list
                next_val = values[i + 1]
                
            # If the episode ended, there is no next value
            if dones[i]:
                next_val = 0
                
            # Calculate TD error: r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i]) - values[i]
            
            # Recursive advantage computation
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            
            # FIXED: Check for NaN/Inf in advantages
            if np.isnan(gae) or np.isinf(gae):
                debug_print(f"Warning: NaN or Inf in GAE calculation. delta={delta}")
                gae = 0.0  # Reset to zero if NaN/Inf
            
            # Guard against extreme values which can cause training instability
            gae = max(min(gae, 10.0), -10.0)
            
            advantages.insert(0, gae)
            
        return advantages
    
    def calculate_explained_variance(self, y_pred, y_true):
        """Calculate explained variance."""
        if len(y_pred) == 0 or len(y_true) == 0:
            return 0
        var_y = np.var(y_true)
        if var_y == 0:
            return 0  # Avoid division by zero
        return 1 - np.var(y_true - y_pred) / var_y
        
    def update_sequence_data(self, states, actions, rewards, values, log_probs, dones, lstm_states=None):
        """
        Prepare sequence data for recurrent policy training.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            values: List of value estimates
            log_probs: List of log probabilities
            dones: List of done flags
            lstm_states: Initial LSTM states
            
        Returns:
            Processed data ready for PPO updates with sequences
        """
        if not self.use_recurrent:
            # If not using recurrent policy, just return the data as is
            return states, actions, rewards, values, log_probs, dones
            
        # Convert to numpy arrays for easier manipulation
        states_array = np.array(states)
        actions_array = np.array(actions)
        rewards_array = np.array(rewards)
        values_array = np.array(values)
        log_probs_array = np.array(log_probs)
        dones_array = np.array(dones)
        
        # Check if we're dealing with multi-dimensional log_probs
        log_probs_shape = log_probs_array.shape
        is_multidim_log_probs = len(log_probs_shape) > 1 and log_probs_shape[-1] > 1
        
        # Flatten multi-dimensional log_probs if necessary
        if is_multidim_log_probs:
            debug_print(f"Flattening multi-dimensional log_probs with shape {log_probs_shape}")
            original_log_probs_shape = log_probs_shape[1:]  # Save dimensions besides batch
            # Take the first column if multi-dimensional
            log_probs_array = log_probs_array.reshape(log_probs_shape[0], -1)[:, 0]
            
        # Check if we're dealing with multi-dimensional values
        values_shape = values_array.shape
        is_multidim_values = len(values_shape) > 1 and values_shape[-1] > 1
        
        # Flatten multi-dimensional values if necessary
        if is_multidim_values:
            debug_print(f"Flattening multi-dimensional values with shape {values_shape}")
            original_values_shape = values_shape[1:]  # Save dimensions besides batch
            # Take the first column if multi-dimensional
            values_array = values_array.reshape(values_shape[0], -1)[:, 0]
        
        # Check action dimensions - flatten if it's multi-dimensional
        action_shape = actions_array.shape
        is_multidim_action = len(action_shape) > 1
        
        # Handle multi-dimensional actions by flattening to 1D
        if is_multidim_action and len(action_shape) > 1:
            # Flatten actions to 1D for sequence processing
            # We'll remember the original shape to restore it later
            original_action_shape = action_shape[1:]  # Save dimensions besides batch
            actions_array_flat = actions_array.reshape(action_shape[0], -1)
            # If action is still a 2D array, keep only the first dimension for sequence processing
            if len(actions_array_flat.shape) > 1 and actions_array_flat.shape[1] == 1:
                actions_array = actions_array_flat.squeeze(1)
            else:
                # Just use the first element if we can't cleanly reshape
                actions_array = actions_array_flat[:, 0]
                
        # Apply debug prints to diagnose shapes
        debug_print(f"Shapes after flattening:")
        debug_print(f"  states: {states_array.shape}")
        debug_print(f"  actions: {actions_array.shape}")
        debug_print(f"  rewards: {rewards_array.shape}")
        debug_print(f"  values: {values_array.shape}")
        debug_print(f"  log_probs: {log_probs_array.shape}")
        debug_print(f"  dones: {dones_array.shape}")
        
        # Find episode boundaries
        episode_ends = np.where(dones_array)[0]
        if len(episode_ends) == 0 or episode_ends[-1] != len(dones_array) - 1:
            # Add the last index if the last episode didn't end with done
            episode_ends = np.append(episode_ends, len(dones_array) - 1)
            
        # Calculate episode starts
        episode_starts = np.zeros_like(episode_ends)
        episode_starts[1:] = episode_ends[:-1] + 1
        
        # Calculate length of each episode
        episode_lengths = episode_ends - episode_starts + 1
        
        # Instead of creating irregular sequences, we'll create fixed-length sequences
        # with padding and a mask to track the actual sequence lengths
        max_seq_len = min(self.recurrent_seq_len, max(episode_lengths))
        
        # Count how many sequences we'll need
        total_sequences = 0
        for length in episode_lengths:
            # Overlapping sequences with 50% overlap for better training
            stride = max_seq_len // 2
            total_sequences += max(1, (length - 1) // stride + 1)
            
        # Pre-allocate arrays for the fixed-length sequences
        padded_states = np.zeros((total_sequences, max_seq_len) + states_array[0].shape, dtype=np.float32)
        padded_actions = np.zeros((total_sequences, max_seq_len), dtype=np.int64)
        padded_rewards = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        padded_values = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        padded_log_probs = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        padded_dones = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        # Mask to track valid timesteps (1=valid, 0=padding)
        padded_mask = np.zeros((total_sequences, max_seq_len), dtype=np.float32)
        
        # Fill the padded arrays
        seq_idx = 0
        for ep_idx in range(len(episode_starts)):
            start_idx = episode_starts[ep_idx]
            end_idx = episode_ends[ep_idx] + 1
            ep_length = end_idx - start_idx
            
            # Create overlapping sequences with stride
            stride = max_seq_len // 2
            for seq_start in range(start_idx, end_idx, stride):
                seq_end = min(seq_start + max_seq_len, end_idx)
                actual_len = seq_end - seq_start
                
                # Skip very short sequences
                if actual_len < 2:
                    continue
                
                # Copy data to padded arrays
                padded_states[seq_idx, :actual_len] = states_array[seq_start:seq_end]
                
                # Handle action copying safely depending on dimensions
                # Only copy the flattened/selected actions
                padded_actions[seq_idx, :actual_len] = actions_array[seq_start:seq_end]
                
                padded_rewards[seq_idx, :actual_len] = rewards_array[seq_start:seq_end]
                padded_values[seq_idx, :actual_len] = values_array[seq_start:seq_end]
                padded_log_probs[seq_idx, :actual_len] = log_probs_array[seq_start:seq_end]
                padded_dones[seq_idx, :actual_len] = dones_array[seq_start:seq_end]
                # Set mask for valid timesteps
                padded_mask[seq_idx, :actual_len] = 1.0
                
                seq_idx += 1
                
        # Trim any unused pre-allocated sequences
        if seq_idx < total_sequences:
            padded_states = padded_states[:seq_idx]
            padded_actions = padded_actions[:seq_idx]
            padded_rewards = padded_rewards[:seq_idx]
            padded_values = padded_values[:seq_idx]
            padded_log_probs = padded_log_probs[:seq_idx]
            padded_dones = padded_dones[:seq_idx]
            padded_mask = padded_mask[:seq_idx]
            
        # If we didn't get any valid sequences, fall back to non-recurrent processing
        if seq_idx == 0:
            debug_print("Warning: No valid sequences found, falling back to non-recurrent processing")
            return states, actions, rewards, values, log_probs, dones
            
        # Store info about the original action dimensions if needed
        result = {
            'states': padded_states,
            'actions': padded_actions,
            'rewards': padded_rewards,
            'values': padded_values,
            'log_probs': padded_log_probs,
            'dones': padded_dones,
            'mask': padded_mask,
            'sequence_length': max_seq_len
        }
        
        if is_multidim_action:
            result['is_multidim_action'] = True
            result['original_action_shape'] = original_action_shape
            
        if is_multidim_values:
            result['is_multidim_values'] = True
            result['original_values_shape'] = original_values_shape
            
        if is_multidim_log_probs:
            result['is_multidim_log_probs'] = True
            result['original_log_probs_shape'] = original_log_probs_shape
            
        return result
        
    def update(self, states: List[np.ndarray], actions: List[int], old_log_probs: List[float], 
               returns: List[float], advantages: List[float], lstm_states=None, dones=None) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Args:
            states: List of observations
            actions: List of actions taken
            old_log_probs: List of log probabilities of actions with old policy
            returns: List of returns (discounted sum of rewards)
            advantages: List of advantages
            lstm_states: Initial LSTM states for recurrent policy (if applicable)
            dones: List of episode terminations (for properly resetting LSTM states)
            
        Returns:
            Dict containing training metrics
        """
        # Special handling for recurrent policy with processed sequence data
        if self.use_recurrent and isinstance(states, dict):
            # FIXED: Simplify to non-recurrent update for debugging
            # This is a temporary solution to verify that basic PPO is working
            try:
                # Extract the sequence data
                states_tensor = torch.FloatTensor(states['states']).to(self.device)
                
                # Flatten the sequences to simple batches
                batch_size, seq_len = states_tensor.shape[:2]
                flattened_states = states_tensor.reshape(-1, *states_tensor.shape[2:])
                
                # Flatten other tensors as well
                actions_tensor = torch.LongTensor(states['actions']).to(self.device).reshape(-1)
                old_log_probs_tensor = torch.FloatTensor(states['log_probs']).to(self.device).reshape(-1)
                returns_tensor = torch.FloatTensor(states['rewards']).to(self.device).reshape(-1)  # Use rewards as returns
                
                # We'll use flattened advantages, either from passed advantages or simple returns - values
                values_tensor = torch.FloatTensor(states['values']).to(self.device).reshape(-1)
                
                # Use the passed advantages if available
                if isinstance(advantages, np.ndarray):
                    # Try to reshape to match the flattened data
                    advantages_np = advantages.reshape(-1)
                    advantages_tensor = torch.FloatTensor(advantages_np).to(self.device)
                else:
                    # Compute simple advantages if none provided
                    advantages_tensor = returns_tensor - values_tensor
                
                # Get the mask for valid timesteps and flatten it
                masks_tensor = torch.FloatTensor(states['mask']).to(self.device).reshape(-1)
                
                # Normalize advantages if there are enough samples
                if self.update_adv_batch_norm and advantages_tensor.shape[0] > 1:
                    # FIXED: Only normalize if standard deviation is significant
                    adv_mean = advantages_tensor.mean()
                    adv_std = advantages_tensor.std()
                    if adv_std > 1e-4:
                        advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
                
                debug_print(f"Flattened data shapes for simplified LSTM update: "
                            f"states={flattened_states.shape}, actions={actions_tensor.shape}, "
                            f"advantages={advantages_tensor.shape}, masks={masks_tensor.shape}")
                
                # Track metrics for the update
                avg_policy_loss = 0
                avg_value_loss = 0
                avg_entropy = 0
                avg_approx_kl = 0
                clip_fractions = []
                
                # Flag for early stopping
                early_stop = False
                
                # Run multiple epochs of training on the data
                for _ in range(self.epochs):
                    # Create indexing for the flattened data
                    indices = torch.randperm(flattened_states.size(0))
                    
                    # Process mini-batches
                    for start_idx in range(0, len(indices), self.batch_size):
                        end_idx = min(start_idx + self.batch_size, len(indices))
                        batch_indices = indices[start_idx:end_idx]
                        
                        # Skip empty batches
                        if len(batch_indices) == 0:
                            continue
                        
                        # Get batch data
                        batch_states = flattened_states[batch_indices]
                        batch_actions = actions_tensor[batch_indices]
                        batch_old_log_probs = old_log_probs_tensor[batch_indices]
                        batch_returns = returns_tensor[batch_indices]
                        batch_advantages = advantages_tensor[batch_indices]
                        batch_masks = masks_tensor[batch_indices]
                        
                        # Forward pass with non-recurrent model (simplification)
                        policy_logits, values = self.model(batch_states)
                        
                        # Make sure values has the right shape
                        if values.dim() > 1:
                            values = values.squeeze(-1)
                        
                        # Create distribution and get log probabilities
                        dist = torch.distributions.Categorical(logits=policy_logits)
                        log_probs = dist.log_prob(batch_actions)
                        entropy = dist.entropy()
                        
                        # Calculate importance ratio
                        ratio = torch.exp(log_probs - batch_old_log_probs)
                        
                        # FIXED: Add debugging for ratio and advantages
                        debug_print(f"Policy stats - ratio shape: {ratio.shape}, ratio min/max: {ratio.min().item()}/{ratio.max().item()}")
                        debug_print(f"Advantage stats - shape: {batch_advantages.shape}, min/max: {batch_advantages.min().item()}/{batch_advantages.max().item()}")
                        
                        # Apply mask to ratio and advantages for valid timesteps only
                        masked_ratio = ratio * batch_masks
                        masked_advantages = batch_advantages * batch_masks
                        
                        # Calculate surrogate objectives
                        surr1 = masked_ratio * masked_advantages
                        surr2 = torch.clamp(masked_ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * masked_advantages
                        
                        # FIXED: Use mean instead of sum for policy loss
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # FIXED: Simplify value loss calculation
                        value_loss = F.mse_loss(values * batch_masks, batch_returns * batch_masks)
                        
                        # Calculate entropy loss
                        entropy_loss = entropy.mean()
                        
                        # Calculate approximate KL divergence for early stopping
                        with torch.no_grad():
                            log_ratio = log_probs - batch_old_log_probs
                            approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                            clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()
                            clip_fractions.append(clip_fraction)
                        
                        # Combine losses using coefficients
                        total_loss = policy_loss + self.vf_coef * value_loss - self.ent_reg * entropy_loss
                        
                        # Perform gradient update
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        
                        # FIXED: Add gradient checking
                        grad_info = {}
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                debug_print(f"Parameter {name} has no gradient!")
                                grad_info[name] = "no_grad"
                            elif param.grad.abs().sum() == 0:
                                debug_print(f"Parameter {name} has zero gradient!")
                                grad_info[name] = "zero_grad"
                            else:
                                grad_info[name] = param.grad.abs().sum().item()
                        
                        # Apply gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        # Perform optimization step
                        self.optimizer.step()
                        
                        # Update metrics
                        avg_policy_loss += policy_loss.item()
                        avg_value_loss += value_loss.item()
                        avg_entropy += entropy_loss.item()
                        avg_approx_kl += approx_kl
                        
                        # Check for early stopping based on KL divergence
                        if approx_kl > 1.5 * self.target_kl:
                            early_stop = True
                            break
                    
                    if early_stop:
                        break
                
                # Calculate average metrics
                num_updates = (self.epochs if not early_stop else _ + 1) * max(1, (len(indices) + self.batch_size - 1) // self.batch_size)
                avg_policy_loss /= max(1, num_updates)
                avg_value_loss /= max(1, num_updates)
                avg_entropy /= max(1, num_updates)
                avg_approx_kl /= max(1, num_updates)
                avg_clip_fraction = np.mean(clip_fractions) if clip_fractions else 0.0
                
                # Calculate explained variance
                with torch.no_grad():
                    explained_var = self.calculate_explained_variance(
                        values.cpu().numpy(), batch_returns.cpu().numpy())
                
                # Return metrics dictionary
                return {
                    'policy_loss': avg_policy_loss,
                    'value_loss': avg_value_loss,
                    'entropy': avg_entropy,
                    'kl': avg_approx_kl,
                    'explained_variance': explained_var,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'clip_fraction': avg_clip_fraction,
                    'early_stopped': early_stop,
                    'grad_info': grad_info,
                }
                
            except Exception as e:
                debug_print(f"Error in recurrent PPO update: {e}")
                import traceback
                traceback.print_exc()
                
                # Return default metrics to avoid breaking training
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy': self.ent_reg,
                    'kl': 0.0,
                    'explained_variance': 0.0,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'clip_fraction': 0.0,
                    'early_stopped': False,
                    'error': str(e),
                }
            
        else:
            # Original update for non-recurrent policy
            # Convert lists to tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            if len(states_tensor.shape) == 3:  # Handle single state
                states_tensor = states_tensor.unsqueeze(0)
                
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            
            # FIXED: Normalize advantages only if std is significant
            if self.update_adv_batch_norm and len(advantages_tensor) > 1:
                adv_mean = advantages_tensor.mean()
                adv_std = advantages_tensor.std()
                if adv_std > 1e-4:
                    advantages_tensor = (advantages_tensor - adv_mean) / (adv_std + 1e-8)
            
            # Mini-batch updates
            n_samples = len(states)
            batch_indices = np.arange(n_samples)
            
            # Track metrics
            avg_policy_loss = 0
            avg_value_loss = 0
            avg_entropy = 0
            avg_approx_kl = 0
            clip_fractions = []
            grad_info = {}
            
            early_stop = False
            
            # Special handling for recurrent policy
            if self.use_recurrent and dones is not None:
                dones_tensor = torch.FloatTensor(dones).to(self.device)
            else:
                dones_tensor = None
                
            # Perform multiple epochs of updates
            for _ in range(self.epochs):
                # Randomize batches each epoch
                np.random.shuffle(batch_indices)
                
                # Process mini-batches
                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    batch_idx = batch_indices[start_idx:end_idx]
                    
                    # Get batch tensors
                    batch_states = states_tensor[batch_idx]
                    batch_actions = actions_tensor[batch_idx]
                    batch_old_log_probs = old_log_probs_tensor[batch_idx]
                    batch_returns = returns_tensor[batch_idx]
                    batch_advantages = advantages_tensor[batch_idx]
                    
                    if self.use_recurrent:
                        # For recurrent policies, we need to organize data into sequences
                        # This is a simplified version that works with continuous sequences
                        # A more sophisticated implementation would respect episode boundaries
                        
                        # Handle sequences for recurrent policy
                        if dones_tensor is not None:
                            batch_dones = dones_tensor[batch_idx]
                        else:
                            batch_dones = None
                        
                        # Format states into sequences for LSTM processing
                        seq_len = min(self.recurrent_seq_len, len(batch_idx))
                        
                        # Get current policy outputs with sequence processing
                        policy_logits, values, _ = self.model(
                            batch_states,
                            lstm_state=None,  # We'll let the model initialize states
                            sequence_length=seq_len if seq_len > 1 else 1
                        )
                        
                        values = values.squeeze(-1) if values.dim() > 1 else values
                    else:
                        # Get current policy outputs for standard policy
                        policy_logits, values = self.model(batch_states)
                        values = values.squeeze(-1)
                    
                    # FIXED: Add debug prints for policy logits and values
                    debug_print(f"Policy logits: shape={policy_logits.shape}, min/max={policy_logits.min().item()}/{policy_logits.max().item()}")
                    debug_print(f"Values: shape={values.shape}, min/max={values.min().item()}/{values.max().item()}")
                    
                    # Calculate policy distribution
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # Compute importance ratio and clipped surrogate objective
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    
                    # FIXED: Add debug print for policy ratio
                    debug_print(f"Policy ratio: min/max={ratio.min().item()}/{ratio.max().item()}")
                    debug_print(f"Advantages: min/max={batch_advantages.min().item()}/{batch_advantages.max().item()}")
                    
                    # Calculate policy loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advantages
                    
                    # FIXED: Use mean instead of sum for policy loss
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # FIXED: Simplify value loss calculation
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # Calculate approximate KL divergence for early stopping
                    with torch.no_grad():
                        log_ratio = log_probs - batch_old_log_probs
                        approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_eps).float()).item()
                        clip_fractions.append(clip_fraction)
                    
                    # Total loss
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_reg * entropy
                    
                    # Perform gradient step
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # FIXED: Add gradient checking
                    for name, param in self.model.named_parameters():
                        if param.grad is None:
                            debug_print(f"Parameter {name} has no gradient!")
                            grad_info[name] = "no_grad"
                        elif param.grad.abs().sum() == 0:
                            debug_print(f"Parameter {name} has zero gradient!")
                            grad_info[name] = "zero_grad"
                        else:
                            grad_info[name] = param.grad.abs().sum().item()
                    
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
                    
                    # Update metrics
                    avg_policy_loss += policy_loss.item()
                    avg_value_loss += value_loss.item()
                    avg_entropy += entropy.item()
                    avg_approx_kl += approx_kl
                    
                    # Check if we need to stop early due to KL divergence
                    if approx_kl > 1.5 * self.target_kl:
                        early_stop = True
                        break
                        
                if early_stop:
                    break
            
            # Compute average metrics
            num_updates = (self.epochs if not early_stop else _ + 1) * math.ceil(n_samples / self.batch_size)
            avg_policy_loss /= max(1, num_updates)
            avg_value_loss /= max(1, num_updates)
            avg_entropy /= max(1, num_updates)
            avg_approx_kl /= max(1, num_updates)
            avg_clip_fraction = np.mean(clip_fractions)
            
            # Compute explained variance
            explained_var = self.calculate_explained_variance(values.detach().cpu().numpy(), batch_returns.cpu().numpy())
        
        # Update metrics dictionary
        self.metrics['policy_loss'].append(avg_policy_loss)
        self.metrics['value_loss'].append(avg_value_loss)
        self.metrics['entropy'].append(avg_entropy)
        self.metrics['approx_kl'].append(avg_approx_kl)
        self.metrics['clip_fraction'].append(avg_clip_fraction)
        self.metrics['stop_iteration'].append(1 if early_stop else 0)
        self.metrics['explained_variance'].append(explained_var)
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
            self.metrics['learning_rate'].append(self.scheduler.get_last_lr()[0])
        else:
            self.metrics['learning_rate'].append(self.lr)
        
        # Update entropy coefficient if using adaptive entropy
        if self.adaptive_entropy:
            # Slowly decay entropy coefficient, but maintain minimum level
            self.ent_reg = max(self.ent_reg * self.entropy_decay_factor, self.min_entropy)
        
        # Count updates since last optimizer reset
        self.updates_since_lr_reset += 1
        
        # Periodically reset optimizer state to escape plateaus
        if self.lr_reset_interval > 0 and self.updates_since_lr_reset >= self.lr_reset_interval:
            self.reset_optimizer_state()
            self.updates_since_lr_reset = 0

        grads_exist = sum(1 for name, param in self.model.named_parameters() 
                if param.requires_grad and param.grad is not None)
        grads_zero = sum(1 for name, param in self.model.named_parameters() 
                        if param.requires_grad and param.grad is not None and param.grad.abs().sum() == 0)
        grads_missing = sum(1 for name, param in self.model.named_parameters() 
                        if param.requires_grad and param.grad is None)
            
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl': avg_approx_kl,
            'explained_variance': explained_var,
            'learning_rate': self.metrics['learning_rate'][-1],
            'clip_fraction': avg_clip_fraction,
            'early_stopped': early_stop,
            'grads_exist': grads_exist,
            'grads_zero': grads_zero, 
            'grads_missing': grads_missing
        }
        
    def select_action(self, state, deterministic=False):
        """
        Select an action using the current policy.
        
        Args:
            state: Current observation
            deterministic: Whether to select the best action deterministically
            
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Value estimate for the current state
            entropy: Entropy of the policy distribution
        """
        with torch.no_grad():
            # Ensure state is a tensor and add batch dimension if needed
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 3:
                state = state.unsqueeze(0)
                
            # Get action using the appropriate forward method based on recurrence
            if self.use_recurrent:
                # Don't pass deterministic to forward method
                policy_logits, value, next_lstm_state = self.model(
                    state, self.lstm_states
                )
                # Update LSTM state
                self.lstm_states = next_lstm_state
                
                # Handle multi-dimensional outputs
                if policy_logits.dim() > 2:  # [batch, sequence, action_dim]
                    # Get policy for the last timestep in the sequence
                    policy_logits = policy_logits[:, -1]
                
                # Create distribution and sample action
                dist = torch.distributions.Categorical(logits=policy_logits)
                if deterministic:
                    action = torch.argmax(policy_logits, dim=1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                # Handle multi-dimensional value outputs
                if value.dim() > 1 and value.size(0) == 1:
                    # If value has sequence dimension, take the last one
                    value = value[:, -1] if value.dim() == 2 else value[0, -1]
            else:
                policy_logits, value = self.model(state)
                
                # Handle multi-dimensional outputs
                if policy_logits.dim() > 2:  # Unexpected extra dimensions
                    policy_logits = policy_logits.squeeze(0)  # Remove batch dim if needed
                
                # Create distribution and sample action
                dist = torch.distributions.Categorical(logits=policy_logits)
                if deterministic:
                    action = torch.argmax(policy_logits, dim=1)
                else:
                    action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                # Squeeze extra dimensions from value
                value = value.squeeze(-1) if value.dim() > 1 else value
                
            # Convert to numpy or python types for environment interaction
            # Safely handle multi-dimensional action tensors
            if action.dim() > 1 or action.size(0) > 1:
                action = action.view(-1)  # Flatten to 1D
                action = action.cpu().numpy()  # Convert to numpy array
            else:
                action = action.item() if action.numel() == 1 else action.cpu().numpy()
                
            # Handle multi-dimensional log_prob
            if isinstance(log_prob, torch.Tensor):
                if log_prob.dim() > 1 or log_prob.size(0) > 1:
                    log_prob = log_prob.view(-1)  # Flatten to 1D
                    log_prob = log_prob.cpu().numpy()  # Convert to numpy array
                else:
                    log_prob = log_prob.item() if log_prob.numel() == 1 else log_prob.cpu().numpy()
                
            # Handle multi-dimensional value
            if isinstance(value, torch.Tensor):
                if value.dim() > 1 or value.size(0) > 1:
                    value = value.view(-1)  # Flatten to 1D
                    value = value.cpu().numpy()  # Convert to numpy array
                else:
                    value = value.item() if value.numel() == 1 else value.cpu().numpy()
                
            # Handle multi-dimensional entropy
            if isinstance(entropy, torch.Tensor):
                if entropy.dim() > 1 or entropy.size(0) > 1:
                    entropy = entropy.view(-1)  # Flatten to 1D
                    entropy = entropy.cpu().numpy()  # Convert to numpy array
                else:
                    entropy = entropy.item() if entropy.numel() == 1 else entropy.cpu().numpy()
                    
            return action, log_prob, value, entropy
            
    def process_done(self, done):
        """
        Process episode termination for LSTM state management.
        
        Args:
            done: Whether the episode has terminated
        """
        if done and self.use_recurrent:
            # Reset LSTM state at the end of an episode
            self.reset_lstm_state()
        
    def update_lr_scheduler(self, reward):
        """Update learning rate based on rewards."""
        if self.lr_scheduler != 'adaptive':
            return
            
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 10:
            self.recent_rewards.pop(0)
            
        # Keep track of best reward seen
        if reward > self.max_reward:
            self.max_reward = reward
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        # If rewards are stagnating for too long, reduce learning rate
        if self.stagnation_counter > 50:
            # Reduce learning rate by 50%
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * 0.5, 1e-6)
            
            # Reset counter
            self.stagnation_counter = 0
            debug_print(f"Learning rate reduced to {self.optimizer.param_groups[0]['lr']}")

    def reset_optimizer_state(self):
        """Reset optimizer state to escape plateaus in training."""
        # Store current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # Recreate optimizer with fresh state
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=current_lr)
        
        # Reset scheduler if needed
        if self.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10000, eta_min=self.lr * 0.1
            )
        elif self.lr_scheduler == 'linear':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: max(1.0 - step / 10000, 0.1)
            )
            
        debug_print("Optimizer state reset to escape potential plateaus")
        
    def set_exploration_mode(self, exploring=True):
        """Set the agent's mode to emphasize exploration or exploitation."""
        if exploring:
            # Boost entropy coefficient for more exploration
            self.ent_reg = max(0.05, self.ent_reg * 2.0)
            debug_print(f"Exploration mode activated - entropy coefficient increased to {self.ent_reg}")
        else:
            # Reduce entropy coefficient for more exploitation
            self.ent_reg = min(0.005, self.ent_reg * 0.5)
            debug_print(f"Exploitation mode activated - entropy coefficient reduced to {self.ent_reg}")
            
        # Also adjust learning rate
        if exploring:
            # Slightly higher learning rate for exploration
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min(param_group['lr'] * 1.5, self.lr * 2.0)
        else:
            # Lower learning rate for fine-tuning
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.7

    def update_performance(self, performance_metric):
        """
        Update exploration parameters based on agent performance.
        Similar to what SB3 does automatically.
        
        Args:
            performance_metric: A measure of agent performance (reward, success rate, etc.)
        """
        # Track performance
        self.recent_rewards.append(performance_metric)
        if len(self.recent_rewards) > 20:  # Keep last 20 episodes
            self.recent_rewards.pop(0)
            
        # Check if we have enough data to make decisions
        if len(self.recent_rewards) < 10:
            return
            
        # Calculate recent performance trends
        recent_avg = sum(self.recent_rewards[-10:]) / 10
        if len(self.recent_rewards) >= 20:
            previous_avg = sum(self.recent_rewards[-20:-10]) / 10
            performance_change = recent_avg - previous_avg
        else:
            # Not enough history yet
            performance_change = 0
            
        # Update maximum reward seen
        if performance_metric > self.max_reward:
            self.max_reward = performance_metric
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        # If performance is improving, gradually shift toward exploitation
        if performance_change > self.entropy_boost_threshold:
            # Slowly reduce entropy for more exploitation
            new_ent_reg = max(self.min_entropy, self.ent_reg * 0.95)
            if new_ent_reg != self.ent_reg:
                debug_print(f"Performance improving - reducing entropy from {self.ent_reg:.4f} to {new_ent_reg:.4f}")
                self.ent_reg = new_ent_reg
                
        # If performance is stagnating or decreasing, boost exploration
        elif self.stagnation_counter > 15 or performance_change < -self.entropy_boost_threshold:
            # Boost entropy for more exploration
            original_ent_reg = self.ent_reg
            self.ent_reg = min(0.1, self.ent_reg * 1.5)
            
            if original_ent_reg != self.ent_reg:
                debug_print(f"Performance stagnating - boosting entropy from {original_ent_reg:.4f} to {self.ent_reg:.4f}")
                
            # Reset stagnation counter
            if self.stagnation_counter > 30:
                # If stagnation continues for too long, consider resetting optimizer
                if self.updates_since_lr_reset > 10:  # Don't reset too frequently
                    self.reset_optimizer_state()
                    self.updates_since_lr_reset = 0
                
                # Reset stagnation counter
                self.stagnation_counter = 0

    def initialize_icm(self, input_shape, action_dim):
        """
        Initialize Intrinsic Curiosity Module for exploration.
        
        Args:
            input_shape: Shape of observations
            action_dim: Dimension of action space
        """
        if not self.use_icm:
            return
            
        # Import here to avoid circular imports
        from src.icm import ICM
        
        # Create ICM module
        self.icm = ICM(
            input_shape=input_shape,
            action_dim=action_dim,
            feature_dim=self.icm_feature_dim,
            forward_scale=self.icm_forward_weight,
            inverse_scale=self.icm_inverse_weight
        ).to(self.device)
        
        # Set up ICM optimizer
        self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=self.icm_lr)
        
        debug_print(f"Initialized ICM with feature dimension {self.icm_feature_dim}")
        
    def update_icm(self, states, actions, next_states):
        """
        Update ICM module.
        
        Args:
            states: Current observations
            actions: Actions taken
            next_states: Next observations
            
        Returns:
            Dict containing ICM losses and rewards
        """
        if not self.use_icm or self.icm is None:
            return None
            
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(np.array(states)).to(self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        if not isinstance(actions, torch.Tensor):
            if isinstance(actions[0], int):
                # Discrete actions
                actions = torch.LongTensor(actions).to(self.device)
            else:
                # Continuous or multi-discrete actions
                actions = torch.FloatTensor(np.array(actions)).to(self.device)
        
        # Forward pass through ICM
        icm_results = self.icm(states, next_states, actions)
        
        # Backward pass
        icm_loss = (self.icm_forward_weight * icm_results['forward_loss'] + 
                   self.icm_inverse_weight * icm_results['inverse_loss'])
        
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.max_grad_norm)
        
        self.icm_optimizer.step()
        
        # Return ICM results
        return {
            'icm_forward_loss': icm_results['forward_loss'].item(),
            'icm_inverse_loss': icm_results['inverse_loss'].item(),
            'icm_loss': icm_loss.item(),
            'intrinsic_reward_mean': icm_results['intrinsic_reward'].mean().item(),
            'intrinsic_reward_max': icm_results['intrinsic_reward'].max().item()
        }
        
    def get_intrinsic_reward(self, state, next_state, action):
        """
        Calculate intrinsic reward for exploration.
        
        Args:
            state: Current observation
            next_state: Next observation
            action: Action taken
            
        Returns:
            Intrinsic reward (exploration bonus)
        """
        if not self.use_icm or self.icm is None:
            return 0.0
            
        try:
            # Get intrinsic reward from ICM
            intrinsic_reward = self.icm.get_intrinsic_reward(state, next_state, action)
            
            # Scale the reward
            scaled_reward = intrinsic_reward * self.icm_reward_scale
            
            return scaled_reward
        except Exception as e:
            debug_print(f"Error in get_intrinsic_reward: {e}")
            return 0.0