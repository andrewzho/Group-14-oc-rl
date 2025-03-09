import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import torch.optim as optim

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
        ent_reg=0.02,  # Start with higher entropy regularization
        max_grad_norm=0.5,
        target_kl=0.05,
        lr_scheduler='cosine',
        adaptive_entropy=True,  # New parameter for adaptive entropy
        min_entropy=0.005,     # Minimum entropy level
        entropy_decay_factor=0.9999,  # Slow decay for entropy
        update_adv_batch_norm=True    # Added parameter for advantage batch normalization
    ):
        """
        Proximal Policy Optimization algorithm with adaptive entropy.
        
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
        self.initial_ent_reg = ent_reg  # Store initial entropy reg for resets
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.adaptive_entropy = adaptive_entropy
        self.min_entropy = min_entropy
        self.entropy_decay_factor = entropy_decay_factor
        self.update_adv_batch_norm = update_adv_batch_norm
        
        # Adaptive learning rate to break out of plateaus
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)
        
        # Enhanced learning rate scheduler
        self.lr_scheduler = lr_scheduler
        if lr_scheduler == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000)
        elif lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2, eta_min=lr/10
            )
        elif lr_scheduler == 'plateau':
            # Reduce LR on plateau to adapt to learning stagnation
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        else:
            self.scheduler = None
            
        # Learning rate warmup
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=100
        )
        self.warmup_steps = 0
        self.warmup_total = 100
        
        # Metrics tracking
        self.policy_losses = []
        self.value_losses = []
        self.entropy_values = []
        self.clip_fractions = []
        self.approx_kl_divs = []
        self.explained_variances = []
        self.learning_rates = []
        
        # Stagnation detection
        self.mean_policy_loss = 0
        self.policy_loss_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 10  # Number of updates with minimal change to detect stagnation
        
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
        
    def update(self, states: List[np.ndarray], actions: List[int], old_log_probs: List[float], 
               returns: List[float], advantages: List[float]) -> Dict[str, float]:
        """
        Update the model using PPO with enhanced stability features.
        
        Args:
            states: List of states
            actions: List of actions
            old_log_probs: List of log probabilities from old policy
            returns: List of discounted returns
            advantages: List of advantages
            
        Returns:
            Dictionary of metrics
        """
        # Sanity check for valid inputs
        if len(states) == 0 or len(actions) == 0:
            print("Warning: Empty batch in update call")
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'clip_fraction': 0.0,
                'approx_kl': 0.0,
                'explained_variance': 0.0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'entropy_reg': self.ent_reg
            }
        
        # Check for NaN values in advantages
        if np.isnan(np.sum(advantages)):
            print("Warning: NaN detected in advantages, skipping update")
            # Reset advantage values to prevent propagation
            advantages = [0.0 if np.isnan(adv) else adv for adv in advantages]
        
        # Apply warmup if still in warmup phase
        if self.warmup_steps < self.warmup_total:
            self.warmup_scheduler.step()
            self.warmup_steps += 1
            
        # Convert to tensors and normalize advantages
        device = next(self.model.parameters()).device
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # Normalize advantages for more stable training
        if self.update_adv_batch_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track metrics
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        avg_clip_fraction = 0
        avg_approx_kl = 0
        
        total_steps = len(states)
        
        # Multiple epochs of training on the same data for better sample efficiency
        for epoch in range(self.epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(total_steps)
            
            # Track epoch-level metrics
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_clip_fraction = 0
            epoch_approx_kl = 0
            epoch_batches = 0
            
            # Process data in mini-batches
            for start in range(0, total_steps, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                state_batch = states[batch_indices]
                action_batch = actions[batch_indices]
                old_log_prob_batch = old_log_probs[batch_indices]
                return_batch = returns[batch_indices]
                advantage_batch = advantages[batch_indices]
                
                # Forward pass
                policy_logits, value_pred = self.model(state_batch)
                dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(action_batch)
                entropy = dist.entropy().mean()
                
                # Check for valid actions
                if (action_batch >= dist.probs.shape[1]).any():
                    print("Invalid action detected!")
                    continue
                
                # Calculate policy loss with clipping
                ratio = torch.exp(new_log_probs - old_log_prob_batch)
                # Add numerical stability - ratio can't be too extreme
                ratio = torch.clamp(ratio, 0.0, 10.0)
                
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss with clipping to prevent too large updates
                value_pred = value_pred.squeeze(-1)
                v_loss_unclipped = F.mse_loss(value_pred, return_batch)
                v_clipped = old_log_prob_batch + torch.clamp(
                    value_pred - old_log_prob_batch,
                    -self.clip_eps,
                    self.clip_eps
                )
                v_loss_clipped = F.mse_loss(v_clipped, return_batch)
                value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
                
                # Calculate KL divergence for early stopping
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_prob_batch
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                
                # Early stopping based on KL divergence
                if approx_kl > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch+1}/{self.epochs} due to reaching KL threshold")
                    break
                
                # Calculate clip fraction
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_eps).float())
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_reg * entropy
                
                # Optimization step with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update epoch metrics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_entropy += entropy.item()
                epoch_clip_fraction += clip_fraction.item()
                epoch_approx_kl += approx_kl
                epoch_batches += 1
            
            # Compute epoch averages
            if epoch_batches > 0:
                avg_policy_loss += epoch_policy_loss / epoch_batches
                avg_value_loss += epoch_value_loss / epoch_batches
                avg_entropy += epoch_entropy / epoch_batches
                avg_clip_fraction += epoch_clip_fraction / epoch_batches
                avg_approx_kl += epoch_approx_kl / epoch_batches
        
        # Calculate final averages
        num_epochs = self.epochs
        avg_policy_loss /= num_epochs
        avg_value_loss /= num_epochs
        avg_entropy /= num_epochs
        avg_clip_fraction /= num_epochs
        avg_approx_kl /= num_epochs
        
        # Check for policy stagnation
        self.policy_loss_history.append(avg_policy_loss)
        if len(self.policy_loss_history) > 10:
            self.policy_loss_history.pop(0)
            
        # Detect stagnation by checking if policy loss is very small
        if abs(avg_policy_loss) < 0.001:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # Break out of stagnation by temporarily increasing entropy
        if self.stagnation_counter >= self.stagnation_threshold:
            print("Policy stagnation detected, increasing entropy regularization")
            self.ent_reg = min(0.1, self.ent_reg * 3)  # Boost entropy to escape local optimum
            self.stagnation_counter = 0
        elif self.adaptive_entropy:
            # Gradually decay entropy over time, but not below minimum
            self.ent_reg = max(self.min_entropy, self.ent_reg * self.entropy_decay_factor)
        
        # Calculate explained variance
        with torch.no_grad():
            _, value_preds = self.model(states)
            value_preds = value_preds.squeeze().cpu().numpy()
            actual_returns = returns.cpu().numpy()
            explained_var = self.calculate_explained_variance(value_preds, actual_returns)
        
        # Update learning rate if using a scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Update plateau scheduler based on episode reward (will be passed later)
                pass
            else:
                self.scheduler.step()
        
        # Store metrics
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_values.append(avg_entropy)
        self.clip_fractions.append(avg_clip_fraction)
        self.approx_kl_divs.append(avg_approx_kl)
        self.explained_variances.append(explained_var)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        metrics = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'clip_fraction': avg_clip_fraction,
            'approx_kl': avg_approx_kl,
            'explained_variance': explained_var,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'entropy_reg': self.ent_reg
        }
        
        return metrics
        
    def update_lr_scheduler(self, reward):
        """Update learning rate scheduler using rewards for plateau detection"""
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(reward)

    def reset_optimizer_state(self):
        """Reset optimizer to recover from low learning rates"""
        print(f"Resetting optimizer. Old learning rate: {self.optimizer.param_groups[0]['lr']}")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        if self.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2, eta_min=self.lr/10
            )
        print(f"Reset optimizer with learning rate: {self.lr}")