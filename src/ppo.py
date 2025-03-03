import torch
import torch.nn.functional as F
import numpy as np

class PPO:
    def __init__(self, model, lr=3e-4, clip_eps=0.2, gamma=0.99, gae_lambda=0.95, epochs=4, batch_size=64):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        # Keep track of losses for monitoring
        self.policy_losses = []
        self.value_losses = []
        self.entropy_values = []

    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy and value function using PPO"""
        # Convert all inputs to tensors and move to model's device
        device = next(self.model.parameters()).device
        states = np.stack(states)  # Shape: [batch_size, 12, 84, 84]
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        
        total_steps = len(states)
        
        # Track metrics for this update
        avg_policy_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        
        for _ in range(self.epochs):
            # Generate random indices for batching
            indices = np.random.permutation(total_steps)
            
            for start_idx in range(0, total_steps, self.batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # Prepare batch data
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                # Forward pass
                policy_logits, values = self.model(batch_states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()  # Entropy bonus for exploration
                
                # Calculate policy loss with clipping
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss with clipping
                value_pred = values.squeeze()
                value_loss = F.mse_loss(value_pred, batch_returns)
                
                # Total loss with entropy regularization (to encourage exploration)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # Track metrics
                avg_policy_loss += policy_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy += entropy.item()
        
        # Calculate averages
        num_batches = (total_steps + self.batch_size - 1) // self.batch_size * self.epochs
        avg_policy_loss /= num_batches
        avg_value_loss /= num_batches
        avg_entropy /= num_batches
        
        # Store for monitoring
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_values.append(avg_entropy)
        
        # Print some metrics
        print(f"Update - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")