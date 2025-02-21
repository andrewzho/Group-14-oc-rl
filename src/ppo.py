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

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        # Convert states from a list of NumPy arrays to a single NumPy array
        states = np.array(states)  # Shape should be [batch_size, 3, 84, 84]
        print("States shape before batching:", states.shape)  # Debug print

        total_steps = len(states)
        for _ in range(self.epochs):
            for idx in range(0, total_steps, self.batch_size):
                # Ensure idx doesn't exceed total_steps
                end_idx = min(idx + self.batch_size, total_steps)
                batch_states = torch.tensor(states[idx:end_idx], dtype=torch.float32)
                batch_actions = torch.tensor(actions[idx:end_idx], dtype=torch.long)
                batch_old_log_probs = torch.tensor(old_log_probs[idx:end_idx], dtype=torch.float32)
                batch_returns = torch.tensor(returns[idx:end_idx], dtype=torch.float32)
                batch_advantages = torch.tensor(advantages[idx:end_idx], dtype=torch.float32)

                policy_logits, values = self.model(batch_states)
                dist = torch.distributions.Categorical(logits=policy_logits)
                new_log_probs = dist.log_prob(batch_actions)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()