import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import math
import time

# Prioritized Experience Replay
class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer for storing experience tuples with priorities."""
    
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Initialize a PrioritizedReplayBuffer object.
        
        Args:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            alpha (float): priority exponent parameter
            beta_start (float): importance sampling weight parameter
            beta_frames (int): number of frames over which to anneal beta
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Storage for experiences
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", 
                                                              "next_state", "done"])
        self.memory = []
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximum priority."""
        e = self.experience(state, action, reward, next_state, done)
        
        if self.size < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.pos] = e
        
        # Set max priority for new experience
        max_priority = max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.buffer_size:
            self.priorities[self.size] = max_priority
        else:
            self.priorities[self.pos] = max_priority
        
        # Update position and size
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        """Sample a batch of experiences based on priorities."""
        if self.size < self.batch_size:
            return None
        
        # Anneal beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Calculate sampling probabilities from priorities
        if self.size == self.buffer_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.size]
        
        # Apply exponent alpha to get sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities = probabilities / sum(probabilities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(probabilities), self.batch_size, replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        
        # Get selected experiences
        experiences = [self.memory[idx] for idx in indices]
        
        # Extract batch components
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        weights = torch.from_numpy(weights).float()
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences."""
        # Ensure priorities is an array even if a scalar is passed
        if np.isscalar(priorities):
            print(f"WARNING: Scalar priority value received: {priorities}. Converting to array.")
            priorities = np.full_like(indices, priorities, dtype=np.float32)
        
        # Ensure we're working with arrays
        indices = np.asarray(indices)
        priorities = np.asarray(priorities)
        
        # Make sure lengths match
        if len(indices) != len(priorities):
            print(f"ERROR: Length mismatch! Indices: {len(indices)}, Priorities: {len(priorities)}")
            # Trim or extend priorities to match indices length
            if len(priorities) > len(indices):
                priorities = priorities[:len(indices)]
            else:
                priorities = np.append(priorities, np.ones(len(indices) - len(priorities)))
        
        # Update priorities in the buffer (vectorized operation)
        # This is faster than a loop for larger batches
        self.priorities[indices] = priorities
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size

# N-step Return Memory for more efficient learning
class NStepReplayBuffer:
    """Buffer for storing n-step experience transitions."""
    
    def __init__(self, n_steps, gamma):
        """Initialize buffer for n-step returns.
        
        Args:
            n_steps: Number of steps for n-step return calculation
            gamma: Discount factor
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def get(self):
        """Get the n-step return experience if available."""
        if len(self.buffer) < self.n_steps:
            return None
        
        # Get the earliest experience in the buffer
        state, action, _, _, _ = self.buffer[0]
        
        # Calculate the n-step reward
        n_reward = 0
        for i in range(self.n_steps):
            n_reward += (self.gamma ** i) * self.buffer[i][2]
        
        # Get the final next state and done
        _, _, _, next_state, done = self.buffer[self.n_steps - 1]
        
        # Return n-step transition
        return state, action, n_reward, next_state, done
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()

# Main Rainbow DQN Agent
class RainbowDQN:
    """Agent implementing Rainbow DQN with RND for intrinsic motivation."""
    
    def __init__(self, state_shape, action_size, device, 
                 buffer_size=100000, batch_size=32, gamma=0.99, 
                 n_steps=3, num_atoms=51, v_min=-10, v_max=10,
                 target_update=1000, lr=0.0001, eps_start=1.0, 
                 eps_end=0.01, eps_decay=30000, alpha=0.5, 
                 beta_start=0.4, beta_frames=100000, 
                 rnd_lr=0.001, int_coef=0.1, ext_coef=1.0,
                 rnd_target_update=50000):
        """Initialize Rainbow DQN agent with RND.
        
        Args:
            state_shape: Shape of state observation (C, H, W)
            action_size: Number of possible actions
            device: Device for computation (CPU/GPU)
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            n_steps: Number of steps for n-step returns
            num_atoms: Number of atoms for distributional RL
            v_min: Minimum value for distributional support
            v_max: Maximum value for distributional support
            target_update: How often to update target network
            lr: Learning rate for optimizer
            eps_start: Starting value for epsilon in epsilon-greedy policy
            eps_end: Minimum value for epsilon
            eps_decay: Decay rate for epsilon
            alpha: Priority exponent for prioritized replay
            beta_start: Initial value of beta for importance sampling
            beta_frames: Number of frames over which to anneal beta
            rnd_lr: Learning rate for RND predictor
            int_coef: Weight for intrinsic reward
            ext_coef: Weight for extrinsic reward
            rnd_target_update: How often to update RND normalization stats
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_steps = n_steps
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.target_update = target_update
        self.rnd_target_update = rnd_target_update
        self.int_coef = int_coef
        self.ext_coef = ext_coef
        
        # Value distribution support
        self.support = torch.linspace(v_min, v_max, num_atoms).to(device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # Import models from rainbow_model.py
        from src.models.rainbow_model import RainbowDQN as DQNModel
        from src.models.rainbow_model import RNDModel
        
        # Initialize networks
        self.policy_net = DQNModel(state_shape, action_size, num_atoms, v_min, v_max).to(device)
        self.target_net = DQNModel(state_shape, action_size, num_atoms, v_min, v_max).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize RND model for intrinsic motivation
        self.rnd_model = RNDModel(state_shape).to(device)
        
        # Initialize optimizers with better parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1.5e-4)
        self.rnd_optimizer = optim.Adam(self.rnd_model.predictor.parameters(), lr=rnd_lr, eps=1.5e-4)
        
        # Add learning rate schedulers for better convergence
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.5)
        self.rnd_scheduler = optim.lr_scheduler.StepLR(self.rnd_optimizer, step_size=100000, gamma=0.5)
        
        # Experience replay buffers
        self.memory = PrioritizedReplayBuffer(buffer_size, batch_size, alpha, beta_start, beta_frames)
        self.n_step_memory = NStepReplayBuffer(n_steps, gamma)
        
        # Epsilon for exploration
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        
        # For RND normalization
        self.obs_rms = RunningMeanStd(shape=())
        self.int_reward_rms = RunningMeanStd(shape=())
        self.update_rnd_counter = 0
    
    def select_action(self, state, evaluate=False):
        """Select an action using epsilon-greedy policy."""
        if evaluate:
            # Use greedy policy for evaluation
            return self.policy_net.act(state, epsilon=0.0)
        
        # Calculate epsilon based on step count
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
                 math.exp(-1. * self.steps_done / self.eps_decay)
        
        # Select action based on epsilon-greedy policy
        return self.policy_net.act(state, epsilon=epsilon)
    
    def preprocess_state(self, state):
        """Preprocess state (normalize, transpose etc)."""
        # Normalize pixel values and convert to channels first
        if isinstance(state, np.ndarray) and state.dtype == np.uint8:
            state = state.astype(np.float32) / 255.0
        
        # Adjust state dimensions if needed
        if len(state.shape) == 3 and state.shape[0] != self.state_shape[0]:
            # Convert from channels-last to channels-first
            state = np.transpose(state, (2, 0, 1))
        
        return state
    
    def calculate_intrinsic_reward(self, next_state):
        """Calculate intrinsic reward using RND prediction error."""
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Get target and prediction from RND model
            target_feature, predict_feature = self.rnd_model(next_state_tensor)
            
            # Calculate prediction error (MSE)
            intrinsic_reward = ((target_feature - predict_feature) ** 2).mean().item()
            
            # Normalize intrinsic reward
            intrinsic_reward = self.normalize_reward(intrinsic_reward)
            
            # Apply a non-linear scaling to make rare states more valuable
            # This helps prevent the agent from getting stuck in a loop of novel states
            intrinsic_reward = np.sqrt(intrinsic_reward) * 3.0
            
            return intrinsic_reward
    
    def normalize_reward(self, reward):
        """Normalize intrinsic reward using running statistics."""
        self.int_reward_rms.update(np.array([reward]))
        reward = reward / (self.int_reward_rms.var ** 0.5 + 1e-8)
        return min(reward, 5.0)  # Clip to prevent excessive intrinsic rewards
    
    def step(self, state, action, reward, next_state, done):
        """Process a step and save to replay memory."""
        # Preprocess states
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        
        # Calculate intrinsic reward
        intrinsic_reward = self.calculate_intrinsic_reward(next_state)
        
        # Combined reward
        combined_reward = self.ext_coef * reward + self.int_coef * intrinsic_reward
        
        # Add to n-step buffer
        self.n_step_memory.add(state, action, combined_reward, next_state, done)
        
        # Get n-step transition if available
        n_step_transition = self.n_step_memory.get()
        if n_step_transition:
            self.memory.add(*n_step_transition)
        
        # Reset n-step buffer if episode ends
        if done:
            self.n_step_memory.clear()
        
        # Increment steps count
        self.steps_done += 1
        
        # Periodically update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Target network updated at step {self.steps_done}")
        
        # Periodically update RND normalization stats
        self.update_rnd_counter += 1
        if self.update_rnd_counter >= self.rnd_target_update:
            self.update_rnd_counter = 0
            print(f"RND stats updated. Intrinsic reward var: {self.int_reward_rms.var:.4f}")
    
    def learn(self):
        """Learn from a batch of experiences."""
        # Sample from replay buffer
        experiences = self.memory.sample()
        if experiences is None:
            return 0  # Not enough samples in buffer
        
        states, actions, rewards, next_states, dones, indices, weights = experiences
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # Reshape actions to match expected dimensions
        actions = actions.squeeze(1)
        
        # Get current Q distributions
        current_q_dist = self.policy_net(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]
        
        # Compute next state Q distributions (for double DQN)
        with torch.no_grad():
            # Get actions from online network
            policy_dist = self.policy_net(next_states)
            policy_dist = policy_dist * self.support.expand_as(policy_dist)
            policy_dist = policy_dist.sum(dim=2)
            next_actions = policy_dist.max(1)[1]
            
            # Get distributions from target network for selected actions
            next_q_dist = self.target_net(next_states)
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]
            
            # Calculate target distribution
            target_q_dist = self.compute_target_distribution(rewards, next_q_dist, dones)
        
        # Calculate loss (cross-entropy)
        loss = -(target_q_dist * torch.log(current_q_dist + 1e-8)).sum(1)
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights).mean()
        
        # CRITICAL FIX: Create an array of priorities based on individual losses
        # Instead of using the mean loss, use the individual losses for each sample
        individual_losses = loss.detach().cpu().numpy()
        
        # Ensure priorities are positive and add a small constant
        priorities = individual_losses + 1e-8
        
        # Double-check that priorities is an array matching the length of indices
        if len(priorities) != len(indices):
            print(f"WARNING: Priority length mismatch! Creating array of correct length.")
            # Fallback: create a uniform array if lengths don't match
            priorities = np.ones_like(indices, dtype=np.float32) * 1.0
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, priorities)
        
        # Update Rainbow model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent large updates
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Step the learning rate schedulers periodically
        if hasattr(self, 'scheduler') and self.steps_done % 1000 == 0:
            self.scheduler.step()
            self.rnd_scheduler.step()
        
        # Reset noise for noisy layers
        self.policy_net.reset_noise()
        
        # Learn intrinsic reward predictor
        self.update_rnd(states)
        
        return weighted_loss.item()
    
    def compute_target_distribution(self, rewards, next_q_dist, dones):
        """Compute target distribution for distributional RL."""
        batch_size = rewards.size(0)
        
        # Calculate n-step discounted return for each atom
        # Tz = r + (1-d) * gamma^n * z
        gamma_n = self.gamma ** self.n_steps
        Tz = rewards + (1 - dones) * gamma_n * self.support.unsqueeze(0)
        
        # Clamp to support range
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        
        # Compute distribution projection
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Ensure upper and lower bounds are within valid range
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability mass
        target_q_dist = torch.zeros(batch_size, self.num_atoms, device=self.device)
        
        for i in range(batch_size):
            for j in range(self.num_atoms):
                target_q_dist[i, l[i, j]] += next_q_dist[i, j] * (u[i, j] - b[i, j])
                target_q_dist[i, u[i, j]] += next_q_dist[i, j] * (b[i, j] - l[i, j])
        
        return target_q_dist
    
    def update_rnd(self, states):
        """Update RND predictor network."""
        # Target features are fixed
        target_features, predict_features = self.rnd_model(states)
        
        # Calculate loss (MSE between prediction and target)
        rnd_loss = F.mse_loss(predict_features, target_features.detach())
        
        # Update predictor
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.rnd_model.predictor.parameters(), 10)
        self.rnd_optimizer.step()
        
        return rnd_loss.item()
    
    def save(self, filename):
        """Save the agent's models."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'rnd_state_dict': self.rnd_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'rnd_optimizer_state_dict': self.rnd_optimizer.state_dict(),
            'steps_done': self.steps_done,
            'int_reward_rms_mean': self.int_reward_rms.mean,
            'int_reward_rms_var': self.int_reward_rms.var,
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load the agent's models."""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.rnd_model.load_state_dict(checkpoint['rnd_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.int_reward_rms.mean = checkpoint['int_reward_rms_mean']
        self.int_reward_rms.var = checkpoint['int_reward_rms_var']
        print(f"Model loaded from {filename}")


# Utility class for normalizing observations and rewards
class RunningMeanStd:
    """Tracks the running mean and standard deviation of a data stream."""
    
    def __init__(self, shape=(), epsilon=1e-4):
        """Initialize running mean and std.
        
        Args:
            shape: Shape of the data
            epsilon: Small constant to avoid division by zero
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
    
    def update(self, x):
        """Update running mean and variance."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)
        
        self.mean = new_mean
        self.var = new_var
        self.count += batch_count 