"""
Enhanced Obstacle Tower training script with improved network architecture,
experience replay, intrinsic rewards, and adaptive learning rates.
"""

import os
import argparse
import datetime
import random
import collections
import time
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.global_settings import IMAGE_SIZE, IMAGE_DEPTH, STATE_SIZE, STATE_STACK
from src.reinforcement_learner import ProximalPolicyTrainer
from src.helper_funcs import create_parallel_envs, atomic_save
from src.data_collector import TrajectoryCollector
from src.reinforcement_learner import LoggingProximalPolicyTrainer
from src.data_collector import TensorboardTrajectoryCollector

if not hasattr(np, "bool"):
    np.bool = np.bool_ 

class AdvancedCNNModel(nn.Module):
    """Enhanced CNN policy network with greater capacity and batch normalization."""
    def __init__(self, input_shape, action_dim):
        super(AdvancedCNNModel, self).__init__()
        # Wider convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=8, stride=4)  # 32 -> 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2)  # 64 -> 128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1)  # 64 -> 128
        
        # Add batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        conv_output_size = self._get_conv_output(input_shape)
        
        # Deeper and wider fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 1024)  # 512 -> 1024
        self.fc2 = nn.Linear(1024, 512)  # Additional layer
        
        self.action_head = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers."""
        o = self.bn1(self.conv1(torch.zeros(1, *shape)))
        o = self.bn2(self.conv2(F.relu(o)))
        o = self.bn3(self.conv3(F.relu(o)))
        return int(np.prod(o.size()))

    def forward(self, memory_state, obses):
        """
        Forward pass through the network.
        
        Args:
            memory_state: State features
            obses: Visual observations
        
        Returns:
            Dictionary with 'actor' and 'critic' outputs
        """
        # We're only using the observation input for this CNN policy
        
        # Make sure obses is a tensor
        if not isinstance(obses, torch.Tensor):
            obses = self.tensor(obses)
            
        # Transpose from [batch, height, width, channels] to [batch, channels, height, width]
        if obses.dim() == 4 and obses.shape[3] in [3, 6]:
            obses = obses.permute(0, 3, 1, 2)
        
        # Process observation through CNN with batch normalization
        x = F.relu(self.bn1(self.conv1(obses)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten and process through fully connected layers
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return {
            'actor': self.action_head(x),
            'critic': self.value_head(x)
        }
    
    def step(self, memory_state, obs):
        """
        Process observations and return actions, values, and log probabilities.
        """
        # Convert numpy arrays to torch tensors
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.fc1.weight.device)
        if isinstance(memory_state, np.ndarray):
            memory_state = torch.from_numpy(memory_state).float().to(self.fc1.weight.device)
        
        # Handle batched inputs - reshape if needed
        if len(obs.shape) == 3:  # Add batch dimension if missing
            obs = obs.unsqueeze(0)
            
        # Reshape to expected format for CNN (batch, channels, height, width)
        if obs.shape[-1] in [3, 6]:  # If channels are last dimension
            obs = obs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
        # Forward pass through the network - notice we're passing both memory_state and obs now
        with torch.no_grad():  # No need to track gradients for inference
            outputs = self.forward(memory_state, obs)
        
        # Get action distribution
        action_logits = outputs['actor']
        
        # Sample actions using the distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        
        # Calculate log probabilities
        log_probs = -F.cross_entropy(action_logits, actions, reduction='none')
        
        return {
            'actions': actions.cpu().numpy(),
            'log_probs': log_probs.detach().cpu().numpy(),
            'critic': outputs['critic'].detach().cpu().numpy().squeeze(),
            'actor': action_logits.detach().cpu().numpy()
        }

    def tensor(self, np_array):
        """Convert a numpy array to a tensor on the correct device."""
        return torch.from_numpy(np_array).float().to(self.fc1.weight.device)

# Standard CNN policy for backward compatibility
class BaseCNNModel(nn.Module):
    """CNN-based policy network for visual observations."""
    def __init__(self, input_shape, action_dim):
        super(BaseCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_output_size = self._get_conv_output(input_shape)
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.action_head = nn.Linear(512, action_dim)
        self.value_head = nn.Linear(512, 1)

    def _get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(F.relu(o))
        o = self.conv3(F.relu(o))
        return int(np.prod(o.size()))

    def forward(self, memory_state, obses):
        """Forward pass through the network."""
        # Make sure obses is a tensor
        if not isinstance(obses, torch.Tensor):
            obses = self.tensor(obses)
            
        # Transpose from [batch, height, width, channels] to [batch, channels, height, width]
        if obses.dim() == 4 and obses.shape[3] in [3, 6]:
            obses = obses.permute(0, 3, 1, 2)
        
        # Process observation through CNN
        x = F.relu(self.conv1(obses))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Use reshape instead of view
        x = F.relu(self.fc1(x))
        
        return {
            'actor': self.action_head(x),
            'critic': self.value_head(x)
        }
    
    def step(self, memory_state, obs):
        """Process observations and return actions, values, and log probabilities."""
        # Convert numpy arrays to torch tensors
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.fc1.weight.device)
        if isinstance(memory_state, np.ndarray):
            memory_state = torch.from_numpy(memory_state).float().to(self.fc1.weight.device)
        
        # Handle batched inputs - reshape if needed
        if len(obs.shape) == 3:  # Add batch dimension if missing
            obs = obs.unsqueeze(0)
            
        # Reshape to expected format for CNN (batch, channels, height, width)
        if obs.shape[-1] in [3, 6]:  # If channels are last dimension
            obs = obs.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
            
        # Forward pass through the network - notice we're passing both memory_state and obs now
        with torch.no_grad():  # No need to track gradients for inference
            outputs = self.forward(memory_state, obs)
        
        # Get action distribution
        action_logits = outputs['actor']
        
        # Sample actions using the distribution
        action_dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_dist.sample()
        
        # Calculate log probabilities
        log_probs = -F.cross_entropy(action_logits, actions, reduction='none')
        
        return {
            'actions': actions.cpu().numpy(),
            'log_probs': log_probs.detach().cpu().numpy(),
            'critic': outputs['critic'].detach().cpu().numpy().squeeze(),
            'actor': action_logits.detach().cpu().numpy()
        }

    def tensor(self, np_array):
        """Convert a numpy array to a tensor on the correct device."""
        return torch.from_numpy(np_array).float().to(self.fc1.weight.device)

class ReplayMemory:
    """Buffer for storing and sampling experience tuples."""
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity
        
    def add(self, state, obs, action, reward, next_state, next_obs, done):
        """Add an experience tuple to the buffer."""
        self.buffer.append((state, obs, action, reward, next_state, next_obs, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences from the buffer."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

class ExplorationBonus:
    """Tracks state visitations to calculate novelty bonuses."""
    def __init__(self, bonus_scale=0.01, hash_function=None):
        self.state_counts = {}
        self.bonus_scale = bonus_scale
        self.hash_function = hash_function or (lambda x: hash(tuple(x.reshape(-1).tolist())))
        
    def get_bonus(self, state):
        """Calculate a bonus reward based on state novelty."""
        state_key = self.hash_function(state)
        count = self.state_counts.get(state_key, 0)
        self.state_counts[state_key] = count + 1
        
        # Calculate bonus (higher for less-visited memory_state)
        bonus = self.bonus_scale / np.sqrt(count + 1)
        return bonus

class DynamicEntropyProximalPolicyTrainer(LoggingProximalPolicyTrainer):
    """ProximalPolicyTrainer with adaptive entropy and learning rate scheduling."""
    def __init__(self, model, epsilon=0.2, gamma=0.99, lam=0.95, lr=1e-4, 
                 ent_reg=0.01, log_dir='runs/tower_agent',
                 use_novelty=False, novelty_scale=0.01,
                 lr_patience=100, lr_factor=0.7, min_lr=1e-6):
        super().__init__(model, epsilon, gamma, lam, lr, ent_reg, log_dir)
        
        # Adaptive entropy parameters
        self.base_ent_reg = ent_reg
        self.last_max_floor = 0
        self.steps_since_improvement = 0
        self.max_entropy = ent_reg * 3
        self.min_entropy = ent_reg / 5
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=lr_factor, 
            patience=lr_patience, verbose=True, min_lr=min_lr
        )
        
        # Novelty detection
        self.use_novelty = use_novelty
        self.novelty_detector = ExplorationBonus(bonus_scale=novelty_scale)
        
        # Experience replay
        self.experience_buffer = ReplayMemory()
        
        # Progress tracking
        self.global_step = 0
        self.training_start_time = time.time()
        
    def outer_loop(self, data_collector, save_path='save.pkl', **kwargs):
        """Run training with adaptive parameters and TensorBoard logging."""
        for i in itertools.count():
            self.global_step = i
            
            # Get a trajectory_store from the environment
            trajectory_store = data_collector.trajectory_store()
            
            # Update the experience buffer with transitions from the trajectory_store
            if len(self.experience_buffer) < self.experience_buffer.capacity:
                self._add_trajectory_store_to_buffer(trajectory_store)
            
            # Add novelty bonuses to rewards if enabled
            if self.use_novelty:
                trajectory_store = self._add_novelty_bonuses(trajectory_store)
            
            # Run ProximalPolicyTrainer updates on the trajectory_store
            terms, last_terms = self.inner_loop(trajectory_store, **kwargs)
            
            # Adapt entropy regularization based on progress
            self._adapt_entropy(data_collector)
            
            # Print training progress
            self.print_outer_loop(i, terms, last_terms)
            
            # Adjust learning rate based on performance
            if hasattr(data_collector, 'floor_info') and data_collector.floor_info:
                current_max_floor = data_collector.floor_info.get('max_floor', 0)
                self.scheduler.step(current_max_floor)
            
            # Log metrics to TensorBoard
            self._log_metrics(i, terms, last_terms, data_collector)
            
            # Save model periodically
            if i % 100 == 0:
                # Save a numbered backup every 1000 steps
                if i % 1000 == 0:
                    backup_path = f"{save_path.split('.')[0]}_{i}.pkl"
                    atomic_save(self.model.state_dict(), backup_path)
                    print(f"Saved backup model to {backup_path}")
                
                # Always save latest
                atomic_save(self.model.state_dict(), save_path)
                print(f"Saved model to {save_path}")

    def _add_trajectory_store_to_buffer(self, trajectory_store):
        """Add experiences from trajectory_store to the experience buffer."""
        for t in range(trajectory_store.num_steps):
            for b in range(trajectory_store.batch_size):
                # Get current state and observation
                state = trajectory_store.memory_state[t, b]
                obs = trajectory_store.obses[t, b]
                
                # Get action and reward
                action = trajectory_store.actions()[t, b]
                reward = trajectory_store.rews[t, b]
                
                # Get next state and observation
                next_state = trajectory_store.memory_state[t+1, b]
                next_obs = trajectory_store.obses[t+1, b]
                
                # Get done flag
                done = trajectory_store.dones[t+1, b]
                
                # Add to buffer
                self.experience_buffer.add(state, obs, action, reward, next_state, next_obs, done)
    
    def _add_novelty_bonuses(self, trajectory_store):
        """Add novelty bonuses to rewards in the trajectory_store."""
        trajectory_store = trajectory_store.copy()
        trajectory_store.rews = trajectory_store.rews.copy()
        
        for t in range(trajectory_store.num_steps):
            for b in range(trajectory_store.batch_size):
                # Calculate novelty bonus from state representation
                state_features = trajectory_store.memory_state[t, b, -1]  # Use the most recent state in the stack
                bonus = self.novelty_detector.get_bonus(state_features)
                
                # Add bonus to reward
                trajectory_store.rews[t, b] += bonus
        
        return trajectory_store
    
    def _adapt_entropy(self, data_collector):
        """Adapt entropy regularization based on progress."""
        # Check if there's been floor progress
        if hasattr(data_collector, 'floor_info') and data_collector.floor_info:
            current_max_floor = data_collector.floor_info.get('max_floor', 0)
            
            # Update tracking statistics
            if not hasattr(self, 'floors_completed'):
                self.floors_completed = []
            if not hasattr(self, 'episode_count'):
                self.episode_count = 0
                
            # Update floors completed and episode count
            if data_collector.floor_info.get('new_floors', 0) > 0:
                self.floors_completed.append(data_collector.floor_info['new_floors'])
                self.episode_count += data_collector.floor_info.get('episodes', 0)
            
            # Adapt entropy based on floor progress
            if current_max_floor > self.last_max_floor:
                # Floor progress - reduce entropy to exploit current policy
                self.steps_since_improvement = 0
                self.ent_reg = max(self.ent_reg * 0.9, self.min_entropy)
                print(f"Floor progress! Reducing entropy to {self.ent_reg:.6f}")
                self.last_max_floor = current_max_floor
            else:
                # No progress - consider increasing entropy
                self.steps_since_improvement += 1
                
                # Increase entropy if stuck for a while
                if self.steps_since_improvement >= 200:
                    self.ent_reg = min(self.ent_reg * 1.2, self.max_entropy)
                    print(f"No progress for {self.steps_since_improvement} steps. "
                        f"Increasing entropy to {self.ent_reg:.6f}")
                    self.steps_since_improvement = 0
    
    def _log_metrics(self, step, terms, last_terms, data_collector):
        """Log training metrics to TensorBoard."""
        # Basic ProximalPolicyTrainer metrics
        self.writer.add_scalar('Training/Loss', last_terms['loss'], step)
        self.writer.add_scalar('Training/ClipFraction', last_terms['clip_frac'], step)
        self.writer.add_scalar('Training/Entropy', last_terms['entropy'], step)
        self.writer.add_scalar('Training/ExplainedVariance', last_terms['explained'], step)
        
        # Learning parameters
        self.writer.add_scalar('Training/EntropyCoefficient', self.ent_reg, step)
        self.writer.add_scalar('Training/LearningRate', self.optimizer.param_groups[0]['lr'], step)
        
        # Training progress
        elapsed_time = time.time() - self.training_start_time
        self.writer.add_scalar('Training/ElapsedTimeHours', elapsed_time / 3600, step)
        self.writer.add_scalar('Training/StepsPerSecond', step / max(1, elapsed_time), step)
        
        # Floor information
        if hasattr(data_collector, 'floor_info') and data_collector.floor_info:
            current_max_floor = data_collector.floor_info.get('max_floor', 0)
            self.writer.add_scalar('Environment/MaxFloor', current_max_floor, step)
            
            if hasattr(self, 'floors_completed') and self.floors_completed:
                avg_floors = sum(self.floors_completed) / len(self.floors_completed)
                self.writer.add_scalar('Environment/AvgFloorsPerEpisode', avg_floors, step)
            
            if hasattr(self, 'episode_count'):
                self.writer.add_scalar('Environment/EpisodeCount', self.episode_count, step)

    def print_outer_loop(self, i, terms, last_terms):
        """Print training progress with detailed metrics."""
        print('step %d: clipped=%f entropy=%f explained=%f' %
            (i, last_terms['clip_frac'], terms['entropy'], terms['explained']))
        
        # Print floor information and learning parameters
        floor_info = getattr(self, 'floor_info', {})
        episodes = getattr(self, 'episode_count', 0)
        
        avg_floors = 0
        if hasattr(self, 'floors_completed') and self.floors_completed:
            avg_floors = sum(self.floors_completed) / len(self.floors_completed)
            
        print(f"  max floor: {self.last_max_floor}, avg floors per episode: {avg_floors:.2f}, episodes: {episodes}")
        print(f"  entropy coef: {self.ent_reg:.6f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}, steps since improvement: {self.steps_since_improvement}")
class AdvancedTrajectoryBatchManager(TensorboardTrajectoryCollector):
    """Enhanced data_collector with improved tracking and longer horizons."""
    def __init__(self, parallel_envs, model, num_steps):
        super().__init__(parallel_envs, model, num_steps)
        
        # Enhanced tracking
        self.episode_lengths = []
        self.episode_returns = []
        self.cumulative_rewards = np.zeros(parallel_envs.num_envs)
        self.steps_at_floor = collections.defaultdict(int)
        
    def trajectory_store(self):
        """Run a trajectory_store and collect experiences with enhanced tracking."""
        result = super().trajectory_store()
        
        # Track floor-specific information
        for t in range(result.num_steps):
            for b in range(result.batch_size):
                if 'current_floor' in result.infos[t][b]:
                    floor = result.infos[t][b]['current_floor']
                    self.steps_at_floor[floor] += 1
                
                # Track returns
                self.cumulative_rewards[b] += result.rews[t, b]
                
                # If episode ended, record stats
                if result.dones[t+1, b]:
                    self.episode_returns.append(self.cumulative_rewards[b])
                    self.cumulative_rewards[b] = 0
        
        return result

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Enhanced ProximalPolicyTrainer Agent")
    
    # Environment settings
    parser.add_argument('--env_path', type=str, required=True, 
                      help='Path to the Obstacle Tower environment')
    parser.add_argument('--num_envs', type=int, default=8, 
                      help='Number of parallel environments')
    
    # TrajectoryBatch settings
    parser.add_argument('--num_steps', type=int, default=256,  # Longer trajectory_stores (was 128)
                      help='Number of timesteps per trajectory_store')
    
    # Model architecture
    parser.add_argument('--use_enhanced_network', action='store_true',
                      help='Use the enhanced network architecture')
    
    # ProximalPolicyTrainer hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,  # Higher default LR (was 1e-4)
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.995,  # Higher discount (was 0.99)
                      help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, 
                      help='GAE lambda')
    parser.add_argument('--clip_eps', type=float, default=0.2, 
                      help='Clipping epsilon')
    parser.add_argument('--ent_reg', type=float, default=0.01,  # Higher entropy (was 0.001)
                      help='Entropy regularization coefficient')
    parser.add_argument('--ent_boost', type=float, default=None, 
                      help='Boost entropy regularization (temporary) to increase exploration')
    
    # Adaptive learning settings
    parser.add_argument('--lr_patience', type=int, default=100,
                      help='Patience before reducing learning rate')
    parser.add_argument('--lr_factor', type=float, default=0.7,
                      help='Factor to reduce learning rate by')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate')
    
    # Experience replay
    parser.add_argument('--use_experience_replay', action='store_true',
                      help='Use experience replay')
    parser.add_argument('--replay_buffer_size', type=int, default=10000,
                      help='Size of experience replay buffer')
    
    # Intrinsic rewards
    parser.add_argument('--use_intrinsic_rewards', action='store_true',
                      help='Use intrinsic rewards for exploration')
    parser.add_argument('--novelty_scale', type=float, default=0.01,
                      help='Scale factor for novelty bonuses')
    
    # Training settings
    parser.add_argument('--worker_start', type=int, default=0,
                      help='Starting worker ID for environments')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cpu or cuda)')
    parser.add_argument('--save_path', type=str, default='./save8.pkl',
                      help='Path to save model checkpoint')
    parser.add_argument('--load_path', type=str, default=None,
                      help='Path to load model checkpoint from')
    parser.add_argument('--save_interval', type=int, default=100,
                      help='Steps between saving checkpoints')
    parser.add_argument('--backup_interval', type=int, default=1000,
                      help='Steps between saving numbered backup checkpoints')
    
    return parser.parse_args()


def main():
    """Main training function with enhanced features."""
    args = parse_args()
    device = torch.device(args.device)

    # Set environment path for use in `create_parallel_envs`
    os.environ['OBS_TOWER_PATH'] = args.env_path

    # Create a batched environment using online solution helper_funcsities
    # Use the worker_start parameter to avoid port conflicts
    parallel_envs = create_parallel_envs(num_envs=args.num_envs, start=args.worker_start)

    # Get observation space shape and format it for CNN input (C, H, W)
    obs_shape = parallel_envs.observation_space.shape
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # Convert (H, W, C) → (C, H, W)
    action_dim = parallel_envs.action_space.n

    # Choose network architecture
    if args.use_enhanced_network:
        print("Using enhanced network architecture")
        model = AdvancedCNNModel(input_shape, action_dim).to(device)
    else:
        print("Using standard network architecture")
        model = BaseCNNModel(input_shape, action_dim).to(device)

    # Load model if specified
    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=device))
        print("Model loaded successfully!")

    # Use entropy boost if specified
    ent_reg = args.ent_boost if args.ent_boost is not None else args.ent_reg
    if args.ent_boost is not None:
        print(f"Using boosted entropy regularization: {ent_reg} (original: {args.ent_reg})")

    # Initialize TensorBoard logging
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'tower_agent_enhanced_{current_time}')

    # Create the ProximalPolicyTrainer instance (standard or adaptive)
    if args.use_intrinsic_rewards:
        print("Using adaptive entropy ProximalPolicyTrainer with intrinsic rewards")
        ppo = DynamicEntropyProximalPolicyTrainer(
            model,
            epsilon=args.clip_eps,
            gamma=args.gamma,
            lam=args.lam,
            lr=args.lr,
            ent_reg=ent_reg,
            log_dir=log_dir,
            use_novelty=True,
            novelty_scale=args.novelty_scale,
            lr_patience=args.lr_patience,
            lr_factor=args.lr_factor,
            min_lr=args.min_lr
        )
    else:
        print("Using standard TensorBoard ProximalPolicyTrainer")
        ppo = LoggingProximalPolicyTrainer(
            model,
            epsilon=args.clip_eps,
            gamma=args.gamma,
            lam=args.lam,
            lr=args.lr,
            ent_reg=ent_reg,
            log_dir=log_dir
        )
    
    # Add save intervals to the ProximalPolicyTrainer instance
    ppo.save_interval = args.save_interval
    ppo.backup_interval = args.backup_interval
    
    # Create enhanced data_collector
    print(f"Using trajectory_store length of {args.num_steps} steps")
    data_collector = AdvancedTrajectoryBatchManager(parallel_envs, model, num_steps=args.num_steps)

    # Print training configuration
    print("\n=== Training Configuration ===")
    print(f"Network: {'Enhanced' if args.use_enhanced_network else 'Standard'}")
    print(f"TrajectoryBatch length: {args.num_steps} steps")
    print(f"Discount factor (gamma): {args.gamma}")
    print(f"Learning rate: {args.lr}")
    print(f"Entropy regularization: {ent_reg}")
    print(f"Intrinsic rewards: {'Enabled' if args.use_intrinsic_rewards else 'Disabled'}")
    print(f"Experience replay: {'Enabled' if args.use_experience_replay else 'Disabled'}")
    print(f"Parallel environments: {args.num_envs}")
    print(f"Device: {args.device}")
    print("===========================\n")

    # Print TensorBoard viewing instructions
    print(f"TensorBoard logs are being written to {log_dir}")
    print("To view training progress, run:")
    print(f"tensorboard --logdir={os.path.dirname(log_dir)}")
    
    # Print resume information if loading a model
    if args.load_path:
        print(f"Continuing training from checkpoint: {args.load_path}")
        print(f"New checkpoints will be saved to: {args.save_path}")
    
    # Start training loop with keyboard interrupt handling
    try:
        ppo.outer_loop(data_collector, save_path=args.save_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Saving final model to {args.save_path}...")
        atomic_save(model.state_dict(), args.save_path)
        print("Model saved successfully!")


if __name__ == '__main__':
    main()