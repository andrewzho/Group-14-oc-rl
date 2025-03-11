import numpy as np
from collections import namedtuple, deque
import random
import torch
import math
import operator

# Experience tuple for storing transitions
Experience = namedtuple('Experience', 
                       field_names=['state', 'action', 'reward', 'next_state', 'done'])

class EnhancedKeyDoorMemory:
    """Memory for key-door interactions to aid exploration."""
    
    def __init__(self, decay_factor=0.95, horizon=1000):
        """Initialize memory for key-door locations.
        
        Args:
            decay_factor (float): Factor for decaying importance of old memories
            horizon (int): Number of steps to consider when tracking sequential interactions
        """
        self.key_locations = {}  # Floor -> list of positions
        self.door_locations = {}  # Floor -> list of positions
        self.key_door_sequences = {}  # Floor -> list of (key_pos, door_pos) tuples
        self.successful_floors = set()  # Set of floors successfully completed
        
        self.decay_factor = decay_factor
        self.horizon = horizon
        self.last_key_position = None
        self.steps_since_key = 0
    
    def add_key_location(self, floor, position):
        """Add a key location to memory."""
        if floor not in self.key_locations:
            self.key_locations[floor] = []
        
        # Check if position is already known (approximately)
        if not self._is_position_known(self.key_locations[floor], position):
            self.key_locations[floor].append(position)
            return True
        return False
    
    def add_door_location(self, floor, position):
        """Add a door location to memory."""
        if floor not in self.door_locations:
            self.door_locations[floor] = []
        
        # Check if position is already known (approximately)
        if not self._is_position_known(self.door_locations[floor], position):
            self.door_locations[floor].append(position)
            return True
        return False
    
    def mark_floor_complete(self, floor):
        """Mark a floor as successfully completed."""
        self.successful_floors.add(floor)
    
    def store_key_door_sequence(self, key_pos, door_pos, floor):
        """Store a successful key-door interaction sequence."""
        if floor not in self.key_door_sequences:
            self.key_door_sequences[floor] = []
        
        # Check if sequence is already known
        for k_pos, d_pos in self.key_door_sequences[floor]:
            if (self._distance(k_pos, key_pos) < 2.0 and 
                self._distance(d_pos, door_pos) < 2.0):
                return False
        
        self.key_door_sequences[floor].append((key_pos, door_pos))
        return True
    
    def get_proximity_bonus(self, position, floor, has_key=False, threshold=5.0):
        """Get bonus based on proximity to interesting locations."""
        bonus = 0.0
        
        if has_key:
            # If agent has a key, give bonus for proximity to doors
            if floor in self.door_locations:
                min_dist = float('inf')
                for door_pos in self.door_locations[floor]:
                    dist = self._distance(position, door_pos)
                    min_dist = min(min_dist, dist)
                
                if min_dist < threshold:
                    # Inverse relationship - closer gets higher bonus
                    bonus += 0.1 * (1.0 - min_dist / threshold)
        else:
            # If agent has no key, give bonus for proximity to keys
            if floor in self.key_locations:
                min_dist = float('inf')
                for key_pos in self.key_locations[floor]:
                    dist = self._distance(position, key_pos)
                    min_dist = min(min_dist, dist)
                
                if min_dist < threshold:
                    # Inverse relationship - closer gets higher bonus
                    bonus += 0.1 * (1.0 - min_dist / threshold)
        
        return bonus
    
    def get_directions_to_target(self, position, floor, has_key):
        """Get direction vector to nearest key or door."""
        if has_key and floor in self.door_locations and len(self.door_locations[floor]) > 0:
            # Find closest door
            target_positions = self.door_locations[floor]
            target_type = 'door'
        elif not has_key and floor in self.key_locations and len(self.key_locations[floor]) > 0:
            # Find closest key
            target_positions = self.key_locations[floor]
            target_type = 'key'
        else:
            return None
        
        # Find closest target
        closest_pos = None
        min_dist = float('inf')
        
        for target_pos in target_positions:
            dist = self._distance(position, target_pos)
            if dist < min_dist:
                min_dist = dist
                closest_pos = target_pos
        
        if closest_pos is None:
            return None
        
        # Calculate direction vector
        direction = [
            closest_pos[i] - position[i] for i in range(min(len(position), len(closest_pos)))
        ]
        
        # Normalize direction
        magnitude = math.sqrt(sum(d*d for d in direction))
        if magnitude > 0:
            direction = [d/magnitude for d in direction]
        
        return {
            'type': target_type,
            'position': closest_pos,
            'distance': min_dist,
            'direction': direction
        }
    
    def update_key_detection(self, position, has_key):
        """Update key detection state."""
        if has_key and self.last_key_position is None:
            # Just picked up a key
            self.last_key_position = position
            self.steps_since_key = 0
        elif not has_key and self.last_key_position is not None:
            # Just used a key
            self.last_key_position = None
        
        if self.last_key_position is not None:
            self.steps_since_key += 1
            if self.steps_since_key > self.horizon:
                # Reset if key not used within horizon
                self.last_key_position = None
                self.steps_since_key = 0
    
    def _is_position_known(self, positions, position, threshold=2.0):
        """Check if a position is already known (within threshold)."""
        for known_pos in positions:
            if self._distance(known_pos, position) < threshold:
                return True
        return False
    
    def _distance(self, pos1, pos2):
        """Calculate Euclidean distance between positions."""
        # Handle different length position tuples
        dims = min(len(pos1), len(pos2))
        return math.sqrt(sum((pos1[i] - pos2[i])**2 for i in range(dims)))


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with Segment Tree implementation for O(log n) operations."""
    
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
        self.memory = []
        self.position = 0
        self.size = 0
        
        # Initialize Segment Tree for priorities
        # Power of 2 greater than or equal to buffer_size
        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2
            
        self.tree_capacity = tree_capacity
        self.sum_tree = np.zeros(2 * tree_capacity - 1)
        self.min_tree = np.ones(2 * tree_capacity - 1) * float('inf')
        
        # For normalization of priorities
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with maximum priority."""
        e = Experience(state, action, reward, next_state, done)
        
        # Add to memory
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.position] = e
        
        # Add to tree with max priority
        priority = self.max_priority ** self.alpha
        self._update_tree(self.position, priority)
        
        # Update position
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self):
        """Sample a batch of experiences based on priorities."""
        if self.size < self.batch_size:
            return None
        
        # Anneal beta
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Sample indices based on priorities
        indices = self._sample_proportional()
        
        # Get selected experiences
        experiences = [self.memory[idx] for idx in indices]
        
        # Extract batch components
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
        
        # Calculate importance sampling weights
        weights = self._calculate_weights(indices, beta)
        
        return (states, actions, rewards, next_states, dones, indices, weights)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences."""
        for idx, priority in zip(indices, priorities):
            # Ensure priorities are positive
            priority = max(priority, 1e-8)
            self.max_priority = max(self.max_priority, priority)
            
            # Update segment tree
            self._update_tree(idx, priority ** self.alpha)
    
    def _update_tree(self, idx, priority):
        """Update the segment trees with new priority value."""
        # Get index in the tree
        tree_idx = idx + self.tree_capacity - 1
        
        # Update sum tree
        change = priority - self.sum_tree[tree_idx]
        self.sum_tree[tree_idx] = priority
        
        # Propagate changes through tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.sum_tree[tree_idx] += change
        
        # Update min tree
        tree_idx = idx + self.tree_capacity - 1
        self.min_tree[tree_idx] = priority
        
        # Propagate min value through tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.min_tree[tree_idx] = min(self.min_tree[2 * tree_idx + 1], self.min_tree[2 * tree_idx + 2])
    
    def _sample_proportional(self):
        """Sample indices based on priorities."""
        indices = []
        p_total = self.sum_tree[0]
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # Add some noise for better exploration
            s = random.uniform(a, b)
            
            # Get index
            idx = self._retrieve(0, s)
            indices.append(idx)
        
        return indices
    
    def _retrieve(self, idx, s):
        """Retrieve index from segment tree based on cumulative priority sum."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.sum_tree):
            return idx - (self.tree_capacity - 1)
        
        if s <= self.sum_tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.sum_tree[left])
    
    def _calculate_weights(self, indices, beta):
        """Calculate importance sampling weights."""
        # Get min probability
        p_min = self.min_tree[0] / self.sum_tree[0]
        max_weight = (p_min * self.size) ** (-beta)
        
        # Calculate weights
        weights = []
        for idx in indices:
            tree_idx = idx + self.tree_capacity - 1
            p_sample = self.sum_tree[tree_idx] / self.sum_tree[0]
            weight = (p_sample * self.size) ** (-beta)
            weights.append(weight / max_weight)
        
        return np.array(weights, dtype=np.float32)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size


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
        
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)