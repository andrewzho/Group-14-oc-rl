"""
Demonstration buffer for storing and retrieving expert demonstrations.
Used for demonstration-based learning to accelerate training.
"""
import os
import pickle
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque


class DemonstrationBuffer:
    """
    Buffer for storing and sampling expert demonstrations.
    Supports saving/loading demonstrations to/from disk.
    """
    def __init__(self, capacity: int = 100000, device: str = None):
        """
        Initialize the demonstration buffer.

        Args:
            capacity: Maximum number of transitions to store
            device: Device for tensor operations (e.g., 'cpu', 'cuda')
        """
        self.capacity = capacity
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Storage for demonstration transitions
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        
        # Track episodes separately for episode-based sampling
        self.episodes = []
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        
        # Track statistics
        self.total_transitions = 0
        self.num_episodes = 0
        
    def add_transition(self, obs: np.ndarray, action: np.ndarray, 
                      reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Add a single transition to the current episode and to the buffer.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        # Add to current episode
        self.current_episode['observations'].append(obs.copy())
        self.current_episode['actions'].append(action.copy())
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_observations'].append(next_obs.copy())
        self.current_episode['dones'].append(done)
        
        # Also add to flat storage for direct sampling
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.next_observations.append(next_obs.copy())
        self.dones.append(done)
        
        self.total_transitions += 1
        
        # If episode is done, store it and reset current episode
        if done:
            self.episodes.append(self.current_episode)
            self.num_episodes += 1
            self.current_episode = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': []
            }
            
        # Enforce capacity limit
        if self.total_transitions > self.capacity:
            # Remove oldest transitions
            excess = self.total_transitions - self.capacity
            
            self.observations = self.observations[excess:]
            self.actions = self.actions[excess:]
            self.rewards = self.rewards[excess:]
            self.next_observations = self.next_observations[excess:]
            self.dones = self.dones[excess:]
            
            self.total_transitions = self.capacity
            
            # Also update episodes if needed
            while excess > 0 and self.episodes:
                oldest_episode = self.episodes[0]
                episode_length = len(oldest_episode['observations'])
                
                if episode_length <= excess:
                    # Remove entire episode
                    self.episodes.pop(0)
                    self.num_episodes -= 1
                    excess -= episode_length
                else:
                    # Partial removal not supported - just leave it
                    break
    
    def add_episode(self, observations: List[np.ndarray], actions: List[np.ndarray],
                    rewards: List[float], next_observations: List[np.ndarray], 
                    dones: List[bool]) -> None:
        """
        Add a complete episode to the buffer.

        Args:
            observations: List of observations
            actions: List of actions
            rewards: List of rewards
            next_observations: List of next observations
            dones: List of done flags
        """
        assert len(observations) == len(actions) == len(rewards) == len(next_observations) == len(dones), \
            "All input lists must have the same length"
        
        # Add episode to episodes list
        episode = {
            'observations': observations.copy(),
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'next_observations': next_observations.copy(),
            'dones': dones.copy()
        }
        self.episodes.append(episode)
        self.num_episodes += 1
        
        # Also add to flat storage
        self.observations.extend(observations)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_observations.extend(next_observations)
        self.dones.extend(dones)
        
        self.total_transitions += len(observations)
        
        # Enforce capacity limit
        if self.total_transitions > self.capacity:
            excess = self.total_transitions - self.capacity
            
            self.observations = self.observations[excess:]
            self.actions = self.actions[excess:]
            self.rewards = self.rewards[excess:]
            self.next_observations = self.next_observations[excess:]
            self.dones = self.dones[excess:]
            
            self.total_transitions = self.capacity
            
            # Update episodes
            while excess > 0 and self.episodes:
                oldest_episode = self.episodes[0]
                episode_length = len(oldest_episode['observations'])
                
                if episode_length <= excess:
                    # Remove entire episode
                    self.episodes.pop(0)
                    self.num_episodes -= 1
                    excess -= episode_length
                else:
                    # Partial removal not supported
                    break
    
    def sample(self, batch_size: int, prioritize_reward: bool = False, 
               return_tensors: bool = True) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample
            prioritize_reward: Whether to prioritize high-reward transitions
            return_tensors: Whether to return PyTorch tensors (vs numpy arrays)
            
        Returns:
            dict: Batch of transitions
        """
        if self.total_transitions == 0:
            raise ValueError("Cannot sample from an empty buffer")
        
        batch_size = min(batch_size, self.total_transitions)
        
        if prioritize_reward:
            # Prioritize sampling based on rewards
            probs = np.array(self.rewards) - min(self.rewards)
            probs = probs + 1e-6  # Avoid zero probabilities
            probs = probs / probs.sum()
            indices = np.random.choice(self.total_transitions, batch_size, p=probs)
        else:
            # Uniform sampling
            indices = np.random.choice(self.total_transitions, batch_size, replace=False)
        
        obs_batch = [self.observations[i] for i in indices]
        action_batch = [self.actions[i] for i in indices]
        reward_batch = [self.rewards[i] for i in indices]
        next_obs_batch = [self.next_observations[i] for i in indices]
        done_batch = [self.dones[i] for i in indices]
        
        if return_tensors:
            # Convert to tensors
            obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
            action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32, device=self.device)
            reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32, device=self.device)
            next_obs_batch = torch.tensor(np.array(next_obs_batch), dtype=torch.float32, device=self.device)
            done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32, device=self.device)
        
        return {
            'observations': obs_batch,
            'actions': action_batch,
            'rewards': reward_batch,
            'next_observations': next_obs_batch,
            'dones': done_batch
        }
    
    def sample_episodes(self, n_episodes: int) -> List[Dict[str, List]]:
        """
        Sample complete episodes from the buffer.

        Args:
            n_episodes: Number of episodes to sample
            
        Returns:
            list: List of episode dictionaries
        """
        if self.num_episodes == 0:
            raise ValueError("Cannot sample from an empty buffer")
        
        n_episodes = min(n_episodes, self.num_episodes)
        indices = np.random.choice(self.num_episodes, n_episodes, replace=False)
        
        return [self.episodes[i] for i in indices]
    
    def save(self, filepath: str) -> None:
        """
        Save the demonstration buffer to a file.

        Args:
            filepath: Path to save the buffer
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create a dictionary with all buffer data
        buffer_data = {
            'observations': self.observations,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_observations': self.next_observations,
            'dones': self.dones,
            'episodes': self.episodes,
            'total_transitions': self.total_transitions,
            'num_episodes': self.num_episodes
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_data, f)
            
        print(f"Demonstration buffer saved to {filepath}")
    
    def load(self, filepath: str) -> bool:
        """
        Load the demonstration buffer from a file.

        Args:
            filepath: Path to the saved buffer
            
        Returns:
            bool: Success or failure
        """
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                buffer_data = pickle.load(f)
            
            # Load data into buffer
            self.observations = buffer_data['observations']
            self.actions = buffer_data['actions']
            self.rewards = buffer_data['rewards']
            self.next_observations = buffer_data['next_observations']
            self.dones = buffer_data['dones']
            self.episodes = buffer_data['episodes']
            self.total_transitions = buffer_data['total_transitions']
            self.num_episodes = buffer_data['num_episodes']
            
            print(f"Loaded {self.num_episodes} episodes with {self.total_transitions} transitions")
            return True
            
        except Exception as e:
            print(f"Error loading buffer: {e}")
            return False
    
    def is_empty(self) -> bool:
        """Check if the buffer is empty"""
        return self.total_transitions == 0
    
    def clear(self) -> None:
        """Clear the buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
        self.episodes = []
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        self.total_transitions = 0
        self.num_episodes = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the buffer"""
        if self.total_transitions == 0:
            return {
                "total_transitions": 0,
                "num_episodes": 0,
                "avg_episode_length": 0,
                "avg_episode_reward": 0,
                "min_reward": 0,
                "max_reward": 0,
                "mean_reward": 0
            }
        
        # Calculate episode statistics
        episode_lengths = [len(ep['rewards']) for ep in self.episodes]
        episode_returns = [sum(ep['rewards']) for ep in self.episodes]
        
        return {
            "total_transitions": self.total_transitions,
            "num_episodes": self.num_episodes,
            "avg_episode_length": np.mean(episode_lengths),
            "avg_episode_reward": np.mean(episode_returns),
            "min_reward": min(self.rewards),
            "max_reward": max(self.rewards),
            "mean_reward": np.mean(self.rewards)
        }


class DemonstrationRecorder:
    """
    Utility for recording new demonstrations from human or agent play.
    """
    def __init__(self, demo_buffer: DemonstrationBuffer):
        """
        Initialize the recorder.

        Args:
            demo_buffer: Buffer to store the demonstrations
        """
        self.demo_buffer = demo_buffer
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        self.recording = False
        
    def start_recording(self) -> None:
        """Start recording a new episode"""
        self.recording = True
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        print("Started recording demonstration")
    
    def record_step(self, obs: np.ndarray, action: np.ndarray, 
                    reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Record a single step.

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
        """
        if not self.recording:
            return
        
        self.current_episode['observations'].append(obs.copy())
        self.current_episode['actions'].append(action.copy())
        self.current_episode['rewards'].append(reward)
        self.current_episode['next_observations'].append(next_obs.copy())
        self.current_episode['dones'].append(done)
        
        if done:
            self.stop_recording()
    
    def stop_recording(self) -> None:
        """
        Stop recording and add the episode to the buffer.
        """
        if not self.recording:
            return
        
        # Add recorded episode to buffer
        self.demo_buffer.add_episode(
            self.current_episode['observations'],
            self.current_episode['actions'],
            self.current_episode['rewards'],
            self.current_episode['next_observations'],
            self.current_episode['dones']
        )
        
        # Calculate statistics
        episode_length = len(self.current_episode['observations'])
        episode_return = sum(self.current_episode['rewards'])
        
        print(f"Finished recording demonstration with {episode_length} steps and return {episode_return:.2f}")
        self.recording = False
    
    def discard_recording(self) -> None:
        """Discard the current recording without saving"""
        if not self.recording:
            return
        
        self.current_episode = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        self.recording = False
        print("Discarded recording")


# Helper functions for creating demonstrations
def create_demonstration_from_env_interaction(env, policy, n_episodes=1, max_steps=1000):
    """
    Create demonstrations by interacting with the environment using a policy.
    
    Args:
        env: Gym-like environment
        policy: Policy that can select actions given observations
        n_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        
    Returns:
        DemonstrationBuffer: Buffer with recorded demonstrations
    """
    demo_buffer = DemonstrationBuffer()
    recorder = DemonstrationRecorder(demo_buffer)
    
    for _ in range(n_episodes):
        obs = env.reset()
        recorder.start_recording()
        
        for step in range(max_steps):
            action = policy(obs)
            next_obs, reward, done, info = env.step(action)
            
            recorder.record_step(obs, action, reward, next_obs, done)
            
            obs = next_obs
            
            if done:
                break
    
    return demo_buffer