"""
Custom neural network architectures for reinforcement learning.

This module contains implementations of various neural network architectures
for reinforcement learning, specifically tailored for the Obstacle Tower Challenge.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim


class ObstacleTowerCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for Obstacle Tower observations.
    Uses a slightly deeper architecture than the Stable Baselines default.
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
    ):
        """
        Initialize the CNN.
        
        Args:
            observation_space: Environment's observation space
            features_dim: Dimension of the extracted features
        """
        super().__init__(observation_space, features_dim)
        
        # Get input shape
        n_input_channels = observation_space.shape[0]
        
        # CNN architecture
        self.cnn = nn.Sequential(
            # First conv layer
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            
            # Third conv layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            # Additional conv layer for more expressivity
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # Fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        """Extract features from observations"""
        return self.linear(self.cnn(observations))


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections.
    """
    def __init__(self, channels):
        """
        Initialize residual block.
        
        Args:
            channels: Number of input/output channels
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward pass with skip connection"""
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class ResNetExtractor(BaseFeaturesExtractor):
    """
    ResNet-style feature extractor for Obstacle Tower observations.
    Uses residual connections for better gradient flow.
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        n_residual_blocks: int = 2
    ):
        """
        Initialize the ResNet extractor.
        
        Args:
            observation_space: Environment's observation space
            features_dim: Dimension of the extracted features
            n_residual_blocks: Number of residual blocks
        """
        super().__init__(observation_space, features_dim)
        
        # Get input shape
        n_input_channels = observation_space.shape[0]
        
        # Initial convolution
        layers = [
            nn.Conv2d(n_input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        
        # Residual blocks
        layers.append(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        
        for _ in range(n_residual_blocks):
            layers.append(ResidualBlock(64))
        
        # Final layers
        layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        
        self.cnn = nn.Sequential(*layers)
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # Fully connected layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        """Extract features from observations"""
        return self.linear(self.cnn(observations))


class LSTMExtractor(BaseFeaturesExtractor):
    """
    CNN + LSTM feature extractor for handling temporal dependencies.
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1
    ):
        """
        Initialize CNN + LSTM extractor.
        
        Args:
            observation_space: Environment's observation space
            features_dim: Output dimension
            lstm_hidden_size: LSTM hidden state size
            lstm_layers: Number of LSTM layers
        """
        super().__init__(observation_space, features_dim)
        
        # Get input shape
        n_input_channels = observation_space.shape[0]
        
        # CNN for extracting spatial features
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output shape
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=n_flatten,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Final projection
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU()
        )
        
        # Hidden state
        self.hidden = None
    
    def forward(self, observations, hidden=None):
        """
        Extract features with temporal dependency.
        
        Args:
            observations: Batched observations [batch_size, channels, height, width]
            hidden: Optional LSTM hidden state
            
        Returns:
            tuple: (features, new_hidden)
        """
        batch_size = observations.shape[0]
        
        # Extract CNN features
        cnn_features = self.cnn(observations)
        
        # Reshape for LSTM [batch_size, seq_len, features]
        lstm_input = cnn_features.view(batch_size, 1, -1)
        
        # Use provided hidden state or zeros
        if hidden is None:
            hidden = self.get_initial_hidden(batch_size, observations.device)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(lstm_input, hidden)
        
        # Extract features from last timestep
        features = lstm_out[:, -1]
        
        # Final projection
        features = self.linear(features)
        
        return features, new_hidden
    
    def get_initial_hidden(self, batch_size, device):
        """Get initial hidden state for LSTM"""
        h0 = torch.zeros(
            self.lstm.num_layers, batch_size, self.lstm.hidden_size,
            device=device
        )
        c0 = torch.zeros(
            self.lstm.num_layers, batch_size, self.lstm.hidden_size,
            device=device
        )
        return (h0, c0)
    
    def reset_hidden(self):
        """Reset LSTM hidden state"""
        self.hidden = None


class DualEncoderExtractor(BaseFeaturesExtractor):
    """
    Dual encoder architecture that separately processes observed pixels
    and structured game state information.
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        state_dim: int = 8
    ):
        """
        Initialize dual encoder.
        
        Args:
            observation_space: Environment's observation space
            features_dim: Output dimension
            state_dim: Dimension of structured state input
        """
        super().__init__(observation_space, features_dim)
        
        # Get input shape
        n_input_channels = observation_space.shape[0]
        
        # CNN for visual features
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute CNN output shape
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.visual_encoder(sample).shape[1]
        
        # Structured state encoder (for game state like floor, keys, etc.)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined features
        self.combined = nn.Sequential(
            nn.Linear(n_flatten + 64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations, state_info=None):
        """
        Extract features from observations and structured state.
        
        Args:
            observations: Visual observations
            state_info: Optional structured state information
            
        Returns:
            torch.Tensor: Combined features
        """
        # Extract visual features
        visual_features = self.visual_encoder(observations)
        
        # Process state info if provided, otherwise use zeros
        if state_info is None:
            batch_size = observations.shape[0]
            state_features = torch.zeros(
                batch_size, 64,
                device=observations.device
            )
        else:
            state_features = self.state_encoder(state_info)
        
        # Combine features
        combined = torch.cat([visual_features, state_features], dim=1)
        return self.combined(combined)


# Policy networks

class SACPolicy(nn.Module):
    """
    Actor-Critic policy network for SAC.
    """
    def __init__(
        self,
        feature_extractor,
        action_dim,
        features_dim
    ):
        """
        Initialize SAC policy.
        
        Args:
            feature_extractor: Feature extraction module
            action_dim: Dimension of action space
            features_dim: Dimension of extracted features
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(features_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Mean and log std for Gaussian policy
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        
        # Critic (Q-value) networks
        self.critic1 = nn.Sequential(
            nn.Linear(features_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(features_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, observations):
        """Extract features from observations"""
        return self.feature_extractor(observations)
    
    def actor_forward(self, features):
        """
        Compute action distribution parameters.
        
        Args:
            features: Extracted features
            
        Returns:
            tuple: (mean, log_std)
        """
        action_features = self.actor(features)
        mean = self.mean_layer(action_features)
        log_std = self.log_std_layer(action_features)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def critic_forward(self, features, actions):
        """
        Compute Q-values for state-action pairs.
        
        Args:
            features: Extracted features
            actions: Actions to evaluate
            
        Returns:
            tuple: (q1, q2)
        """
        # Concatenate features and actions
        critic_input = torch.cat([features, actions], dim=1)
        
        # Compute Q-values from both critics
        q1 = self.critic1(critic_input)
        q2 = self.critic2(critic_input)
        
        return q1, q2


# Create a registry of available feature extractors
feature_extractors = {
    "cnn": ObstacleTowerCNN,
    "resnet": ResNetExtractor,
    "lstm": LSTMExtractor,
    "dual": DualEncoderExtractor
}


def create_feature_extractor(extractor_name, observation_space, **kwargs):
    """
    Create a feature extractor by name.
    
    Args:
        extractor_name: Name of extractor ("cnn", "resnet", "lstm", "dual")
        observation_space: Environment observation space
        **kwargs: Additional arguments for the extractor
        
    Returns:
        BaseFeaturesExtractor: Instantiated feature extractor
    """
    if extractor_name not in feature_extractors:
        raise ValueError(f"Unknown feature extractor: {extractor_name}")
    
    extractor_class = feature_extractors[extractor_name]
    return extractor_class(observation_space, **kwargs)