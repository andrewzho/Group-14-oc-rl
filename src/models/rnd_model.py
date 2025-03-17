"""
Random Network Distillation (RND) model for intrinsic rewards.

RND works by training a predictor network to match a fixed randomly initialized
target network. The prediction error serves as a measure of state novelty.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional

class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for processing image observations.
    Used as the base network for both RND target and predictor networks.
    """
    def __init__(self, input_shape: Tuple[int, int, int], feature_dim: int, 
                 cnn_features: Dict[str, Any]):
        """
        Args:
            input_shape: Shape of input observations (channels, height, width)
            feature_dim: Output feature dimension
            cnn_features: CNN architecture parameters
        """
        super(CNNFeatureExtractor, self).__init__()
        
        # Build CNN layers based on config
        layers = []
        in_channels = input_shape[0]
        
        for layer_config in cnn_features:
            layers.append(nn.Conv2d(
                in_channels=layer_config["in_channels"],
                out_channels=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config["stride"]
            ))
            layers.append(nn.ReLU())
            in_channels = layer_config["out_channels"]
            
        self.cnn = nn.Sequential(*layers)
        
        # Calculate the flattened feature size
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self.cnn(dummy_input)
        flat_size = int(np.prod(dummy_output.shape[1:]))
        
        # Fully connected output layer
        self.fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        features = self.cnn(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.fc(features)


class RNDModel(nn.Module):
    """
    Random Network Distillation model with target and predictor networks.
    """
    def __init__(self, observation_shape: Tuple[int, int, int], config: Dict[str, Any]):
        """
        Args:
            observation_shape: Shape of input observations (channels, height, width)
            config: RND configuration parameters
        """
        super(RNDModel, self).__init__()
        
        self.feature_dim = config["feature_dim"]
        self.normalization = config["normalization"]
        self.update_proportion = config["update_proportion"]
        
        # Running statistics for observation normalization
        if self.normalization:
            self.obs_rms = RunningMeanStd(shape=observation_shape)
        
        # Target network (fixed random weights)
        self.target = CNNFeatureExtractor(
            input_shape=observation_shape,
            feature_dim=self.feature_dim,
            cnn_features=config.get("cnn_features", [
                {"in_channels": observation_shape[0], "out_channels": 32, "kernel_size": 8, "stride": 4},
                {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2},
                {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1},
            ])
        )
        
        # Predictor network (trained to predict target)
        self.predictor = CNNFeatureExtractor(
            input_shape=observation_shape,
            feature_dim=self.feature_dim,
            cnn_features=config.get("cnn_features", [
                {"in_channels": observation_shape[0], "out_channels": 32, "kernel_size": 8, "stride": 4},
                {"in_channels": 32, "out_channels": 64, "kernel_size": 4, "stride": 2},
                {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1},
            ])
        )
        
        # Initialize target network and freeze weights
        for param in self.target.parameters():
            param.requires_grad = False
            
        # Optimizer for predictor network
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(), 
            lr=config["learning_rate"]
        )
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations if enabled"""
        if self.normalization:
            # Update running mean/std
            if self.training:
                self.obs_rms.update(obs.cpu().numpy())
            
            # Apply normalization
            obs_np = obs.cpu().numpy()
            normalized_obs = np.clip(
                (obs_np - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
                -5.0, 5.0
            )
            return torch.FloatTensor(normalized_obs).to(self.device)
        return obs
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both target and predictor networks
        
        Args:
            obs: Observation tensor of shape (batch_size, channels, height, width)
            
        Returns:
            tuple: (target_features, predictor_features)
        """
        # Ensure input is on the correct device
        obs = obs.to(self.device)
        
        # Normalize if enabled
        if self.normalization:
            obs = self.normalize_obs(obs)
        
        # Generate features from both networks
        with torch.no_grad():
            target_features = self.target(obs)
        
        predictor_features = self.predictor(obs)
        
        return target_features, predictor_features
    
    def calculate_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Calculate intrinsic reward as the prediction error
        
        Args:
            obs: Observation tensor
            
        Returns:
            torch.Tensor: Intrinsic reward values
        """
        with torch.no_grad():
            target_features, predictor_features = self.forward(obs)
            intrinsic_reward = F.mse_loss(
                predictor_features, target_features, reduction='none'
            ).mean(dim=1)
            
        return intrinsic_reward
    
    def update(self, obs: torch.Tensor) -> Dict[str, float]:
        """
        Update the predictor network to better predict the target
        
        Args:
            obs: Batch of observations
            
        Returns:
            dict: Training metrics
        """
        # Forward pass
        target_features, predictor_features = self.forward(obs)
        
        # Compute prediction loss
        loss = F.mse_loss(predictor_features, target_features)
        
        # Update predictor network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"rnd_loss": loss.item()}


class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a data stream.
    Used for observation normalization in RND.
    """
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        """
        Args:
            shape: Shape of the data
            epsilon: Small constant to avoid division by zero
        """
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        
    def update(self, x: np.ndarray) -> None:
        """
        Update running mean and variance with new batch of data
        
        Args:
            x: New batch of data with shape (..., *shape)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
        
    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, 
                            batch_count: int) -> None:
        """
        Update running statistics from batch statistics
        
        Args:
            batch_mean: Mean of the batch
            batch_var: Variance of the batch
            batch_count: Number of samples in the batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count