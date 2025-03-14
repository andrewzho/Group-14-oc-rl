import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math  # Add import for math functions
import os

# Add global debug flag that respects verbosity setting
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'

def debug_print(*args, **kwargs):
    """Print only when debug mode is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)

class ICMFeatureEncoder(nn.Module):
    """
    Encoder network for ICM that converts observations to feature representations.
    """
    def __init__(self, input_shape, feature_dim=256):
        super(ICMFeatureEncoder, self).__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        # CNN layers to extract features from observations
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # [12, 84, 84] -> [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [32, 20, 20] -> [64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # [64, 9, 9] -> [64, 7, 7]
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate output size of convolutional layers
        conv_out_size = self._get_conv_output(input_shape)
        
        # Linear layer to get final feature representation
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
    
    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers."""
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv(input)
            return int(np.prod(output.size()))
    
    def forward(self, x):
        """
        Convert observations to feature representations.
        
        Args:
            x: Observation tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Feature representation of shape [batch_size, feature_dim]
        """
        # Ensure input has correct shape
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # Normalize input to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.conv(x)
        return self.fc(features)


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM) for exploration bonuses.
    
    The ICM consists of:
    1. A feature encoder network that converts observations to feature representations
    2. A forward model that predicts the next feature representation given the current
       feature representation and action
    3. An inverse model that predicts the action given the current and next feature representations
    
    The intrinsic reward is the error in the forward model prediction, indicating how novel
    or surprising the next state is to the agent.
    """
    def __init__(self, input_shape, action_dim, feature_dim=256, 
                 forward_scale=0.2, inverse_scale=0.8):
        super(ICM, self).__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.forward_scale = forward_scale  # Weight for forward loss
        self.inverse_scale = inverse_scale  # Weight for inverse loss
        
        # Feature encoder network
        self.feature_encoder = ICMFeatureEncoder(input_shape, feature_dim)
        
        # Forward model: predicts next feature representation from current feature + action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        # Inverse model: predicts action from current and next feature representations
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # For normalizing intrinsic rewards
        self.reward_scale = 1.0
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0.001  # Small non-zero value to avoid division by zero
    
    def forward(self, state, next_state, action):
        """
        Forward pass through the ICM.
        
        Args:
            state: Current observation tensor of shape [batch_size, channels, height, width]
            next_state: Next observation tensor of shape [batch_size, channels, height, width]
            action: Action tensor of shape [batch_size, action_dim] or [batch_size]
            
        Returns:
            forward_loss: Loss of the forward model
            inverse_loss: Loss of the inverse model
            intrinsic_reward: Exploration bonus for each state-action pair
        """
        # Ensure actions are in one-hot format for continuous value operations
        if action.dim() == 1:
            action_one_hot = torch.zeros(action.size(0), self.action_dim, 
                                         device=action.device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        else:
            action_one_hot = action
        
        # Encode states into feature representations
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Inverse model: predict action from state and next state features
        inverse_input = torch.cat([state_feat, next_state_feat], dim=1)
        action_pred = self.inverse_model(inverse_input)
        
        # Forward model: predict next state features from state features and action
        forward_input = torch.cat([state_feat, action_one_hot], dim=1)
        next_state_feat_pred = self.forward_model(forward_input)
        
        # Calculate forward loss (prediction error for next state features)
        forward_loss = F.mse_loss(next_state_feat_pred, next_state_feat.detach(), 
                                 reduction='none').mean(dim=1)
        
        # Calculate inverse loss (action prediction error)
        if action.dim() == 1:
            inverse_loss = F.cross_entropy(action_pred, action, reduction='none')
        else:
            inverse_loss = F.mse_loss(action_pred, action_one_hot, reduction='none').sum(dim=1)
        
        # Intrinsic reward is the error in predicting the next state features
        # This incentivizes exploration of novel states
        intrinsic_reward = forward_loss.detach()
        
        # Normalize intrinsic rewards
        self._update_reward_normalization(intrinsic_reward)
        normalized_reward = self._normalize_reward(intrinsic_reward)
        
        return {
            'forward_loss': forward_loss.mean(),
            'inverse_loss': inverse_loss.mean(),
            'intrinsic_reward': normalized_reward,
            'raw_reward': intrinsic_reward
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
        try:
            with torch.no_grad():
                results = self.forward(state, next_state, action)
                return results['intrinsic_reward']
        except Exception as e:
            debug_print(f"Warning: Error computing intrinsic reward: {e}")
            # Return zero reward on error to avoid breaking training
            if isinstance(state, torch.Tensor):
                return torch.zeros(state.size(0), device=state.device)
            else:
                return 0.0
    
    def _update_reward_normalization(self, rewards):
        """Update running statistics for normalizing rewards."""
        try:
            batch_mean = rewards.mean().item()
            batch_var = max(0.0, rewards.var().item())  # Ensure variance is non-negative
            batch_count = rewards.size(0)
            
            # Check for NaN or Inf values
            if math.isnan(batch_mean) or math.isinf(batch_mean) or math.isnan(batch_var) or math.isinf(batch_var):
                debug_print("Warning: NaN or Inf detected in reward statistics, skipping update")
                return
            
            # Update running statistics using Welford's online algorithm
            delta = batch_mean - self.running_mean
            self.running_mean += delta * batch_count / (self.count + batch_count)
            
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
            self.running_var = M2 / (self.count + batch_count)
            
            self.count += batch_count
        except Exception as e:
            debug_print(f"Warning: Error updating reward normalization: {e}")
    
    def _normalize_reward(self, reward, clip=5.0):
        """Normalize rewards using running statistics."""
        try:
            # Check for NaN values
            if torch.isnan(reward).any():
                debug_print("Warning: NaN rewards detected, replacing with zeros")
                reward = torch.where(torch.isnan(reward), torch.zeros_like(reward), reward)
            
            # Avoid division by zero or negative variance
            safe_std = max(1e-8, math.sqrt(max(1e-8, self.running_var)))
            
            normalized = reward / safe_std
            
            # Clip to prevent extremely large values
            normalized = torch.clamp(normalized, -clip, clip)
            
            # Scale the rewards to a reasonable range
            return normalized * self.reward_scale
        except Exception as e:
            debug_print(f"Warning: Error normalizing reward: {e}")
            # Return zero reward on error to avoid breaking training
            return torch.zeros_like(reward)
    
    def set_reward_scale(self, scale):
        """Set the scale factor for intrinsic rewards."""
        self.reward_scale = scale 