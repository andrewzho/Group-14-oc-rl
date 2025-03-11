import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Factorized Noisy Linear Layer for better exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

# Residual block for deep networks
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# Rainbow DQN Network with distributional RL (C51)
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51, Vmin=-10, Vmax=10, 
                 sigma_init=0.5, hidden_size=512, noisy=True):
        """
        Rainbow DQN Network combining multiple improvements:
        - Dueling architecture
        - Distributional RL (C51)
        - Noisy Networks
        
        Args:
            input_shape: Shape of input observations (C, H, W)
            num_actions: Number of possible actions
            num_atoms: Number of atoms for distributional RL
            Vmin: Minimum value of support
            Vmax: Maximum value of support
            sigma_init: Initial value of noise standard deviation
            hidden_size: Size of hidden layer
            noisy: Whether to use noisy networks
        """
        super(RainbowDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.noisy = noisy
        
        # Calculate support for distributional RL
        self.register_buffer('supports', torch.linspace(Vmin, Vmax, num_atoms))
        self.delta_z = (Vmax - Vmin) / (num_atoms - 1)
        
        # CNN feature extraction layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Add residual blocks for deeper representations
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        
        # Calculate size of convolution output
        conv_output_size = self._get_conv_output(input_shape)
        
        # Value stream (noisy or regular)
        if noisy:
            self.value_fc = NoisyLinear(conv_output_size, hidden_size, sigma_init)
            self.value = NoisyLinear(hidden_size, num_atoms, sigma_init)
        else:
            self.value_fc = nn.Linear(conv_output_size, hidden_size)
            self.value = nn.Linear(hidden_size, num_atoms)
        
        # Advantage stream (noisy or regular)
        if noisy:
            self.advantage_fc = NoisyLinear(conv_output_size, hidden_size, sigma_init)
            self.advantage = NoisyLinear(hidden_size, num_actions * num_atoms, sigma_init)
        else:
            self.advantage_fc = nn.Linear(conv_output_size, hidden_size)
            self.advantage = nn.Linear(hidden_size, num_actions * num_atoms)
    
    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers"""
        # Create a temporary Sequential module with just the convolutional layers
        # to avoid calling self.target before it's defined
        temp_conv = nn.Sequential(
            nn.Conv2d(shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Pass a dummy input through the temporary network
        o = torch.zeros(1, *shape)
        o = temp_conv(o)
        return int(np.prod(o.size()))
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        return x
    
    def forward(self, x):
        """Forward pass through the network"""
        batch_size = x.size(0)
        
        # Extract features with CNN
        x = self._forward_conv(x)
        x = x.reshape(batch_size, -1)
        
        # Dueling architecture
        value = self.value_fc(x)
        value = self.value(F.relu(value))
        
        advantage = self.advantage_fc(x)
        advantage = self.advantage(F.relu(advantage))
        
        # Reshape value and advantage to match dimensions
        value = value.reshape(batch_size, 1, self.num_atoms)
        advantage = advantage.reshape(batch_size, self.num_actions, self.num_atoms)
        
        # Combine value and advantage (dueling architecture)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Get probabilities with softmax
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        """Reset noise for all noisy layers"""
        if not self.noisy:
            return
            
        # Reset noise for value stream
        self.value_fc.reset_noise()
        self.value.reset_noise()
        
        # Reset noise for advantage stream
        self.advantage_fc.reset_noise()
        self.advantage.reset_noise()
    
    def act(self, state, epsilon=0.0):
        """Select action with epsilon-greedy policy"""
        if np.random.random() < epsilon:
            # Random action
            return np.random.randint(self.num_actions)
        else:
            # Greedy action based on Q-values
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                if next(self.parameters()).is_cuda:
                    state = state.cuda()
                
                # Get probability distribution
                dist = self.forward(state)
                
                # Calculate expected value: sum(p(x) * x)
                expected_value = dist * self.supports.expand_as(dist)
                expected_value = expected_value.sum(2)
                
                # Get action with highest expected value
                action = expected_value.max(1)[1].item()
                return action
                
    def get_q_values(self, state):
        """Get Q-values for a state"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                state = state.cuda()
            
            # Get probability distribution
            dist = self.forward(state)
            
            # Calculate expected value: sum(p(x) * x)
            expected_value = dist * self.supports.expand_as(dist)
            expected_value = expected_value.sum(2)
            
            return expected_value


# Random Network Distillation (RND) for intrinsic motivation
class RNDModel(nn.Module):
    def __init__(self, input_shape, output_size=512):
        """
        Random Network Distillation model for exploration bonuses.
        Contains:
        1. Target network (fixed random weights)
        2. Predictor network (learns to predict target network's output)
        
        Args:
            input_shape: Shape of input observations (C, H, W)
            output_size: Size of embedding vector
        """
        super(RNDModel, self).__init__()
        
        # Calculate convolutional output size first
        conv_output_size = self._get_conv_output(input_shape)
        
        # Target Network (fixed weights, produces random features)
        self.target = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        
        # Predictor Network (learns to predict target)
        self.predictor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        
        # Initialize target network with random weights and freeze
        for param in self.target.parameters():
            param.requires_grad = False
    
    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers"""
        # Create a temporary Sequential module with just the convolutional layers
        temp_conv = nn.Sequential(
            nn.Conv2d(shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        o = torch.zeros(1, *shape)
        o = temp_conv(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass through both networks
        
        Returns:
            target_feature: Output from target network
            predict_feature: Output from predictor network
        """
        target_feature = self.target(x)
        predict_feature = self.predictor(x)
        
        return target_feature, predict_feature 