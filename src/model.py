import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-5)  # Lower momentum for more stable statistics
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-5)  # Lower momentum for more stable statistics
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    """Channel attention module to focus on important features"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        return x * attention_map

class PPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__()
        # Input shape is (12, 84, 84) for 4 stacked frames (4 * 3 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),  # [12, 84, 84] -> [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [32, 20, 20] -> [64, 9, 9]
            nn.ReLU(),
            ResidualBlock(64),  # Add residual block for better gradient flow
            ChannelAttention(64),  # Add attention to focus on important features
            SpatialAttention(kernel_size=5),  # Add spatial attention
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # [64, 9, 9] -> [64, 7, 7]
            nn.ReLU(),
            ResidualBlock(64),  # Another residual block
            ChannelAttention(64),  # Add attention
            SpatialAttention(kernel_size=3),  # Add spatial attention with smaller kernel
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # [64, 7, 7] -> [128, 5, 5]
            nn.ReLU(),
            nn.Flatten()  # [128, 5, 5] -> [3200]
        )
        conv_out_size = self._get_conv_output(input_shape)
        
        # Separate advantage (policy) and value streams for better learning
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased dropout for better regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout to second layer too
            nn.Linear(256, num_actions)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Increased dropout for better regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout to second layer too
            nn.Linear(256, 1)
        )
        
        # Layer norm for better training stability
        self.layer_norm_advantage = nn.LayerNorm(512)
        self.layer_norm_value = nn.LayerNorm(512)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers."""
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv(input)
            return int(np.prod(output.size()))

    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization with proper scaling."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            gain = 1.0
            if module.weight.shape[0] == 1:  # Value stream final layer
                gain = 0.01
            elif len(module.weight.shape) >= 2 and module.weight.shape[1] > 0:
                # For the policy head, use smaller gain
                if module.weight.shape[0] > 10:  # Likely the policy output layer
                    gain = np.sqrt(0.1)
                else:
                    gain = np.sqrt(2)
            
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # Normalize input to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.conv(x)
        
        # Apply advantage stream with layer normalization
        adv_features = self.advantage_stream[0](features)  # First linear layer
        adv_features = self.layer_norm_advantage(adv_features)
        adv_features = self.advantage_stream[1](adv_features)  # ReLU
        adv_features = self.advantage_stream[2](adv_features)  # Dropout
        adv_features = self.advantage_stream[3](adv_features)  # Second linear layer
        adv_features = self.advantage_stream[4](adv_features)  # ReLU
        adv_features = self.advantage_stream[5](adv_features)  # Dropout
        policy_logits = self.advantage_stream[6](adv_features)  # Output layer
        
        # Apply value stream with layer normalization
        val_features = self.value_stream[0](features)  # First linear layer
        val_features = self.layer_norm_value(val_features)
        val_features = self.value_stream[1](val_features)  # ReLU
        val_features = self.value_stream[2](val_features)  # Dropout
        val_features = self.value_stream[3](val_features)  # Second linear layer
        val_features = self.value_stream[4](val_features)  # ReLU
        val_features = self.value_stream[5](val_features)  # Dropout
        value = self.value_stream[6](val_features)  # Output layer
        
        return policy_logits, value

    def get_action_and_value(self, x, deterministic=False):
        """Convenience method for getting actions and values in one call."""
        policy_logits, value = self(x)
        dist = torch.distributions.Categorical(logits=policy_logits)
        
        if deterministic:
            action = torch.argmax(policy_logits, dim=1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

if __name__ == "__main__":
    net = PPONetwork(input_shape=(12, 84, 84), num_actions=54)
    print(net)
    
    # Test forward pass
    x = torch.randn(4, 12, 84, 84)  # Batch of 4 examples
    policy, value = net(x)
    print(f"Policy output shape: {policy.shape}")  # Should be [4, 54]
    print(f"Value output shape: {value.shape}")    # Should be [4, 1]