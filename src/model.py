import torch
import torch.nn as nn
import numpy as np

class PPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__()
        # Input shape is (12, 84, 84) for 4 stacked frames (4 * 3 channels)
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),  # [12, 84, 84] -> [32, 20, 20]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # [32, 20, 20] -> [64, 9, 9]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # [64, 9, 9] -> [64, 7, 7]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # [64, 7, 7] -> [128, 5, 5]
            nn.ReLU(),
            nn.Flatten()  # [128, 5, 5] -> [3200]
        )
        conv_out_size = self._get_conv_output(input_shape)
        
        # Separate advantage (policy) and value streams for better learning
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _get_conv_output(self, shape):
        """Calculate the output size of the convolutional layers."""
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv(input)
            return int(np.prod(output.size()))

    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # Normalize input to [0, 1] if not already
        if x.max() > 1.0:
            x = x / 255.0
            
        features = self.conv(x)
        policy_logits = self.advantage_stream(features)
        value = self.value_stream(features)
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
        return action, log_prob, value

if __name__ == "__main__":
    net = PPONetwork(input_shape=(12, 84, 84), num_actions=54)
    print(net)
    
    # Test forward pass
    x = torch.randn(4, 12, 84, 84)  # Batch of 4 examples
    policy, value = net(x)
    print(f"Policy output shape: {policy.shape}")  # Should be [4, 54]
    print(f"Value output shape: {value.shape}")    # Should be [4, 1]