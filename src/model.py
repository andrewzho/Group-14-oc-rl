import torch
import torch.nn as nn

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
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # [64, 7, 7] -> [64, 5, 5]
            nn.ReLU()
        )
        conv_out_size = 64 * 5 * 5  # 1600
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),  # Larger hidden layer
            nn.ReLU()
        )
        self.policy = nn.Linear(1024, num_actions)
        self.value = nn.Linear(1024, 1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        x = self.fc(x)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

if __name__ == "__main__":
    net = PPONetwork(input_shape=(12, 84, 84), num_actions=54)
    print(net)