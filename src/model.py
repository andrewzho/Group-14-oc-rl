import torch
import torch.nn as nn

class PPONetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(PPONetwork, self).__init__()
        # input_shape should be (3, 84, 84) for RGB images, but adjust based on actual output
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),  # [3, 84, 84] → [32, 19, 19]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),  # [32, 19, 19] → [64, 8, 8]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # [64, 8, 8] → [64, 6, 6]
            nn.ReLU()
        )
        # Compute conv_out_size based on debug output [1, 64, 7, 7]
        conv_out_size = 64 * 7 * 7  # 3136 (actual output from debug)
        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),  # Update to 3136
            nn.ReLU()
        )
        # Policy head (actor)
        self.policy = nn.Linear(512, num_actions)
        # Value head (critic)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        # Ensure input is batched [batch_size, 3, 84, 84]; if unbatched, add batch dimension
        if x.dim() == 3:  # Unbatched input [3, 84, 84]
            x = x.unsqueeze(0)  # Add batch dimension [1, 3, 84, 84]
        # print("Input shape to PPONetwork (after batching if needed):", x.shape)
        # print("Input data sample:", x[0, 0, :5, :5] if x.dim() >= 4 else x)  # Print a small sample of the input
        x = self.conv(x)
        # print("Shape after conv:", x.shape)
        # print("Conv output data sample:", x[0, 0, :5, :5] if x.dim() >= 4 else x)  # Print a small sample after conv
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 3136]
        # print("Shape after flattening:", x.shape)
        # print("Flattened data sample:", x[0, :5] if x.dim() >= 2 else x)  # Print a small sample after flattening
        x = self.fc(x)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

if __name__ == "__main__":
    net = PPONetwork(input_shape=(3, 84, 84), num_actions=6)  # Adjust based on env
    print(net)