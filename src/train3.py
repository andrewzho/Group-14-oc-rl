import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from obstacle_tower_env import ObstacleTowerEnv
from src.utils import save_checkpoint, load_checkpoint, ActionFlattener
from src.create_env import create_obstacle_tower_env
import argparse
import os

# Patch np.bool issue dynamically
if not hasattr(np, "bool"):
    np.bool = np.bool_

print("Patched np.bool issue dynamically.")

# Now import mlagents_envs
import mlagents_envs.rpc_utils

def log_message(message):
    with open("train3_log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")
    print(message)

class CEMPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CEMPolicy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def train_cem(env, policy, num_iterations=100, batch_size=50, elite_frac=0.2, learning_rate=0.01):
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    num_elite = max(1, int(batch_size * elite_frac))  # Ensure at least 1 elite sample
    
    for iteration in range(num_iterations):
        log_message(f"Starting Iteration {iteration+1}/{num_iterations}...")

        observations, actions, rewards = [], [], []

        for i in range(batch_size):
            log_message(f"  Episode {i+1}/{batch_size} starting...")
            obs = env.reset()
            done = False
            total_reward = 0
            step_count = 0  # Track step count to prevent infinite loops

            while not done and step_count < 500:
                step_count += 1
                log_message("    Processing Observation...")
                if isinstance(obs, tuple):
                    obs = np.concatenate([np.array(o).flatten() for o in obs])  # Flatten Tuple observations
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

                log_message("    Predicting Action...")
                action_probs = policy(obs_tensor).detach().numpy().flatten()
                action = np.random.choice(len(action_probs), p=np.exp(action_probs) / np.sum(np.exp(action_probs)))  # Softmax selection

                log_message(f"    Chosen action (before formatting): {action}")

                if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                    action = np.array(np.unravel_index(action, env.action_space.nvec), dtype=np.int32).reshape(1, -1)
                else:
                    action = np.array([action], dtype=np.int32)

                log_message(f"    Taking Action: {action}")
                next_obs, reward, done, _ = env.step(action)
                log_message(f"    ✅ Reward received: {reward}")

                observations.append(obs)
                actions.append(action.flatten())  # Ensure correct shape
                rewards.append(reward)
                total_reward += reward
                obs = next_obs

            log_message(f"  Episode {i+1} total reward: {total_reward}")

        log_message(f"Selecting top {num_elite} elite actions out of {batch_size}")
        elite_indices = np.argsort(rewards)[-num_elite:]
        elite_observations = np.array(observations)[elite_indices]
        elite_actions = np.array(actions)[elite_indices]

        if len(elite_observations) > 0:
            optimizer.zero_grad()
            predictions = policy(torch.FloatTensor(elite_observations))
            
            if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                elite_actions_tensor = torch.FloatTensor(elite_actions)
                loss = nn.MSELoss()(predictions, elite_actions_tensor)  # Use MSE for MultiDiscrete
            else:
                elite_actions_tensor = torch.LongTensor(elite_actions.squeeze())
                loss = nn.CrossEntropyLoss()(predictions, elite_actions_tensor)
            
            loss.backward()
            optimizer.step()
            log_message(f"Updated policy with loss: {loss.item()}")
        else:
            log_message("No elite actions found, skipping update. Consider modifying the reward function.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_path', type=str, required=True, help='Path to the Obstacle Tower environment')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of training iterations')
    args = parser.parse_args()
    
    env = create_obstacle_tower_env(args.env_path)
    if isinstance(env.observation_space, gym.spaces.Box):
        state_dim = int(np.prod(env.observation_space.shape))  # Flatten single Box space
    elif isinstance(env.observation_space, gym.spaces.Tuple):
        state_dim = sum(int(np.prod(space.shape)) for space in env.observation_space)  # Flatten Tuple space
    else:
        raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")

    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        action_dim = np.sum(env.action_space.nvec)  # Adjusted action space handling
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    policy = CEMPolicy(state_dim, action_dim)
    train_cem(env, policy, num_iterations=args.num_iterations)
    
    save_checkpoint({'state_dict': policy.state_dict()}, 'cem_checkpoint.pth')

if __name__ == '__main__':
    main()
