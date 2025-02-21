import gym
from obstacle_tower_env import ObstacleTowerEnv
from src.model import PPONetwork
from src.ppo import PPO
from src.utils import normalize, save_checkpoint
import torch
import numpy as np
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train PPO agent for Obstacle Tower")
    parser.add_argument('--realtime', action='store_true', help="Enable real-time mode for visualization")
    args = parser.parse_args()

    # Environment setup
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=True, realtime_mode=args.realtime)  # Adjust path
    obs_shape = (3, 84, 84)  # Assuming resized images
    
    # Debug print to inspect action space
    print("Action space:", env.action_space)
    print("Action space type:", type(env.action_space))
    print("Action space nvec:", env.action_space.nvec if isinstance(env.action_space, gym.spaces.MultiDiscrete) else None)

    # Handle MultiDiscrete action space
    if isinstance(env.action_space, gym.spaces.MultiDiscrete):
        num_actions = np.prod(env.action_space.nvec)  # Total number of combinations
    else:
        num_actions = env.action_space.n  # For Discrete space

    print("Number of actions:", num_actions)

    # Model and PPO setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPONetwork(input_shape=obs_shape, num_actions=num_actions).to(device)
    ppo = PPO(model)

    # Training loop
    max_steps = 1000000
    steps_done = 0
    episode_rewards = []

    while steps_done < max_steps:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0

        obs = env.reset()
        obs = np.transpose(obs, (2, 0, 1))  # To (C, H, W) → [3, 84, 84]
        obs = torch.tensor(obs, dtype=torch.float32).to(device) / 255.0  # Shape [3, 84, 84]

        for _ in range(2048):  # Collect 2048 steps
            with torch.no_grad():
                # Ensure obs is batched [1, 3, 84, 84]
                obs_batched = obs.unsqueeze(0)  # Add batch dimension
                policy_logits, value = model(obs_batched)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action = dist.sample().item()  # Sample a single action and convert to scalar

            next_obs, reward, done, _ = env.step(action)
            next_obs = np.transpose(next_obs, (2, 0, 1))  # To (C, H, W) → [3, 84, 84]
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device) / 255.0  # Shape [3, 84, 84]

            states.append(obs.cpu().numpy())  # Shape [3, 84, 84]
            actions.append(action)
            rewards.append(reward)
            log_probs.append(dist.log_prob(torch.tensor([action])).item())  # Adjust log_prob for single action
            values.append(value.item())
            dones.append(done)
            episode_reward += reward
            steps_done += 1

            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0
                obs = env.reset()
                obs = np.transpose(obs, (2, 0, 1))
                obs = torch.tensor(obs, dtype=torch.float32).to(device) / 255.0

        # Compute returns and advantages with batched obs
        with torch.no_grad():
            obs_batched = obs.unsqueeze(0)  # Add batch dimension
            _, next_value = model(obs_batched)
        advantages = ppo.compute_gae(rewards, values, next_value.item(), dones)
        returns = [a + v for a, v in zip(advantages, values)]
        advantages = normalize(advantages)

        # Update policy
        ppo.update(states, actions, log_probs, returns, advantages)

        # Save checkpoint periodically
        if steps_done % 10000 == 0:
            save_checkpoint(model, f"logs/checkpoints/step_{steps_done}.pth")

    env.close()

if __name__ == "__main__":
    main()