import gym
from obstacle_tower_env import ObstacleTowerEnv
from src.model import PPONetwork
from src.ppo import PPO
from src.utils import normalize, save_checkpoint, ActionFlattener
import torch
import numpy as np
import argparse
from collections import deque
from mlagents_envs.exception import UnityCommunicatorStoppedException

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Obstacle Tower")
    parser.add_argument('--realtime', action='store_true', help="Enable real-time mode for visualization")
    args = parser.parse_args()

    # Environment setup
    env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False, realtime_mode=args.realtime)
    action_flattener = ActionFlattener(env.action_space.nvec)
    num_actions = action_flattener.action_space.n
    print("Number of actions:", num_actions)

    # Model and PPO setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPONetwork(input_shape=(12, 84, 84), num_actions=num_actions).to(device)
    ppo = PPO(model, lr=1e-4, clip_eps=0.1, epochs=10, batch_size=128)

    # Training loop
    max_steps = 1000000
    steps_done = 0
    episode_rewards = []

    while steps_done < max_steps:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0
        frame_stack = deque(maxlen=4)

        # Initialize frame stack
        try:
            obs = env.reset()
        except UnityCommunicatorStoppedException as e:
            print(f"Error during initial reset: {e}")
            env.close()
            return
        obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
        for _ in range(4):
            frame_stack.append(obs)
        state = np.concatenate(frame_stack, axis=0)
        obs = torch.tensor(state, dtype=torch.float32).to(device)

        for _ in range(2048):
            with torch.no_grad():
                obs_batched = obs.unsqueeze(0)
                policy_logits, value = model(obs_batched)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action_idx = dist.sample().item()
                action = action_flattener.lookup_action(action_idx)
                action = np.array(action)

            next_obs, reward, done, _ = env.step(action)
            move_idx, _, jump_idx, _ = action
            shaped_reward = reward
            if move_idx != 0:
                shaped_reward += 0.01
            if jump_idx == 1:
                shaped_reward -= 0.005

            next_obs = np.transpose(next_obs[0], (2, 0, 1)) / 255.0
            frame_stack.append(next_obs)
            next_state = np.concatenate(frame_stack, axis=0)
            next_obs = torch.tensor(next_state, dtype=torch.float32).to(device)

            states.append(state)
            actions.append(action_idx)
            rewards.append(shaped_reward)
            log_probs.append(dist.log_prob(torch.tensor([action_idx])).item())
            values.append(value.item())
            dones.append(done)
            episode_reward += reward
            steps_done += 1

            obs = next_obs
            state = next_state
            if done:
                episode_rewards.append(episode_reward)
                np.save("rewards.npy", episode_rewards)
                print(f"Episode reward: {episode_reward}")
                episode_reward = 0
                try:
                    obs = env.reset()
                except UnityCommunicatorStoppedException as e:
                    print(f"Error during reset after episode: {e}")
                    env.close()
                    return
                obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
                frame_stack = deque(maxlen=4)
                for _ in range(4):
                    frame_stack.append(obs)
                state = np.concatenate(frame_stack, axis=0)
                obs = torch.tensor(state, dtype=torch.float32).to(device)

        with torch.no_grad():
            obs_batched = obs.unsqueeze(0)
            _, next_value = model(obs_batched)
        advantages = ppo.compute_gae(rewards, values, next_value.item(), dones)
        returns = [a + v for a, v in zip(advantages, values)]
        advantages = normalize(advantages)

        ppo.update(states, actions, log_probs, returns, advantages)

        if steps_done % 10000 == 0:
            save_checkpoint(model, f"logs/checkpoints/step_{steps_done}.pth")

    env.close()

if __name__ == "__main__":
    main()