import gym
from obstacle_tower_env import ObstacleTowerEnv
from src.model import PPONetwork
from src.ppo import PPO
from src.utils import normalize, save_checkpoint, ActionFlattener
import torch
import numpy as np
import argparse
from collections import deque
import os
from mlagents_envs.exception import UnityCommunicatorStoppedException
import time
import logging
from src.create_env import create_obstacle_tower_env

# Set Unity to run in headless mode
os.environ['DISPLAY'] = ''

# Add at the top of your train.py file
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('obstacle_tower')

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Obstacle Tower")
    parser.add_argument('--realtime', action='store_true', help="Enable real-time mode for visualization")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory to save logs and checkpoints")
    parser.add_argument('--num_steps', type=int, default=1000000, help="Number of steps to train for")
    args = parser.parse_args()

    # Create directories if they don't exist
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Environment setup
    logger.info("Setting up environment...")
    env = create_obstacle_tower_env(realtime_mode=args.realtime)
    logger.info("Environment created")
    action_flattener = ActionFlattener(env.action_space.nvec)
    num_actions = action_flattener.action_space.n
    print(f"Number of actions: {num_actions}")

    # Model and PPO setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PPONetwork(input_shape=(12, 84, 84), num_actions=num_actions).to(device)
    
    # Load checkpoint if specified
    if args.checkpoint:
        try:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint from {args.checkpoint}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Improved PPO parameters
    ppo = PPO(
        model, 
        lr=3e-4,  # Slightly higher learning rate
        clip_eps=0.2,  # Standard PPO clipping parameter
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE lambda parameter
        epochs=4,  # Fewer epochs but more data
        batch_size=64  # Smaller batch size
    )

    # Training loop
    max_steps = args.num_steps
    steps_done = 0
    episode_rewards = []
    episode_lengths = []
    episode_floors = []
    
    start_time = time.time()
    current_floor = 0
    
    while steps_done < max_steps:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0
        episode_length = 0
        frame_stack = deque(maxlen=4)

        # Initialize frame stack
        logger.info(f"Resetting environment...")
        try:
            obs = env.reset()
            logger.info(f"Environment reset complete. Observation shape: {obs[0].shape}")
        except UnityCommunicatorStoppedException as e:
            print(f"Error during initial reset: {e}")
            env.close()
            return
            
        obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
        for _ in range(4):
            frame_stack.append(obs)
        state = np.concatenate(frame_stack, axis=0)
        obs = torch.tensor(state, dtype=torch.float32).to(device)

        # Collect trajectory
        for _ in range(2048):
            with torch.no_grad():
                obs_batched = obs.unsqueeze(0)
                policy_logits, value = model(obs_batched)
                dist = torch.distributions.Categorical(logits=policy_logits)
                action_idx = dist.sample().item()
                action = action_flattener.lookup_action(action_idx)
                action = np.array(action)

            next_obs, reward, done, info = env.step(action)
            
            # Track highest floor reached
            if info["current_floor"] > current_floor:
                current_floor = info["current_floor"]
                print(f"New floor reached: {current_floor}")
            
            # Enhanced reward shaping
            move_idx, rot_idx, jump_idx, _ = action
            shaped_reward = reward
            
            # Encourage forward movement
            if move_idx == 1:  # Forward movement
                shaped_reward += 0.01
            # Small penalty for staying still
            elif move_idx == 0:
                shaped_reward -= 0.001
                
            # Small penalty for excessive rotation
            if rot_idx != 0:
                shaped_reward -= 0.0005
                
            # Small penalty for jumping without reason
            if jump_idx == 1:
                shaped_reward -= 0.005
                
            # Time penalty to encourage faster completion
            shaped_reward -= 0.0001
            
            # Success reward for keys and floor completion remains in the environment

            next_obs = np.transpose(next_obs[0], (2, 0, 1)) / 255.0
            frame_stack.append(next_obs)
            next_state = np.concatenate(frame_stack, axis=0)
            next_obs = torch.tensor(next_state, dtype=torch.float32).to(device)

            states.append(state)
            actions.append(action_idx)
            rewards.append(shaped_reward)
            log_probs.append(dist.log_prob(torch.tensor([action_idx], device=device)).item())
            values.append(value.item())
            dones.append(done)
            
            episode_reward += reward  # Track actual reward
            episode_length += 1
            steps_done += 1

            obs = next_obs
            state = next_state
            
            if done:
                # Log episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_floors.append(info["current_floor"])
                
                elapsed_time = time.time() - start_time
                episodes_completed = len(episode_rewards)
                steps_per_sec = steps_done / elapsed_time
                
                print(f"Episode {episodes_completed} - Reward: {episode_reward:.2f}, Length: {episode_length}, "
                      f"Floor: {info['current_floor']}, Steps: {steps_done}, "
                      f"Steps/sec: {steps_per_sec:.2f}")
                
                # Save statistics
                np.save(os.path.join(args.log_dir, "rewards.npy"), episode_rewards)
                np.save(os.path.join(args.log_dir, "lengths.npy"), episode_lengths)
                np.save(os.path.join(args.log_dir, "floors.npy"), episode_floors)
                
                episode_reward = 0
                episode_length = 0
                
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

        # Compute returns and advantages
        with torch.no_grad():
            obs_batched = obs.unsqueeze(0)
            _, next_value = model(obs_batched)
        advantages = ppo.compute_gae(rewards, values, next_value.item(), dones)
        returns = [a + v for a, v in zip(advantages, values)]
        advantages = normalize(advantages)

        # Update policy
        ppo.update(states, actions, log_probs, returns, advantages)

        # Save checkpoint
        if steps_done % 10000 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"step_{steps_done}.pth")
            save_checkpoint(model, checkpoint_path)

    env.close()

if __name__ == "__main__":
    main()