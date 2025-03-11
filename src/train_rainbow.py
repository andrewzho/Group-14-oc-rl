import os
import argparse
import numpy as np
import torch
import time
from collections import deque
import matplotlib.pyplot as plt
import datetime
import json
import gym
import math
import random
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.rainbow_dqn import RainbowDQN
from src.create_env import create_obstacle_tower_env
from src.utils import ActionFlattener, MetricsTracker

def to_python_type(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [to_python_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_python_type(value) for key, value in obj.items()}
    else:
        return obj

def plot_metrics(log_dir, metrics, show=False):
    """Plot training metrics."""
    plt.figure(figsize=(20, 10))
    
    # Plot rewards
    plt.subplot(2, 3, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot episode lengths
    plt.subplot(2, 3, 2)
    plt.plot(metrics['episode_lengths'])
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    # Plot floor reached
    plt.subplot(2, 3, 3)
    plt.plot(metrics['floors_reached'])
    plt.title('Max Floor Reached')
    plt.xlabel('Episode')
    plt.ylabel('Floor')
    
    # Plot losses
    if len(metrics['losses']) > 0:
        plt.subplot(2, 3, 4)
        plt.plot(metrics['losses'])
        plt.title('Loss')
        plt.xlabel('Training step')
        plt.ylabel('Loss')
    
    # Plot intrinsic rewards
    if len(metrics['intrinsic_rewards']) > 0:
        plt.subplot(2, 3, 5)
        plt.plot(metrics['intrinsic_rewards'])
        plt.title('Intrinsic Rewards')
        plt.xlabel('Step')
        plt.ylabel('Intrinsic Reward')
    
    # Plot training progress
    plt.subplot(2, 3, 6)
    plt.plot(metrics['steps_per_second'])
    plt.title('Training Speed')
    plt.xlabel('Update')
    plt.ylabel('Steps/second')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(log_dir, f'rainbow_metrics_{timestamp}.png'))
    
    if show:
        plt.show()
    plt.close()

def preprocess_frame_stack(obs, frame_stacks):
    """Process observations and add to frame stacks for multiple environments."""
    batch_size = len(obs)
    processed_states = []
    
    for i in range(batch_size):
        # Get observation for this environment
        env_obs = obs[i]
        frame_stack = frame_stacks[i]
        
        # Process observation
        if isinstance(env_obs, tuple):
            img_obs = env_obs[0]
        else:
            img_obs = env_obs
        
        # Normalize to [0,1] if needed
        if img_obs.dtype == np.uint8:
            img_obs = img_obs.astype(np.float32) / 255.0
        
        # Convert to channels-first format
        img_obs = np.transpose(img_obs, (2, 0, 1))
        
        # Add to frame stack
        frame_stack.append(img_obs)
        
        # Stack frames
        state = np.concatenate(list(frame_stack), axis=0)
        processed_states.append(state)
    
    # Stack all states into a batch
    return np.array(processed_states)

def shape_reward(reward, info, prev_info=None):
    """Apply reward shaping to provide denser rewards."""
    shaped_reward = reward
    
    # Bonus for moving (based on position change)
    if prev_info is not None and 'x_pos' in info and 'x_pos' in prev_info:
        # Calculate distance moved
        dx = info['x_pos'] - prev_info['x_pos']
        dz = info.get('z_pos', 0) - prev_info.get('z_pos', 0)
        dist_moved = np.sqrt(dx**2 + dz**2)
        
        # Small bonus for movement to encourage exploration
        if dist_moved > 0.1:
            shaped_reward += 0.01
    
    # Bonus for exploring new areas
    if 'visit_count' in info:
        # Larger bonus for less visited areas
        visit_count = info['visit_count']
        if visit_count == 0:  # First visit
            shaped_reward += 0.05
        elif visit_count < 3:  # Recently discovered
            shaped_reward += 0.02
    
    # Bonus for key collection
    if prev_info is not None and 'total_keys' in info and 'total_keys' in prev_info:
        if info['total_keys'] > prev_info['total_keys']:
            shaped_reward += 0.5  # Significant bonus for getting a key
    
    # Bonus for door interactions
    if 'door_found' in info and info['door_found']:
        shaped_reward += 0.2
        
    # Bonus for time efficiency
    if 'time_remaining' in info and prev_info is not None and 'time_remaining' in prev_info:
        # Small bonus for efficient use of time
        if info['time_remaining'] > prev_info['time_remaining']:
            shaped_reward += 0.01
    
    return shaped_reward

def make_env(env_path, seed=None, realtime_mode=False, rank=0):
    """Factory function to create environments with proper seeding."""
    def _init():
        env = create_obstacle_tower_env(
            executable_path=env_path,
            realtime_mode=realtime_mode,
            timeout=300,
            worker_id=rank  # Use rank for unique worker_id
        )
        if seed is not None:
            env.seed(seed + rank)
        return env
    return _init

def train(args):
    """Main training function."""
    print("Starting Rainbow DQN training...")
    
    # Create log directory
    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.log_dir = f"./logs/rainbow_{timestamp}"
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(to_python_type(vars(args)), f, indent=4)
    
    # Set device for computation
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create vectorized environments
    print(f"Creating {args.n_envs} parallel environments...")
    env_fns = [make_env(args.env_path, args.seed, args.realtime_mode, i) for i in range(args.n_envs)]
    
    if args.n_envs > 1:
        # Use SubprocVecEnv for multiple environments (runs in separate processes)
        env = SubprocVecEnv(env_fns)
    else:
        # Use DummyVecEnv for a single environment (simpler, no multiprocessing)
        env = DummyVecEnv(env_fns)
    
    # Create action flattener for the MultiDiscrete action space
    action_flattener = None
    if hasattr(env.action_space, 'n'):
        action_size = env.action_space.n
    else:
        action_flattener = ActionFlattener(env.action_space.nvec)
        action_size = action_flattener.action_space.n
    
    # Set observation shape for RGB frames with frame stacking
    state_shape = (12, 84, 84)  # 4 stacked RGB frames, channels first
    print(f"State shape: {state_shape}, Action size: {action_size}")
    
    # Initialize frame stacks for each environment
    frame_stacks = [deque(maxlen=4) for _ in range(args.n_envs)]
    
    # Get initial observation
    obs = env.reset()
    
    # Fill frame stacks with initial observations
    for i in range(args.n_envs):
        env_obs = obs[i]
        if isinstance(env_obs, tuple):
            img_obs = env_obs[0]
        else:
            img_obs = env_obs
        
        # Convert to channels-first
        img_obs = np.transpose(img_obs, (2, 0, 1))
        
        for _ in range(4):
            frame_stacks[i].append(img_obs)
    
    # Stack frames to get initial states
    states = np.array([np.concatenate(list(fs), axis=0) for fs in frame_stacks])
    
    # Get state shape from first environment
    state_shape = states[0].shape
    print(f"State shape: {state_shape}, Action size: {action_size}")
    
    # Create agent
    agent = RainbowDQN(
        state_shape=state_shape,
        action_size=action_size,
        device=device,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        n_steps=args.n_steps,
        target_update=args.target_update,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        alpha=args.alpha,
        beta_start=args.beta_start,
        int_coef=args.int_coef,
        ext_coef=args.ext_coef,
        rnd_lr=args.rnd_lr
    )
    
    # Load checkpoint if specified
    if args.checkpoint is not None:
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'floors_reached': [],
        'losses': [],
        'intrinsic_rewards': [],
        'steps_per_second': []
    }
    
    # For tracking environment episodes
    env_episodes = [0] * args.n_envs
    env_rewards = [0] * args.n_envs
    env_steps = [0] * args.n_envs
    env_floors = [0] * args.n_envs
    env_dones = [False] * args.n_envs
    
    # Training loop
    total_steps = 0
    episode = 0
    
    # For tracking training speed
    training_start_time = time.time()
    last_checkpoint_time = training_start_time
    last_metrics_time = training_start_time
    
    # For reward shaping
    prev_infos = [None] * args.n_envs
    
    print("Starting training loop...")
    
    try:
        while total_steps < args.num_steps:
            # Select actions for all environments
            actions = []
            for i in range(args.n_envs):
                action = agent.select_action(states[i])
                actions.append(action)
            
            # Convert actions to the format expected by the environment
            action_arrays = []
            for i, action in enumerate(actions):
                if isinstance(action, (int, np.int64, np.int32)):
                    if action_flattener is not None:
                        # Convert flat action to multi-discrete format
                        action_array = action_flattener.lookup_action(action)
                    else:
                        action_array = [action]
                else:
                    action_array = action
                action_arrays.append(action_array)
            
            # Step environments
            next_obs, rewards, dones, infos = env.step(action_arrays)
            
            # Process observations and update frame stacks
            next_states = preprocess_frame_stack(next_obs, frame_stacks)
            
            # Apply reward shaping and step agent for each environment
            for i in range(args.n_envs):
                # Apply reward shaping if enabled
                shaped_reward = shape_reward(rewards[i], infos[i], prev_infos[i]) if args.reward_shaping else rewards[i]
                prev_infos[i] = infos[i].copy() if infos[i] else None
                
                # Calculate intrinsic reward
                intrinsic_reward = agent.calculate_intrinsic_reward(next_states[i])
                
                # Step the agent (add to replay buffer)
                agent.step(states[i], actions[i], shaped_reward, next_states[i], dones[i])
                
                # Update episode tracking
                env_rewards[i] += rewards[i]  # Track original rewards for metrics
                env_steps[i] += 1
                
                # Track floor level
                if 'current_floor' in infos[i] and infos[i]['current_floor'] > env_floors[i]:
                    env_floors[i] = infos[i]['current_floor']
                    print(f"Env {i} - New floor reached: {env_floors[i]} (Episode {env_episodes[i]+1}, Total steps: {total_steps})")
                
                # Handle episode completion
                if dones[i]:
                    episode += 1
                    
                    # Record metrics for completed episode
                    metrics['episode_rewards'].append(env_rewards[i])
                    metrics['episode_lengths'].append(env_steps[i])
                    metrics['floors_reached'].append(env_floors[i])
                    
                    # Print episode summary
                    print(f"Env {i} - Episode {env_episodes[i]+1} - Reward: {env_rewards[i]:.2f}, "
                          f"Length: {env_steps[i]}, Floor: {env_floors[i]}, Total Steps: {total_steps}")
                    
                    # Save floor-specific checkpoints
                    if env_floors[i] > 0:
                        floor_path = os.path.join(args.log_dir, f"rainbow_floor_{env_floors[i]}.pth")
                        if not os.path.exists(floor_path):
                            print(f"Saving checkpoint for floor {env_floors[i]}")
                            agent.save(floor_path)
                    
                    # Reset episode tracking for this environment
                    env_episodes[i] += 1
                    env_rewards[i] = 0
                    env_steps[i] = 0
                    env_floors[i] = 0
            
            # Update states
            states = next_states
            
            # Increment step counter
            total_steps += args.n_envs
            
            # Learn every few steps
            if total_steps % args.learn_every == 0 and total_steps > args.learning_starts:
                loss = agent.learn()
                if loss > 0:  # Only record non-zero losses
                    metrics['losses'].append(loss)
            
            # Save checkpoint periodically
            current_time = time.time()
            if current_time - last_checkpoint_time > args.checkpoint_interval:
                checkpoint_path = os.path.join(args.log_dir, f"rainbow_step_{total_steps}.pth")
                agent.save(checkpoint_path)
                last_checkpoint_time = current_time
            
            # Log metrics periodically
            if current_time - last_metrics_time > args.metrics_interval:
                # Calculate steps per second
                steps_per_sec = total_steps / (current_time - training_start_time)
                metrics['steps_per_second'].append(steps_per_sec)
                
                # Save metrics
                with open(os.path.join(args.log_dir, 'metrics.json'), 'w') as f:
                    json.dump(to_python_type(metrics), f, indent=4)
                
                # Plot metrics
                plot_metrics(args.log_dir, metrics)
                
                last_metrics_time = current_time
                
                print(f"Progress - Total Steps: {total_steps}, Episodes: {episode}, "
                      f"Steps/sec: {steps_per_sec:.2f}")
            
            # Run evaluation periodically
            if episode > 0 and episode % args.eval_interval == 0:
                evaluate(agent, create_obstacle_tower_env(args.env_path), device, action_flattener, 
                         args.eval_episodes, args.max_episode_steps, args.render)
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Save final checkpoint and metrics
        final_checkpoint_path = os.path.join(args.log_dir, "rainbow_final.pth")
        agent.save(final_checkpoint_path)
        
        with open(os.path.join(args.log_dir, 'metrics.json'), 'w') as f:
            json.dump(to_python_type(metrics), f, indent=4)
        
        plot_metrics(args.log_dir, metrics, show=False)
        
        # Close environments
        env.close()
        
        print(f"Training complete. Total steps: {total_steps}, Episodes: {episode}")
        print(f"Results saved to {args.log_dir}")

def evaluate(agent, env, device, action_flattener=None, episodes=5, max_steps=1000, render=False):
    """Evaluate the agent's performance."""
    print("\nEvaluating agent...")
    
    # Initialize frame stacks
    frame_stacks = [deque(maxlen=4) for _ in range(episodes)]
    
    returns = []
    floors_reached = []
    
    for i in range(episodes):
        # Reset environments
        obs = env.reset()
        
        # Clear frame stacks and refill
        for frame_stack in frame_stacks:
            frame_stack.clear()
        if isinstance(obs, tuple):
            img_obs = obs[0]
        else:
            img_obs = obs
        
        img_obs = np.transpose(img_obs, (2, 0, 1))
        
        for _ in range(4):
            for frame_stack in frame_stacks:
                frame_stack.append(img_obs)
        
        states = preprocess_frame_stack(obs, frame_stacks)
        
        # Initialize episode metrics
        episode_return = 0
        max_floor = 0
        done = [False] * episodes
        step = 0
        
        while not all(done) and step < max_steps:
            # Select actions (deterministically for evaluation)
            actions = agent.select_action(states, evaluate=True)
            
            # Convert the discrete actions to the format expected by the environments
            # The environments expect numpy arrays that can be reshaped
            if isinstance(actions, (list, tuple)):
                action_arrays = [np.array(action_flattener.lookup_action(action), dtype=np.int32) for action in actions]
            else:
                action_arrays = [np.array([actions], dtype=np.int32)]
            
            # Step environments
            obs, rewards, dones, infos = env.step(action_arrays)
            
            # Process new states
            next_states = preprocess_frame_stack(obs, frame_stacks)
            
            # Update states
            states = next_states
            
            # Track episode statistics
            episode_return += sum(rewards)
            step += 1
            
            # Track floors
            for j in range(episodes):
                if 'current_floor' in infos[j] and infos[j]['current_floor'] > max_floor:
                    max_floor = infos[j]['current_floor']
                    print(f"Evaluation - New floor reached: {max_floor}")
            
            # Render if requested
            if render:
                env.render()
        
        # Episode complete
        returns.append(episode_return)
        floors_reached.append(max_floor)
        
        print(f"Evaluation Episode {i+1}/{episodes} - Return: {episode_return:.2f}, "
              f"Steps: {step}, Floor: {max_floor}")
    
    # Calculate average metrics
    avg_return = sum(returns) / len(returns)
    avg_floor = sum(floors_reached) / len(floors_reached)
    max_floor_reached = max(floors_reached)
    
    print(f"\nEvaluation complete - Avg Return: {avg_return:.2f}, "
          f"Avg Floor: {avg_floor:.2f}, Max Floor: {max_floor_reached}\n")
    
    return avg_return, avg_floor, max_floor_reached

def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train Rainbow DQN on Obstacle Tower')
    
    # Environment settings
    parser.add_argument('--env_path', type=str, default='./ObstacleTower/obstacletower.x86_64',
                        help='Path to Obstacle Tower executable')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n_envs', type=int, default=1,
                        help='Number of parallel environments')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Enable reward shaping for denser rewards')
    
    # Training settings
    parser.add_argument('--num_steps', type=int, default=2000000,
                        help='Total number of training steps')
    parser.add_argument('--max_episode_steps', type=int, default=5000,
                        help='Maximum steps per episode')
    parser.add_argument('--learn_every', type=int, default=4,
                        help='Learning frequency')
    parser.add_argument('--learning_starts', type=int, default=10000,
                        help='Steps before learning starts')
    
    # Rainbow DQN settings
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--n_steps', type=int, default=3,
                        help='Number of steps for n-step returns')
    parser.add_argument('--target_update', type=int, default=10000,
                        help='Target network update frequency')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--eps_start', type=float, default=1.0,
                        help='Starting epsilon for exploration')
    parser.add_argument('--eps_end', type=float, default=0.05,
                        help='Final epsilon')
    parser.add_argument('--eps_decay', type=int, default=100000,
                        help='Epsilon decay rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Priority exponent for PER')
    parser.add_argument('--beta_start', type=float, default=0.4,
                        help='Initial beta for importance sampling weights')
    
    # RND settings
    parser.add_argument('--int_coef', type=float, default=1.0,
                        help='Intrinsic reward coefficient')
    parser.add_argument('--ext_coef', type=float, default=1.0,
                        help='Extrinsic reward coefficient')
    parser.add_argument('--rnd_lr', type=float, default=0.001,
                        help='Learning rate for RND predictor')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint_interval', type=int, default=900,
                        help='Seconds between checkpoint saves')
    parser.add_argument('--metrics_interval', type=int, default=300,
                        help='Seconds between metrics updates')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Episodes between evaluations')
    parser.add_argument('--eval_episodes', type=int, default=3,
                        help='Number of episodes for evaluation')
    
    # Visualization
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--realtime_mode', action='store_true',
                        help='Run environment in realtime mode')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

if __name__ == '__main__':
    main() 