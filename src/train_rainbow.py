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

def preprocess_frame_stack(obs, frame_stack):
    """Process observation and add to frame stack."""
    # For ObstacleTower environment
    if isinstance(obs, tuple):
        # Get only the visual observation
        img_obs = obs[0]
    else:
        img_obs = obs
    
    # Normalize to [0,1] if needed
    if img_obs.dtype == np.uint8:
        img_obs = img_obs.astype(np.float32) / 255.0
    
    # Convert to channels-first format
    img_obs = np.transpose(img_obs, (2, 0, 1))
    
    # Add to frame stack
    frame_stack.append(img_obs)
    
    # Stack frames
    state = np.concatenate(list(frame_stack), axis=0)
    
    return state

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
    
    # Create environment
    print("Creating environment...")
    env = create_obstacle_tower_env(
        executable_path=args.env_path,
        realtime_mode=False,
        timeout=300
    )
    
    # Seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if hasattr(env, 'seed'):
            env.seed(args.seed)
    
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
    
    # Initialize frame stack
    frame_stack = deque(maxlen=4)
    
    # Get initial observation to determine state shape
    obs = env.reset()
    
    # Fill frame stack with initial observation
    if isinstance(obs, tuple):
        img_obs = obs[0]
    else:
        img_obs = obs
    
    # Convert to channels-first
    img_obs = np.transpose(img_obs, (2, 0, 1))
    
    for _ in range(4):
        frame_stack.append(img_obs)
    
    # Stack frames to get state
    state = np.concatenate(list(frame_stack), axis=0)
    
    # Get state shape
    state_shape = state.shape
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
    
    # Training loop
    total_steps = 0
    episode = 0
    
    # For tracking training speed
    training_start_time = time.time()
    last_checkpoint_time = training_start_time
    last_metrics_time = training_start_time
    
    print("Starting training loop...")
    
    try:
        while total_steps < args.num_steps:
            episode += 1
            episode_reward = 0
            episode_steps = 0
            episode_loss = []
            episode_intrinsic = []
            
            # Reset environment
            obs = env.reset()
            
            # Clear frame stack and refill
            frame_stack.clear()
            if isinstance(obs, tuple):
                img_obs = obs[0]
            else:
                img_obs = obs
            
            img_obs = np.transpose(img_obs, (2, 0, 1))
            
            for _ in range(4):
                frame_stack.append(img_obs)
            
            state = np.concatenate(list(frame_stack), axis=0)
            
            # Track maximum floor reached
            max_floor = 0
            done = False
            
            while not done:
                # Select an action
                action = agent.select_action(state)
                
                # Convert the discrete action to the format expected by the environment
                # The environment expects a numpy array that can be reshaped
                if isinstance(action, (int, np.int64, np.int32)):
                    if action_flattener is not None:
                        # Convert the flat action index to the multi-discrete action format
                        action_array = np.array(action_flattener.lookup_action(action), dtype=np.int32)
                    else:
                        # If no flattener, just convert to a numpy array
                        action_array = np.array([action], dtype=np.int32)
                else:
                    action_array = action
                
                # Step the environment
                obs, reward, done, info = env.step(action_array)
                
                # Process new observation
                next_state = preprocess_frame_stack(obs, frame_stack)
                
                # Calculate intrinsic reward
                intrinsic_reward = agent.calculate_intrinsic_reward(next_state)
                episode_intrinsic.append(intrinsic_reward)
                
                # Process the step
                agent.step(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                
                # Track episode statistics
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Track floor
                if 'current_floor' in info and info['current_floor'] > max_floor:
                    max_floor = info['current_floor']
                    print(f"New floor reached: {max_floor} (Episode {episode}, Step {total_steps})")
                
                # Learn every few steps
                if total_steps % args.learn_every == 0 and total_steps > args.learning_starts:
                    loss = agent.learn()
                    episode_loss.append(loss)
                
                # Render if requested
                if args.render:
                    env.render()
                
                # Check for episode termination by max length
                if episode_steps >= args.max_episode_steps:
                    print(f"Episode terminated after reaching max steps ({args.max_episode_steps})")
                    done = True
                
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
                    
                    print(f"Episode {episode} in progress - Steps: {total_steps}, "
                          f"Reward: {episode_reward:.2f}, Floor: {max_floor}, "
                          f"Steps/sec: {steps_per_sec:.2f}")
            
            # Episode complete
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_steps)
            metrics['floors_reached'].append(max_floor)
            
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                metrics['losses'].append(avg_loss)
            
            if episode_intrinsic:
                avg_intrinsic = sum(episode_intrinsic) / len(episode_intrinsic)
                metrics['intrinsic_rewards'].append(avg_intrinsic)
            
            # Print episode summary
            print(f"Episode {episode} - Reward: {episode_reward:.2f}, Length: {episode_steps}, "
                  f"Floor: {max_floor}, Total Steps: {total_steps}")
            
            # Save floor-specific checkpoints
            if max_floor > 0:
                floor_path = os.path.join(args.log_dir, f"rainbow_floor_{max_floor}.pth")
                if not os.path.exists(floor_path):
                    print(f"Saving checkpoint for floor {max_floor}")
                    agent.save(floor_path)
            
            # Evaluate periodically
            if episode % args.eval_interval == 0:
                evaluate(agent, env, device, args.eval_episodes, args.max_episode_steps, args.render)
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        # Save final checkpoint and metrics
        final_checkpoint_path = os.path.join(args.log_dir, "rainbow_final.pth")
        agent.save(final_checkpoint_path)
        
        with open(os.path.join(args.log_dir, 'metrics.json'), 'w') as f:
            json.dump(to_python_type(metrics), f, indent=4)
        
        plot_metrics(args.log_dir, metrics, show=False)
        
        # Close environment
        env.close()
        
        print(f"Training complete. Total steps: {total_steps}, Episodes: {episode}")
        print(f"Results saved to {args.log_dir}")

def evaluate(agent, env, device, episodes=5, max_steps=1000, render=False):
    """Evaluate the agent's performance."""
    print("\nEvaluating agent...")
    
    # Initialize frame stack
    frame_stack = deque(maxlen=4)
    
    returns = []
    floors_reached = []
    
    for i in range(episodes):
        # Reset environment
        obs = env.reset()
        
        # Clear frame stack and refill
        frame_stack.clear()
        if isinstance(obs, tuple):
            img_obs = obs[0]
        else:
            img_obs = obs
        
        img_obs = np.transpose(img_obs, (2, 0, 1))
        
        for _ in range(4):
            frame_stack.append(img_obs)
        
        state = np.concatenate(list(frame_stack), axis=0)
        
        # Initialize episode metrics
        episode_return = 0
        max_floor = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Select action (deterministically for evaluation)
            action = agent.select_action(state, evaluate=True)
            
            # Convert the discrete action to the format expected by the environment
            # The environment expects a numpy array that can be reshaped
            if isinstance(action, (int, np.int64, np.int32)):
                if action_flattener is not None:
                    # Convert the flat action index to the multi-discrete action format
                    action_array = np.array(action_flattener.lookup_action(action), dtype=np.int32)
                else:
                    # If no flattener, just convert to a numpy array
                    action_array = np.array([action], dtype=np.int32)
            else:
                action_array = action
                
            # Step environment
            obs, reward, done, info = env.step(action_array)
            
            # Process new observation
            next_state = preprocess_frame_stack(obs, frame_stack)
            
            # Update state
            state = next_state
            
            # Track episode statistics
            episode_return += reward
            step += 1
            
            # Track floor
            if 'current_floor' in info and info['current_floor'] > max_floor:
                max_floor = info['current_floor']
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
    parser.add_argument('--eps_end', type=float, default=0.01,
                        help='Final epsilon')
    parser.add_argument('--eps_decay', type=int, default=100000,
                        help='Epsilon decay rate')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Priority exponent for PER')
    parser.add_argument('--beta_start', type=float, default=0.4,
                        help='Initial beta for importance sampling weights')
    
    # RND settings
    parser.add_argument('--int_coef', type=float, default=0.5,
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