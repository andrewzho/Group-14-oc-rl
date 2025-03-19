"""
Enhanced Obstacle Tower training script with visualization capabilities.

This script extends the original training script with real-time visualization
and TensorBoard integration to display agent gameplay.

Usage:
  # Train with visualization:
  python visualization_main.py --env_path=/path/to/ObstacleTower --display --record_video
  
  # Visualize without training:
  python visualization_main.py --env_path=/path/to/ObstacleTower --load_path=save.pkl --display --visualize_only
  
  # Enable specific visualizations:
  python visualization_main.py --env_path=/path/to/ObstacleTower --display  # Show real-time window
  python visualization_main.py --env_path=/path/to/ObstacleTower --record_video  # Save videos only
  
Note: Requires OpenCV for real-time display (pip install opencv-python)
"""

import os
import argparse
import datetime
import torch
import numpy as np
import sys
import time

# Set non-interactive matplotlib backend to avoid GUI errors
import matplotlib
matplotlib.use('Agg')

from src.global_settings import IMAGE_SIZE, IMAGE_DEPTH
from src.reinforcement_learner import LoggingProximalPolicyTrainer
from src.helper_funcs import create_parallel_envs, atomic_save
from src.trainer_main import DynamicEntropyProximalPolicyTrainer
from src.parallel_envs import ParallelGymWrapper

# Import visualization components
from src.visualization.visualization_wrapper import VisualizationWrapper
from src.visualization.tensorboard_integration import TensorboardVisualizer
from src.visualization.visualized_collector import VisualizedTrajectoryCollector

# Import models from the trainer_main.py
from src.trainer_main import BaseCNNModel, AdvancedCNNModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Agent with Visualization")
    
    # Environment settings
    parser.add_argument('--env_path', type=str, required=True, 
                      help='Path to the Obstacle Tower environment')
    parser.add_argument('--num_envs', type=int, default=8, 
                      help='Number of parallel environments')
    
    # Trajectory settings
    parser.add_argument('--num_steps', type=int, default=256,
                      help='Number of timesteps per trajectory_store')
    
    # Model architecture
    parser.add_argument('--model_type', type=str, default='advanced',
                      choices=['base', 'advanced'],
                      help='Type of CNN model to use')
    
    # ProximalPolicyTrainer hyperparameters
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.995,
                      help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95, 
                      help='GAE lambda')
    parser.add_argument('--clip_eps', type=float, default=0.2, 
                      help='Clipping epsilon')
    parser.add_argument('--ent_reg', type=float, default=0.01,
                      help='Entropy regularization coefficient')
    
    # Visualization settings
    parser.add_argument('--display', action='store_true',
                      help='Display agent gameplay in real-time window')
    parser.add_argument('--record_video', action='store_true',
                      help='Record agent gameplay videos')
    parser.add_argument('--video_dir', type=str, default='./agent_videos',
                      help='Directory to save agent videos')
    parser.add_argument('--video_freq', type=int, default=10,
                      help='How often to record videos (every N episodes)')
    parser.add_argument('--display_env', type=int, default=0,
                      help='Which environment to visualize (0 to num_envs-1)')
    parser.add_argument('--visualize_only', action='store_true',
                      help='Only run visualization without training')
    parser.add_argument('--max_episodes', type=int, default=100,
                      help='Maximum episodes for visualization mode')
    
    # Video quality settings
    parser.add_argument('--video_quality', type=str, default='high',
                      choices=['low', 'medium', 'high', 'ultra'], 
                      help='Quality of recorded videos (ultra=1080p)')
    parser.add_argument('--display_width', type=int, default=800,
                      help='Width of the display window')
    parser.add_argument('--display_height', type=int, default=600,
                      help='Height of the display window')
    
    # Training settings
    parser.add_argument('--worker_start', type=int, default=0,
                      help='Starting worker ID for environments')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cpu or cuda)')
    parser.add_argument('--save_path', type=str, default='./save_visual.pkl',
                      help='Path to save model checkpoint')
    parser.add_argument('--load_path', type=str, default=None,
                      help='Path to load model checkpoint from')
    parser.add_argument('--use_enhanced_network', action='store_true',
                  help='Use the enhanced network architecture')
    parser.add_argument('--use_intrinsic_rewards', action='store_true',
                  help='Use intrinsic rewards for exploration')
    
    return parser.parse_args()

def create_visualized_environments(args, tensorboard_visualizer=None):
    """
    Create environments with visualization wrappers.
    
    This simplified version just creates regular environments and handles
    visualization separately during training.
    """
    # Set environment path
    os.environ['OBS_TOWER_PATH'] = args.env_path
    
    # Create regular parallel environments
    parallel_envs = create_parallel_envs(num_envs=args.num_envs, start=args.worker_start)
    
    # Just return the regular environments
    return parallel_envs


def main():
    """
    Main function to run training with visualization or visualization-only mode.
    """
    args = parse_args()
    
    # Handle visualization-only mode
    if args.visualize_only:
        visualize_only(args)
        return
    
    # Otherwise, run training with visualization
    train_with_visualization(args)

def train_with_visualization(args):
    """Run training with visualization capabilities."""
    device = torch.device(args.device)

    # Create environments 
    parallel_envs = create_visualized_environments(args)

    # Get observation space shape and format it for CNN
    obs_shape = parallel_envs.observation_space.shape
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    action_dim = parallel_envs.action_space.n

    # Create model
    if args.use_enhanced_network:
        print("Using advanced network architecture")
        model = AdvancedCNNModel(input_shape, action_dim).to(device)
    else:
        print("Using base network architecture")
        model = BaseCNNModel(input_shape, action_dim).to(device)

    # Load model if specified
    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=device))
        print("Model loaded successfully!")

    # Initialize TensorBoard logging and visualizer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'tower_agent_viz_{current_time}')
    visualizer = TensorboardVisualizer(log_dir=log_dir)

    # Create data collector with visualization
    data_collector = VisualizedTrajectoryCollector(
        parallel_envs=parallel_envs,
        model=model,
        num_steps=args.num_steps,
        visualizer=visualizer,
        display_env_idx=args.display_env,
        video_record_freq=args.video_freq,
        display=args.display,  # Pass display flag
        record_video=args.record_video,  # Pass record flag
        display_width=args.display_width,
        display_height=args.display_height,
        video_quality=args.video_quality
    )

    # Create PPO trainer
    if args.use_intrinsic_rewards:
        trainer = DynamicEntropyProximalPolicyTrainer(
            model=model,
            epsilon=args.clip_eps,
            gamma=args.gamma,
            lam=args.lam,
            lr=args.lr,
            ent_reg=args.ent_reg,
            log_dir=log_dir
        )
    else:
        trainer = LoggingProximalPolicyTrainer(
            model=model,
            epsilon=args.clip_eps,
            gamma=args.gamma,
            lam=args.lam,
            lr=args.lr,
            ent_reg=args.ent_reg,
            log_dir=log_dir
        )

    # Print training configuration
    print("\n=== Training Configuration with Visualization ===")
    print(f"Network: {'Advanced' if args.use_enhanced_network else 'Base'}")
    print(f"Environments: {args.num_envs}")
    print(f"Display environment: {args.display_env}")
    print(f"Display enabled: {args.display}")
    print(f"Video recording: {args.record_video}")
    print(f"Device: {args.device}")
    print("=================================================\n")

    # Print TensorBoard viewing instructions
    print(f"TensorBoard logs are being written to {log_dir}")
    print("To view training progress and agent visualizations, run:")
    print(f"tensorboard --logdir={os.path.dirname(log_dir)}\n")

    try:
        # Start training with visualization
        trainer.outer_loop(data_collector, save_path=args.save_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print(f"Saving final model to {args.save_path}...")
        atomic_save(model.state_dict(), args.save_path)
        print("Model saved successfully!")

def visualize_only(args):
    """Run agent for visualization only without training."""
    device = torch.device(args.device)
    
    # Set environment path
    os.environ['OBS_TOWER_PATH'] = args.env_path
    
    # Create a single environment for visualization
    from src.helper_funcs import create_single_env
    env = create_single_env(args.worker_start)
    
    # Wrap with visualization wrapper
    env = VisualizationWrapper(
        env=env,
        save_dir=args.video_dir,
        display=args.display,
        record_episodes=args.record_video,
        video_quality=args.video_quality,
        display_width=args.display_width,
        display_height=args.display_height
    )
    
    # Get observation space shape and format it for CNN
    obs_shape = env.observation_space.shape
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    action_dim = env.action_space.n
    
    # Create model
    if args.model_type == 'advanced':
        model = AdvancedCNNModel(input_shape, action_dim).to(device)
    else:
        model = BaseCNNModel(input_shape, action_dim).to(device)
    
    # Load model
    if args.load_path and os.path.exists(args.load_path):
        print(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Error: Must provide a valid model checkpoint for visualization mode")
        return
    
    # Setup TensorBoard visualizer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'tower_agent_viz_{current_time}')
    visualizer = TensorboardVisualizer(log_dir=log_dir)
    
    # Run the agent for visualization
    print(f"Running agent for visualization for {args.max_episodes} episodes...")
    
    # Reset the environment
    obs = env.reset()
    episode_count = 0
    episode_step = 0
    episode_reward = 0
    frames = []
    
    try:
        while episode_count < args.max_episodes:
            # Get action from model
            memory_state = np.zeros([50, 22], dtype=np.float32)  # Dummy memory state
            model_output = model.step(memory_state, obs)
            action = model_output['actions'][0]
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Update observation and reward
            obs = next_obs
            episode_reward += reward
            episode_step += 1
            
            # Store frame for video recording
            if isinstance(obs, np.ndarray):
                frames.append(obs.copy())
            
            # Log to TensorBoard
            visualizer.log_observation(obs, episode_count * 1000 + episode_step)
            visualizer.log_action(action, episode_count * 1000 + episode_step)
            
            # Print floor information if available
            if 'current_floor' in info:
                print(f"Step {episode_step}, Floor: {info['current_floor']}, Reward: {reward:.2f}")
            
            # Reset if episode is done
            if done:
                print(f"Episode {episode_count+1} finished with reward {episode_reward:.2f} after {episode_step} steps")
                
                # Log episode video to TensorBoard if requested
                if args.record_video and len(frames) > 0:
                    visualizer.log_episode_video(frames, episode_count)
                    print(f"Recorded video with {len(frames)} frames")
                
                # Reset for next episode
                episode_count += 1
                episode_step = 0
                episode_reward = 0
                frames = []
                obs = env.reset()
                
                # Sleep a little between episodes
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    
    # Close environment
    env.close()
    
    print("Visualization complete!")

if __name__ == '__main__':
    main()