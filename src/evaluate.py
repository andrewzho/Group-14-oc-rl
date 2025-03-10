import gym
from obstacle_tower_env import ObstacleTowerEnv
from src.model import PPONetwork
from src.utils import load_checkpoint, ActionFlattener
import torch
import numpy as np
import argparse
from collections import deque
import os
import time
import warnings
from src.create_env import create_obstacle_tower_env

# Import cv2 for visualization
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    warnings.warn("OpenCV (cv2) not found. Visualization may be limited.")
    CV2_AVAILABLE = False

# Try to import imageio as fallback for video creation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

def main(args):
    """
    Run a trained agent in the Obstacle Tower environment.
    
    Args:
        args: Command line arguments
    """
    # Add default values for potentially missing args
    if not hasattr(args, 'render'):
        args.render = True
    if not hasattr(args, 'deterministic'):
        args.deterministic = 0.9
    if not hasattr(args, 'max_ep_steps'):
        args.max_ep_steps = 2000
    if not hasattr(args, 'episodes'):
        args.episodes = 5
    if not hasattr(args, 'video_path'):
        args.video_path = None
    if not hasattr(args, 'floor'):
        args.floor = None
    if not hasattr(args, 'seed'):
        args.seed = None
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create environment with visualization if requested
    env = create_obstacle_tower_env(
        executable_path=args.env_path,
        realtime_mode=True,  # Always use realtime for visualization
        no_graphics=not args.render  # Allow graphics for rendering
    )
    print("Environment created")
    
    # Set up action flattener
    action_flattener = ActionFlattener(env.action_space.nvec)
    num_actions = action_flattener.action_space.n
    print(f"Number of actions: {num_actions}")

    # Create model
    model = PPONetwork(input_shape=(12, 84, 84), num_actions=num_actions).to(device)
    
    # Load checkpoint
    if not args.checkpoint:
        print("Error: Must provide a checkpoint to evaluate")
        return
    
    print(f"Loading checkpoint from {args.checkpoint}")
    model, _, _ = load_checkpoint(model, args.checkpoint)
    model.eval()  # Set to evaluation mode
    
    # Initialize video frames list if needed
    if args.render and args.video_path:
        if not hasattr(main, 'video_frames'):
            main.video_frames = []
    
    # Check visualization capabilities
    can_visualize = args.render and CV2_AVAILABLE
    if args.render and not CV2_AVAILABLE:
        print("Warning: OpenCV (cv2) not available. Live visualization disabled.")
        if not args.video_path:
            print("No video path specified and OpenCV not available. Disabling rendering.")
            args.render = False
    
    # Run episodes
    total_rewards = []
    max_floors = []
    episode_floors = []
    
    print(f"Running {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        # Initialize frame stack
        frame_stack = deque(maxlen=4)
        episode_reward = 0
        episode_steps = 0
        max_floor = 0
        floors_visited = set()
        
        # Reset environment
        obs = env.reset()
        
        # Apply seed if specified
        if args.seed is not None:
            env.seed(args.seed)
        
        # Apply starting floor if specified
        if args.floor is not None:
            env.floor(args.floor)
            
        # Process observation
        obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
        for _ in range(4):
            frame_stack.append(obs)
        state = np.concatenate(frame_stack, axis=0)
        obs_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        
        done = False
        
        # Run episode
        while not done:
            # Select action
            with torch.no_grad():
                obs_batched = obs_tensor.unsqueeze(0)
                policy_logits, value = model(obs_batched)
                
                # Use more deterministic policy for visualization
                # Less random for clearer demonstration
                if np.random.random() < args.deterministic:
                    action_idx = torch.argmax(policy_logits, dim=1).item()
                else:
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    action_idx = dist.sample().item()
                
                action = action_flattener.lookup_action(action_idx)
                action = np.array(action)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Handle rendering
            if args.render:
                render_img = next_obs[0]
                
                # Save frame for video if requested
                if args.video_path:
                    if not hasattr(main, 'video_frames'):
                        main.video_frames = []
                    main.video_frames.append(render_img.copy())  # Copy to avoid reference issues
                
                # Show frame if visualization is enabled
                if can_visualize:
                    try:
                        cv2.imshow('Obstacle Tower', render_img[..., ::-1])  # RGB to BGR for OpenCV
                        cv2.waitKey(1)
                    except Exception as e:
                        print(f"Error displaying frame: {e}")
                        can_visualize = False  # Disable visualization if it fails
            
            # Update statistics
            current_floor = info["current_floor"]
            if current_floor > max_floor:
                max_floor = current_floor
            floors_visited.add(current_floor)
            
            episode_reward += reward
            episode_steps += 1
            
            # Process next observation
            next_obs = np.transpose(next_obs[0], (2, 0, 1)) / 255.0
            frame_stack.append(next_obs)
            next_state = np.concatenate(frame_stack, axis=0)
            obs_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
            
            # Print progress
            if episode_steps % 100 == 0 or done:
                print(f"Episode {episode+1}, Step {episode_steps}, Floor {current_floor}, Reward {episode_reward:.2f}")
                
            # Optional timeout
            if args.max_ep_steps and episode_steps >= args.max_ep_steps:
                print(f"Episode {episode+1} timed out after {episode_steps} steps")
                break
        
        # Record episode results
        total_rewards.append(episode_reward)
        max_floors.append(max_floor)
        episode_floors.append(list(floors_visited))
        
        print(f"Episode {episode+1} complete: Reward={episode_reward:.2f}, Max Floor={max_floor}, Steps={episode_steps}")
    
    # Print summary
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_floor = sum(max_floors) / len(max_floors)
    max_floor_overall = max(max_floors)
    
    print("\n" + "="*50)
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Max Floor: {avg_floor:.2f}")
    print(f"Maximum Floor Reached: {max_floor_overall}")
    print("Floor Progression:")
    for i, floors in enumerate(episode_floors):
        print(f"  Episode {i+1}: {sorted(floors)}")
    print("="*50)
    
    # Save video if requested
    if args.render and args.video_path and hasattr(main, 'video_frames') and len(main.video_frames) > 0:
        try:
            frames = main.video_frames
            print(f"Saving video with {len(frames)} frames...")
            height, width, _ = frames[0].shape
            
            # Try to use cv2 if available
            if CV2_AVAILABLE:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video = cv2.VideoWriter(args.video_path, fourcc, 30, (width, height))
                    
                    for frame in frames:
                        video.write(frame[..., ::-1])  # RGB to BGR
                        
                    video.release()
                    print(f"Video saved to {args.video_path}")
                except Exception as cv_err:
                    print(f"Error saving video with OpenCV: {cv_err}")
                    raise  # Re-raise to try fallback
            else:
                raise ImportError("OpenCV not available")
                
        except Exception as e:
            # Fallback to imageio if cv2 fails
            if IMAGEIO_AVAILABLE:
                try:
                    print("Falling back to imageio for video saving...")
                    imageio.mimsave(args.video_path, frames, fps=30)
                    print(f"Video saved to {args.video_path} using imageio")
                except Exception as img_err:
                    print(f"Error saving video with imageio: {img_err}")
            else:
                print(f"Could not save video - both OpenCV and imageio unavailable")
    
    # Clean up
    env.close()
    if CV2_AVAILABLE:
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained agent on Obstacle Tower')
    
    # Environment settings
    parser.add_argument('--env_path', type=str, default="./ObstacleTower/obstacletower.x86_64",
                        help='Path to Obstacle Tower executable')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for environment (optional)')
    parser.add_argument('--floor', type=int, default=None,
                        help='Starting floor (optional)')
    
    # Model settings
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--deterministic', type=float, default=0.9,
                        help='Probability of choosing the most likely action')
    
    # Evaluation settings
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--max_ep_steps', type=int, default=1000,
                        help='Maximum steps per episode')
    
    # Visualization settings
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to save video (requires --render)')
    
    args = parser.parse_args()
    
    main(args) 