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
import matplotlib.pyplot as plt
from PIL import Image

# Ensure cv2 is conditionally imported only when needed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV (cv2) not available - visualization features will be limited")

# Try to import imageio as fallback for video creation
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("imageio not available - fallback video saving will not work")

def evaluate_agent(env, model, action_flattener, device, episodes=10, max_steps=1000, render=False, realtime_mode=False, 
                  starting_floor=None, video_path=None):
    """
    Evaluate the agent for a number of episodes
    """
    # Set rendering and realtime modes properly
    if hasattr(env, 'engine_config'):
        # Set time scale based on realtime_mode
        if realtime_mode:
            env.engine_config.set_configuration_parameters(time_scale=1.0)
        else:
            env.engine_config.set_configuration_parameters(time_scale=20.0)

    if starting_floor is not None:
        env.floor(starting_floor)
    
    # Get action dimension from the flattener
    action_dim = action_flattener.action_space.n
    
    returns = []
    floors_reached = []
    steps_per_episode = []
    
    # For video recording
    if render and video_path and CV2_AVAILABLE:
        video_frames = []
    else:
        video_frames = None

    visualization_enabled = render and CV2_AVAILABLE

    for i in range(episodes):
        print(f"Starting evaluation episode {i+1}/{episodes}")
        obs = env.reset()
        
        # Initialize frame stack with 4 copies of the first observation
        frame_stack = deque(maxlen=4)
        if isinstance(obs, tuple):
            # Handle Obstacle Tower's tuple observation
            img_obs = obs[0]
        else:
            img_obs = obs
            
        # Normalize and convert to channels-first format (transpose if needed)
        img_obs = np.transpose(img_obs, (2, 0, 1)) / 255.0  # Convert to channels-first and normalize
        
        # Fill the stack with initial frame
        for _ in range(4):
            frame_stack.append(img_obs)
            
        # Stack the frames to create input with shape (12, 84, 84)
        state = np.concatenate(list(frame_stack), axis=0)
        
        # Convert to torch tensor and move to device
        state = torch.FloatTensor(state).to(device)
        if len(state.shape) == 3:
            state = state.unsqueeze(0)  # Add batch dimension
            
        episode_return = 0
        episode_steps = 0
        done = False
        current_floor = 0

        # Initialize display window only if rendering is enabled
        if visualization_enabled:
            try:
                cv2.namedWindow('Agent Evaluation', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Agent Evaluation', 640, 480)
            except Exception as e:
                print(f"Warning: Could not initialize OpenCV window: {e}")
                visualization_enabled = False

        # Main evaluation loop
        while not done and episode_steps < max_steps:
            episode_steps += 1
            
            # Get action from policy
            with torch.no_grad():
                policy_logits, value = model(state)
                action_idx = torch.argmax(policy_logits, dim=1).cpu().numpy()[0]
                
                # Convert action index to actual action using the flattener
                action = action_flattener.lookup_action(action_idx)
                action = np.array(action)
                
            # Step environment with selected action
            obs, reward, done, info = env.step(action)
            
            if isinstance(obs, tuple):
                img_obs = obs[0]
            else:
                img_obs = obs

            # Track current floor
            if 'current_floor' in info and info['current_floor'] > current_floor:
                current_floor = info['current_floor']
                print(f"Reached floor {current_floor} on episode {i+1}")

            # Render if requested
            if visualization_enabled:
                try:
                    # For display - use the raw observation for visualization
                    display_img = np.array(img_obs).astype(np.uint8)
                    if display_img.shape[-1] == 1:  # If grayscale, convert to RGB
                        display_img = np.repeat(display_img, 3, axis=-1)
                        
                    # Add text with info
                    if CV2_AVAILABLE:
                        # Add text with environment information
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(display_img, f"Floor: {info.get('current_floor', 0)}", (10, 20), font, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_img, f"Keys: {info.get('total_keys', 0)}", (10, 40), font, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_img, f"Time: {info.get('time_remaining', 0):.1f}", (10, 60), font, 0.5, (255, 255, 255), 1)
                        cv2.putText(display_img, f"Reward: {episode_return:.1f}", (10, 80), font, 0.5, (255, 255, 255), 1)
                        
                        # Show the image
                        cv2.imshow('Agent Evaluation', display_img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                        
                        # Save for video if requested
                        if video_path:
                            video_frames.append(display_img.copy())
                        
                        # Handle window events and add delay if in realtime mode
                        key = cv2.waitKey(1 if realtime_mode else 1)
                        if key == 27:  # ESC key
                            print("Evaluation terminated by user")
                            break
                except Exception as e:
                    print(f"Warning: Visualization error: {e}")
                    visualization_enabled = False  # Disable visualization if it fails
            
            # Process the new observation for the model
            img_obs = np.transpose(img_obs, (2, 0, 1)) / 255.0  # Convert to channels-first and normalize
            frame_stack.append(img_obs)
            
            # Stack the frames to create input with shape (12, 84, 84)
            next_state = np.concatenate(list(frame_stack), axis=0)
            
            # Convert to torch tensor and move to device
            state = torch.FloatTensor(next_state).to(device)
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            
            episode_return += reward

            # For demonstration purpose, print progress every 100 steps
            if episode_steps % 100 == 0:
                print(f"Episode {i+1}, Step {episode_steps}, Current Floor: {info.get('current_floor', 0)}, Return: {episode_return:.2f}")
        
        # Episode complete
        returns.append(episode_return)
        floors_reached.append(current_floor)
        steps_per_episode.append(episode_steps)
        print(f"Episode {i+1} complete - Return: {episode_return:.2f}, Floor: {current_floor}, Steps: {episode_steps}")

    # Close OpenCV window if opened
    if visualization_enabled:
        cv2.destroyAllWindows()

    # Save video if we've collected frames
    if render and video_path and video_frames:
        try:
            if CV2_AVAILABLE:
                print(f"Saving video to {video_path}...")
                height, width, layers = video_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
                
                for frame in video_frames:
                    video.write(frame[:, :, ::-1])  # Convert RGB to BGR for OpenCV
                video.release()
                print(f"Video saved to {video_path}")
            elif IMAGEIO_AVAILABLE:
                print(f"OpenCV video writing failed. Trying imageio as fallback...")
                imageio.mimsave(video_path, video_frames, fps=30)
                print(f"Video saved using imageio to {video_path}")
            else:
                print(f"Cannot save video - neither OpenCV nor imageio are available")
        except Exception as e:
            print(f"Error saving video: {e}")

    # Return evaluation results
    mean_return = np.mean(returns)
    mean_floor = np.mean(floors_reached)
    mean_steps = np.mean(steps_per_episode)
    
    results = {
        'returns': returns,
        'mean_return': mean_return,
        'floors_reached': floors_reached,
        'mean_floor': mean_floor,
        'steps_per_episode': steps_per_episode,
        'mean_steps': mean_steps
    }
    
    print(f"Evaluation complete - Mean return: {mean_return:.2f}, Mean floor: {mean_floor:.2f}, Mean steps: {mean_steps:.2f}")
    return results

def evaluate_main(args=None):
    """
    Main evaluation function
    """
    if args is None:
        parser = argparse.ArgumentParser(description='Evaluate a trained ObstacleTower agent')
        parser.add_argument('--env_path', type=str, default='./ObstacleTower/obstacletower.x86_64', 
                            help='Path to ObstacleTower executable')
        parser.add_argument('--seed', type=int, default=0, help='Random seed')
        parser.add_argument('--floor', type=int, default=None, help='Starting floor')
        parser.add_argument('--checkpoint', type=str, required=True, help='Path to agent checkpoint')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                            help='Device to use (cuda or cpu)')
        parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
        parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
        parser.add_argument('--max_ep_steps', type=int, default=1000, help='Maximum steps per episode')
        parser.add_argument('--render', action='store_true', help='Render the environment')
        parser.add_argument('--realtime_mode', action='store_true', help='Run in realtime mode')
        parser.add_argument('--video_path', type=str, default=None, help='Path to save video of evaluation')
        args = parser.parse_args()

    print("Starting evaluation with parameters:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create environment
    env = create_obstacle_tower_env(
        executable_path=args.env_path,
        realtime_mode=args.realtime_mode,
        timeout=300,
        no_graphics=not args.render
    )
    print(f"Environment created. Action space: {env.action_space}, Observation space: {env.observation_space}")

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.seed is not None:
        env.seed(args.seed)
    
    # Set starting floor if specified
    if args.floor is not None:
        env.floor(args.floor)
        print(f"Setting starting floor to {args.floor}")

    # Create action flattener for the MultiDiscrete action space
    if hasattr(env.action_space, 'n'):
        action_flattener = None
        action_dim = env.action_space.n
    else:
        action_flattener = ActionFlattener(env.action_space.nvec)
        action_dim = action_flattener.action_space.n
    
    # Create the model with input shape (12, 84, 84) for 4 stacked RGB frames
    input_shape = (12, 84, 84)  # Fixed shape that the model expects
    print(f"Using input shape: {input_shape}, Action dimension: {action_dim}")
    model = PPONetwork(input_shape=input_shape, num_actions=action_dim).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        model, _, _ = load_checkpoint(model, args.checkpoint)
        model.eval()  # Set to evaluation mode
    
    # Run evaluation
    eval_results = evaluate_agent(
        env=env,
        model=model,
        action_flattener=action_flattener,
        device=device,
        episodes=args.episodes,
        max_steps=args.max_ep_steps,
        render=args.render,
        realtime_mode=args.realtime_mode,
        starting_floor=args.floor,
        video_path=args.video_path
    )
    
    # Close environment
    env.close()
    
    return eval_results

if __name__ == "__main__":
    evaluate_main() 