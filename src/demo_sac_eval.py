"""
Evaluation script for trained DemoSAC agents on Obstacle Tower.

This script allows evaluating a trained agent without training:
- Load a saved model
- Run it for a specified number of episodes
- Compute and display performance metrics
- Optionally record evaluation videos

Run with: python -m src.demo_sac_eval --model-path path/to/model [--args]
"""
import os
import time
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecVideoRecorder

# Import custom modules
from config import *
from src.models.rnd_model import RNDModel
from src.agents.demo_sac import DemoSAC
from src.sac_utils .demo_buffer import DemonstrationBuffer
from src.envs.wrappers import make_obstacle_tower_env
from src.models.networks import create_feature_extractor

try:
    from obstacle_tower_env import ObstacleTowerEnv
except ImportError:
    print("Warning: obstacle_tower_env not found. Make sure it's installed for evaluation.")
    # Define a placeholder for development without the environment
    class ObstacleTowerEnv:
        def __init__(self, *args, **kwargs):
            pass


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate a trained DemoSAC agent on Obstacle Tower")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True, 
                        help="Path to saved model")
    
    # Environment settings
    parser.add_argument("--env-path", type=str, default=None, 
                        help="Path to Obstacle Tower executable")
    parser.add_argument("--workers", type=int, default=ENV_CONFIG["worker_id"],
                        help="Worker ID for environment")
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["seed"],
                        help="Random seed")
    
    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["n_eval_episodes"],
                        help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", default=EVAL_CONFIG["deterministic"],
                        help="Use deterministic actions")
    parser.add_argument("--render", action="store_true", default=EVAL_CONFIG["render"],
                        help="Render environment")
    parser.add_argument("--max-steps", type=int, default=TRAIN_CONFIG["max_episode_steps"],
                        help="Maximum steps per episode")
    
    # Video recording
    parser.add_argument("--record", action="store_true", 
                        help="Record video of evaluation")
    parser.add_argument("--video-dir", type=str, default="videos",
                        help="Directory to save videos")
    parser.add_argument("--video-length", type=int, default=3000,
                        help="Maximum length of recorded video (in steps)")
    
    # Analysis
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate performance plots")
    
    # Environment wrappers
    parser.add_argument("--use-custom-wrappers", action="store_true",
                        help="Use custom environment wrappers")
    parser.add_argument("--reward-shaping", action="store_true",
                        help="Enable reward shaping in custom wrappers")
    parser.add_argument("--frame-skip", type=int, default=4,
                        help="Number of frames to skip (action repeat)")
    
    args = parser.parse_args()
    return args


def make_env(env_path=None, worker_id=1, realtime_mode=False, config=None):
    """
    Create and configure Obstacle Tower environment.
    
    Args:
        env_path: Path to Obstacle Tower executable
        worker_id: Worker ID for environment
        realtime_mode: Whether to run in realtime
        config: Environment configuration
        
    Returns:
        callable: Function that creates and returns the environment
    """
    if config is None:
        config = ENV_CONFIG
    
    def _init():
        env = ObstacleTowerEnv(
            environment_path=env_path,
            worker_id=worker_id,
            retro=config["retro"],
            realtime_mode=realtime_mode,
            timeout_wait=config["timeout_wait"],
            docker_training=config["docker_training"],
        )
        # Set random seed
        env.seed(config["seed"])
        
        return env
    
    return _init


def preprocess_obstacle_tower(env, stack_frames=4, grayscale=True, new_size=(84, 84), 
                              record=False, video_dir=None, video_length=3000):
    """
    Apply preprocessing wrappers to Obstacle Tower environment.
    Optionally wrap in VecVideoRecorder for recording.
    
    Args:
        env: Environment to preprocess
        stack_frames: Number of frames to stack
        grayscale: Whether to convert to grayscale
        new_size: New resolution (height, width)
        record: Whether to record video
        video_dir: Directory to save videos
        video_length: Maximum video length in steps
        
    Returns:
        VecEnv: Preprocessed vectorized environment
    """
    # Import here to avoid circular imports
    from src.demo_sac_train import preprocess_obstacle_tower as preprocess_fn
    
    # Apply standard preprocessing
    env = preprocess_fn(env, stack_frames, grayscale, new_size)
    
    # Wrap in video recorder if requested
    if record and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"obstacle_tower_eval_{timestamp}")
        env = VecVideoRecorder(
            env,
            video_folder=video_path,
            record_video_trigger=lambda x: x == 0,  # Record from the start
            video_length=video_length,
            name_prefix="obstacle_tower_demo_sac"
        )
        print(f"Recording video to {video_path}")
    
    return env


def create_env_with_custom_wrappers(env_fn, config, use_reward_shaping=True, 
                                  frame_skip=4, max_steps=1000,
                                  record=False, video_dir=None, video_length=3000):
    """
    Create environment with custom wrappers from src.envs.wrappers
    
    Args:
        env_fn: Function that creates the base environment
        config: Environment configuration
        use_reward_shaping: Whether to enable reward shaping
        frame_skip: Number of frames to skip (action repeat)
        max_steps: Maximum steps per episode
        record: Whether to record video
        video_dir: Directory to save videos
        video_length: Maximum video length in steps
        
    Returns:
        VecEnv: Preprocessed vectorized environment
    """
    # Create base environment
    env = env_fn()
    
    # Apply custom wrappers
    env = make_obstacle_tower_env(
        env,
        grayscale=config["grayscale"],
        resize=config["resolution"],
        normalize=True,
        frame_stack=config["stack_frames"],
        frame_skip=frame_skip,
        reward_shaping=use_reward_shaping,
        time_limit=max_steps,
        log_info=True
    )
    
    # Vectorize for stable-baselines compatibility
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)
    
    # Wrap in video recorder if requested
    if record and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(video_dir, f"obstacle_tower_eval_{timestamp}")
        env = VecVideoRecorder(
            env,
            video_folder=video_path,
            record_video_trigger=lambda x: x == 0,  # Record from the start
            video_length=video_length,
            name_prefix="obstacle_tower_demo_sac"
        )
        print(f"Recording video to {video_path}")
    
    return env


def evaluate_agent(model, env, n_episodes=5, deterministic=True, render=False, max_steps=1000):
    """
    Evaluate agent for a specified number of episodes.
    
    Args:
        model: Trained model
        env: Environment to evaluate in
        n_episodes: Number of episodes
        deterministic: Whether to use deterministic actions
        render: Whether to render environment
        max_steps: Maximum steps per episode
        
    Returns:
        dict: Evaluation results
    """
    episode_rewards = []
    episode_lengths = []
    floors_reached = []
    keys_collected = []
    doors_opened = []
    time_per_step = []
    
    print(f"Evaluating agent for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        step_count = 0
        floor, keys, doors = 0, 0, 0
        
        while not done and step_count < max_steps:
            start_time = time.time()
            
            # Get action from model
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            end_time = time.time()
            
            episode_reward += reward
            step_count += 1
            time_per_step.append(end_time - start_time)
            
            # Extract Obstacle Tower specific metrics
            if "floor" in info[0]:
                floor = max(floor, info[0]["floor"])
            if "keys_collected" in info[0]:
                keys = max(keys, info[0]["keys_collected"])
            if "doors_opened" in info[0]:
                doors = max(doors, info[0]["doors_opened"])
            
            if render:
                env.render()
            
            # If this is a vectorized environment, check if the episode is done
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        floors_reached.append(floor)
        keys_collected.append(keys)
        doors_opened.append(doors)
        
        print(f"Episode {episode+1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, Steps={step_count}, "
              f"Floor={floor}, Keys={keys}, Doors={doors}")
    
    # Calculate average metrics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_floor = np.mean(floors_reached)
    avg_keys = np.mean(keys_collected)
    avg_doors = np.mean(doors_opened)
    avg_step_time = np.mean(time_per_step) if time_per_step else 0
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    print(f"Average Floor Reached: {avg_floor:.2f}")
    print(f"Average Keys Collected: {avg_keys:.2f}")
    print(f"Average Doors Opened: {avg_doors:.2f}")
    print(f"Average Time per Step: {avg_step_time*1000:.2f} ms")
    print(f"Steps per Second: {1/avg_step_time:.2f}")
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "floors_reached": floors_reached,
        "keys_collected": keys_collected,
        "doors_opened": doors_opened,
        "time_per_step": time_per_step,
        "avg_reward": avg_reward,
        "avg_length": avg_length,
        "avg_floor": avg_floor,
        "avg_keys": avg_keys,
        "avg_doors": avg_doors,
        "avg_step_time": avg_step_time
    }


def generate_plots(results, output_dir):
    """
    Generate and save performance plots.
    
    Args:
        results: Evaluation results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame for easier plotting
    data = pd.DataFrame({
        "Episode": np.arange(1, len(results["episode_rewards"]) + 1),
        "Reward": results["episode_rewards"],
        "Length": results["episode_lengths"],
        "Floor": results["floors_reached"],
        "Keys": results["keys_collected"],
        "Doors": results["doors_opened"]
    })
    
    # Save data as CSV
    data.to_csv(os.path.join(output_dir, f"eval_results_{timestamp}.csv"), index=False)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(data["Episode"], data["Reward"], marker='o')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"rewards_{timestamp}.png"))
    
    # Plot floors, keys, doors
    plt.figure(figsize=(10, 6))
    plt.plot(data["Episode"], data["Floor"], marker='o', label="Floors")
    plt.plot(data["Episode"], data["Keys"], marker='s', label="Keys")
    plt.plot(data["Episode"], data["Doors"], marker='^', label="Doors")
    plt.title("Game Progress Metrics")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"progress_metrics_{timestamp}.png"))
    
    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(data["Episode"], data["Length"], marker='o')
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"episode_lengths_{timestamp}.png"))
    
    print(f"Plots saved to {output_dir}")


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directories
    if args.record:
        os.makedirs(args.video_dir, exist_ok=True)
    if args.plot:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return
    
    # Create and preprocess environment
    print("Creating environment...")
    env_fn = make_env(
        env_path=args.env_path,
        worker_id=args.workers,
        realtime_mode=args.render,
        config=ENV_CONFIG
    )
    
    # Use custom wrappers if requested
    if args.use_custom_wrappers:
        print("Using custom environment wrappers")
        env = create_env_with_custom_wrappers(
            env_fn=env_fn,
            config=ENV_CONFIG,
            use_reward_shaping=args.reward_shaping,
            frame_skip=args.frame_skip,
            max_steps=args.max_steps,
            record=args.record,
            video_dir=args.video_dir,
            video_length=args.video_length
        )
    else:
        env = env_fn()
        env = preprocess_obstacle_tower(
            env,
            stack_frames=ENV_CONFIG["stack_frames"],
            grayscale=ENV_CONFIG["grayscale"],
            new_size=ENV_CONFIG["resolution"],
            record=args.record,
            video_dir=args.video_dir,
            video_length=args.video_length
        )
    
    # Load demonstration buffer for agent that expects it
    demo_buffer = None
    try:
        demo_path = f"{args.model_path}_demos.pkl"
        if os.path.exists(demo_path):
            print(f"Loading demonstrations from {demo_path}...")
            demo_buffer = DemonstrationBuffer()
            demo_buffer.load(demo_path)
    except Exception as e:
        print(f"Warning: Could not load demonstrations: {e}")
    
    # Load RND model if available
    rnd_model = None
    try:
        rnd_path = f"{args.model_path}_rnd.pt"
        if os.path.exists(rnd_path):
            print(f"Loading RND model from {rnd_path}...")
            # Get observation shape from environment
            obs_shape = env.observation_space.shape
            rnd_model = RNDModel(observation_shape=obs_shape, config=RND_CONFIG)
            rnd_model.load_state_dict(torch.load(rnd_path, map_location="cpu"))
    except Exception as e:
        print(f"Warning: Could not load RND model: {e}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        # Try to load as DemoSAC first
        model = DemoSAC.load(
            args.model_path,
            env=env,
            rnd_model=rnd_model,
            demo_buffer=demo_buffer
        )
    except Exception as e:
        print(f"Warning: Could not load as DemoSAC, trying generic load: {e}")
        # Fallback to generic load
        from stable_baselines3 import SAC
        model = SAC.load(args.model_path, env=env)
    
    print("Model loaded successfully")
    
    # Evaluate agent
    results = evaluate_agent(
        model=model,
        env=env,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
        max_steps=args.max_steps
    )
    
    # Generate plots if requested
    if args.plot:
        generate_plots(results, args.output_dir)
    
    # Close environment
    env.close()
    
    print("Evaluation completed successfully")


if __name__ == "__main__":
    main()