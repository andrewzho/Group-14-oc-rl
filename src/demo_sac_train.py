"""
Main training script for Demonstration-boosted SAC with RND for Obstacle Tower.

This script combines all components:
- Obstacle Tower environment with preprocessing
- Random Network Distillation for intrinsic rewards
- Demonstration learning from expert examples
- SAC reinforcement learning algorithm

Run with: python -m src.demo_sac_train [--args]
"""
# Import the numpy patch first to fix np.bool deprecation
from src.np_patch import *
import os
import time
import argparse
import numpy as np
import torch
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Import custom modules
from config import *
from src.models.rnd_model import RNDModel
from src.agents.demo_sac import DemoSAC, RNDUpdateCallback
from src.sac_utils .demo_buffer import DemonstrationBuffer
from src.sac_utils .callbacks import (
    TensorboardCallback, DemonstrationRecorderCallback,
    ObstacleTowerEvalCallback, SaveCheckpointCallback
)
from src.envs.wrappers import make_obstacle_tower_env
from src.models.networks import create_feature_extractor
from src.sac_utils .preprocessing import create_obstacle_tower_preprocessing_pipeline

from obstacle_tower_env import ObstacleTowerEnv


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Demonstration-boosted SAC with RND for Obstacle Tower")
    
    # Environment settings
    parser.add_argument("--env-path", type=str, default=None, 
                        help="Path to Obstacle Tower executable")
    parser.add_argument("--workers", type=int, default=ENV_CONFIG["worker_id"],
                        help="Worker ID for environment")
    parser.add_argument("--seed", type=int, default=ENV_CONFIG["seed"],
                        help="Random seed")
    parser.add_argument("--graphics", action="store_true",
                        help="Enable graphics (default: disabled)")
    
    # Demo settings
    parser.add_argument("--demo-path", type=str, default=DEMO_CONFIG["demo_file"],
                        help="Path to demonstration file")
    parser.add_argument("--no-demos", action="store_true", 
                        help="Disable demonstration learning")
    parser.add_argument("--pretrain-steps", type=int, default=DEMO_CONFIG["pretrain_steps"],
                        help="Number of pretraining steps on demonstrations")
    
    # RND settings
    parser.add_argument("--no-rnd", action="store_true", 
                        help="Disable RND intrinsic rewards")
    parser.add_argument("--rnd-coef", type=float, default=RND_CONFIG["intrinsic_reward_coef"],
                        help="Intrinsic reward coefficient")
    
    # Training settings
    parser.add_argument("--timesteps", type=int, default=TRAIN_CONFIG["total_timesteps"],
                        help="Total timesteps for training")
    parser.add_argument("--save-freq", type=int, default=TRAIN_CONFIG["save_freq"],
                        help="Save frequency in timesteps")
    parser.add_argument("--eval-freq", type=int, default=TRAIN_CONFIG["eval_freq"],
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR),
                        help="Directory for tensorboard logs")
    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                        help="Directory for saving models")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume training from")
    
    # Network architecture options
    parser.add_argument("--network", type=str, default="cnn",
                      choices=["cnn", "resnet", "lstm", "dual"],
                      help="Feature extractor architecture")
    parser.add_argument("--use-custom-wrappers", action="store_true",
                      help="Use custom environment wrappers")
    parser.add_argument("--reward-shaping", action="store_true",
                      help="Enable reward shaping in custom wrappers")
    parser.add_argument("--frame-skip", type=int, default=4,
                      help="Number of frames to skip (action repeat)")
    
    
    
    args = parser.parse_args()
    return args


def make_env(env_path=None, worker_id=1, realtime_mode=False, config=None, no_graphics=True):
    """
    Create and configure memory-efficient Obstacle Tower environment for SAC.
    
    Args:
        env_path: Path to Obstacle Tower executable
        worker_id: Worker ID for environment
        realtime_mode: Whether to run in realtime
        config: Environment configuration
        no_graphics: Whether to disable graphics
        
    Returns:
        callable: Function that creates and returns the environment
    """
    if config is None:
        config = ENV_CONFIG
    
    def _init():
        # Import the memory-efficient wrapper
        from src.envs.memory_efficient_wrapper import create_memory_efficient_env
        
        # Create memory-efficient environment
        env = create_memory_efficient_env(
            env_path=env_path,
            worker_id=worker_id,
            stack_frames=config["stack_frames"],
            seed=config["seed"]
        )
        
        return env
    
    return _init


def preprocess_obstacle_tower(env, stack_frames=4, grayscale=True, new_size=(84, 84)):
    """
    This function is no longer needed as the environment is already preprocessed
    by the ObstacleTowerGymWrapper. It's kept for compatibility.
    
    Args:
        env: Environment to preprocess
        stack_frames: Number of frames to stack
        grayscale: Whether to convert to grayscale
        new_size: New resolution (height, width)
        
    Returns:
        env: The environment (unchanged)
    """
    # The preprocessing is now done in the ObstacleTowerGymWrapper
    return env


def create_env_with_custom_wrappers(env_fn, config, use_reward_shaping=True, frame_skip=4, max_steps=1000):
    """
    Create environment with custom wrappers from src.envs.wrappers
    
    Args:
        env_fn: Function that creates the base environment
        config: Environment configuration
        use_reward_shaping: Whether to enable reward shaping
        frame_skip: Number of frames to skip (action repeat)
        max_steps: Maximum steps per episode
        
    Returns:
        VecTransposeImage: Preprocessed vectorized environment
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
    env = DummyVecEnv([lambda: env])
    
    # Transpose for PyTorch (channel first)
    env = VecTransposeImage(env)
    
    return env


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create timestamp for run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"demosac_rnd_{timestamp}"
    log_path = os.path.join(args.log_dir, run_name)
    model_path = os.path.join(args.model_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Starting training run: {run_name}")
    print(f"Log path: {log_path}")
    print(f"Model path: {model_path}")
    
    print("Creating environment...")
    env_fn = make_env(
        env_path=args.env_path,
        worker_id=args.workers,
        realtime_mode=False,
        config=ENV_CONFIG,
        no_graphics=not args.graphics
    )
    
    # Create the environment - no additional preprocessing needed
    env = env_fn()
    print(f"Created environment with observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Close this instance and create a new one with proper wrappers
    env.close()
    
    # Now create the environment again with the correct wrappers
    env = env_fn()
    
    # Use standard preprocessing - we'll add custom wrappers later once basic setup works
    env = preprocess_obstacle_tower(
        env,
        stack_frames=ENV_CONFIG["stack_frames"],
        grayscale=ENV_CONFIG["grayscale"],
        new_size=ENV_CONFIG["resolution"]
    )
    
    # Create evaluation environment (with different worker ID)
    eval_env_fn = make_env(
        env_path=args.env_path,
        worker_id=args.workers + 10,
        realtime_mode=False,
        config=ENV_CONFIG,
        no_graphics=not args.graphics
    )
    
    # Use same wrapper type for eval env
    if args.use_custom_wrappers:
        eval_env = create_env_with_custom_wrappers(
            env_fn=eval_env_fn,
            config=ENV_CONFIG,
            use_reward_shaping=args.reward_shaping,
            frame_skip=args.frame_skip,
            max_steps=TRAIN_CONFIG["max_episode_steps"]
        )
    else:
        eval_env = eval_env_fn()
        eval_env = preprocess_obstacle_tower(
            eval_env,
            stack_frames=ENV_CONFIG["stack_frames"],
            grayscale=ENV_CONFIG["grayscale"],
            new_size=ENV_CONFIG["resolution"]
        )
    
    # Determine observation shape
    obs_shape = env.observation_space.shape
    print(f"Observation shape: {obs_shape}")
    
    # Load or create demonstration buffer
    demo_buffer = None
    if not args.no_demos:
        print(f"Loading demonstrations from {args.demo_path}...")
        demo_buffer = DemonstrationBuffer()
        if os.path.exists(args.demo_path):
            success = demo_buffer.load(args.demo_path)
            if success:
                print(f"Loaded {demo_buffer.total_transitions} demonstration transitions")
            else:
                print("Failed to load demonstrations, creating new buffer")
        else:
            print(f"Demonstration file not found, creating new buffer")
    
    # Create RND model for intrinsic rewards
    rnd_model = None
    if not args.no_rnd:
        print("Creating RND model...")
        rnd_config = RND_CONFIG.copy()
        rnd_config["intrinsic_reward_coef"] = args.rnd_coef
        rnd_model = RNDModel(observation_shape=obs_shape, config=rnd_config)
    
    # Create or load agent
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Loading model from {args.resume}...")
        model = DemoSAC.load(
            args.resume,
            env=env,
            demo_buffer=demo_buffer,
            rnd_model=rnd_model,
            pretrain_steps=0,  # Skip pretraining for resumed model
            tensorboard_log=log_path,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully")
    else:
        print("Creating new model...")
        # Set demo config
        demo_kwargs = {}
        if not args.no_demos and demo_buffer is not None:
            demo_kwargs = {
                "demo_buffer": demo_buffer,
                "demo_batch_size": DEMO_CONFIG["demo_batch_size"],
                "bc_loss_coef": DEMO_CONFIG["bc_loss_coef"],
                "pretrain_steps": args.pretrain_steps
            }
        
        # Set RND config
        rnd_kwargs = {}
        if not args.no_rnd and rnd_model is not None:
            rnd_kwargs = {
                "rnd_model": rnd_model,
                "intrinsic_reward_coef": args.rnd_coef
            }
        
        # Set up policy kwargs with custom network if requested
        policy_kwargs = {
            "normalize_images": False,  # Important: Images are already normalized and in channel-first format
        }

        # Add additional network settings if using custom network
        if args.network != "cnn":
            print(f"Using custom {args.network} network architecture")
            from src.models.networks import create_feature_extractor
            
            # Set up custom feature extractor
            policy_kwargs["features_extractor_class"] = lambda obs_space: create_feature_extractor(
                args.network, 
                obs_space,
                features_dim=NETWORK_CONFIG["hidden_dim"]
            )
            policy_kwargs["features_extractor_kwargs"] = {}  # Empty dict since we pass params directly above
            
            # Special handling for LSTM networks
            if args.network == "lstm":
                policy_kwargs["features_extractor_kwargs"]["lstm_hidden_size"] = 256
                policy_kwargs["features_extractor_kwargs"]["lstm_layers"] = 1
                
            # Special handling for dual encoder networks
            elif args.network == "dual":
                policy_kwargs["features_extractor_kwargs"]["state_dim"] = 8
                
        # Create model
        model = DemoSAC(
            policy=SAC_CONFIG["policy_type"],
            env=env,
            learning_rate=SAC_CONFIG["learning_rate"],
            buffer_size=SAC_CONFIG["buffer_size"],
            learning_starts=SAC_CONFIG["learning_starts"],
            batch_size=SAC_CONFIG["batch_size"],
            tau=SAC_CONFIG["tau"],
            gamma=SAC_CONFIG["gamma"],
            train_freq=SAC_CONFIG["train_freq"],
            gradient_steps=SAC_CONFIG["gradient_steps"],
            ent_coef=SAC_CONFIG["ent_coef"],
            target_update_interval=SAC_CONFIG["target_update_interval"],
            tensorboard_log=log_path,
            verbose=SAC_CONFIG["verbose"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs=policy_kwargs,
            **demo_kwargs,
            **rnd_kwargs
        )
    
    # Create callbacks
    callbacks = []
    
    # TensorBoard logging
    callbacks.append(TensorboardCallback())
    
    # RND update callback
    if not args.no_rnd and rnd_model is not None:
        callbacks.append(RNDUpdateCallback(
            rnd_model=rnd_model,
            update_freq=1000,
            verbose=1
        ))
    
    # Demonstration recording callback
    if not args.no_demos and demo_buffer is not None:
        callbacks.append(DemonstrationRecorderCallback(
            demo_buffer=demo_buffer,
            min_reward_percentile=80.0,
            min_floor=1,
            verbose=1
        ))
    
    # Evaluation callback
    callbacks.append(ObstacleTowerEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=EVAL_CONFIG["n_eval_episodes"],
        eval_freq=args.eval_freq,
        best_model_save_path=model_path,
        log_path=log_path,
        deterministic=EVAL_CONFIG["deterministic"],
        verbose=1
    ))
    
    # Checkpoint saving
    callbacks.append(SaveCheckpointCallback(
        save_path=model_path,
        save_freq=args.save_freq,
        save_demo_buffer=not args.no_demos,
        save_rnd_model=not args.no_rnd,
        verbose=1
    ))
    
    # Combine all callbacks
    callback = CallbackList(callbacks)
    
    # Print configuration summary
    print("\nTraining Configuration:")
    print(f"- Network Architecture: {args.network}")
    print(f"- Custom Wrappers: {'Enabled' if args.use_custom_wrappers else 'Disabled'}")
    if args.use_custom_wrappers:
        print(f"- Reward Shaping: {'Enabled' if args.reward_shaping else 'Disabled'}")
        print(f"- Frame Skip: {args.frame_skip}")
    print(f"- Demonstrations: {'Disabled' if args.no_demos else 'Enabled'}")
    print(f"- Random Network Distillation: {'Disabled' if args.no_rnd else 'Enabled'}")
    print(f"- Training Timesteps: {args.timesteps}")
    print()
    
    # Train the agent
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        log_interval=TRAIN_CONFIG["log_interval"],
    )
    
    # Save the final model
    final_model_path = os.path.join(model_path, "final_model")
    print(f"Saving final model to {final_model_path}")
    model.save(final_model_path)
    
    # Save the final demonstration buffer
    if not args.no_demos and demo_buffer is not None:
        final_demo_path = os.path.join(model_path, "final_demonstrations.pkl")
        print(f"Saving final demonstration buffer to {final_demo_path}")
        demo_buffer.save(final_demo_path)
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()