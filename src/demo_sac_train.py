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
import traceback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

# Import custom modules
from config import *
from src.models.rnd_model import RNDModel
from src.agents.demo_sac import DemoSAC, RNDUpdateCallback
from src.sac_utils.demo_buffer import DemonstrationBuffer
from src.sac_utils.callbacks import (
    TensorboardCallback, DemonstrationRecorderCallback,
    ObstacleTowerEvalCallback, SaveCheckpointCallback,
    EarlyStoppingCallback, TrainingProgressCallback
)
from src.envs.wrappers import make_obstacle_tower_env
from src.utils import validate_parameters, setup_network_architecture, create_memory_efficient_env, ConsoleLogger

# Add a proper Gym-inheriting compatibility wrapper
import gym

class GymCompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to handle compatibility between old and new Gym versions.
    Old Gym (< 0.26.0): step() returns (obs, reward, done, info)
    New Gym (>= 0.26.0): step() returns (obs, reward, terminated, truncated, info)
    
    This wrapper inherits from gym.Wrapper to maintain compatibility with SB3.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def reset(self, **kwargs):
        try:
            # Try new API first (obs, info)
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0]  # Just return the observation
            return result  # If it's not a tuple, return as is
        except TypeError:
            # Fall back to old API (just obs)
            return self.env.reset()
    
    def step(self, action):
        result = self.env.step(action)
        
        # Check the shape of the result
        if isinstance(result, tuple):
            if len(result) == 5:  # New Gym API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                return obs, reward, done, info
            elif len(result) == 4:  # Old Gym API: obs, reward, done, info
                return result
            else:
                raise ValueError(f"Unexpected result format from env.step(): got {len(result)} values")
        else:
            raise ValueError("env.step() did not return a tuple")


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
    parser.add_argument("--require-demos", action="store_true",
                        help="Exit if demonstrations cannot be loaded")
    parser.add_argument("--pretrain-steps", type=int, default=DEMO_CONFIG["pretrain_steps"],
                        help="Number of pretraining steps on demonstrations")
    
    # RND settings
    parser.add_argument("--no-rnd", action="store_true", 
                        help="Disable RND intrinsic rewards")
    parser.add_argument("--rnd-coef", type=float, default=RND_CONFIG["intrinsic_reward_coef"],
                        help="Intrinsic reward coefficient")
    parser.add_argument("--rnd-update-freq", type=int, default=1000,
                        help="RND model update frequency")
    
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
    
    # Memory efficiency options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output")
    parser.add_argument("--buffer-size", type=int, default=SAC_CONFIG["buffer_size"],
                      help="Size of the replay buffer")
    parser.add_argument("--memory-efficient", action="store_true",
                      help="Enable memory-efficient mode (smaller buffer, downsampled observations)")
    parser.add_argument("--obs-width", type=int, default=32,
                      help="Width of observation in memory-efficient mode")
    parser.add_argument("--obs-height", type=int, default=32,
                      help="Height of observation in memory-efficient mode")
    parser.add_argument("--force-no-rnd", action="store_true",
                      help="Disable RND when using memory-efficient mode")
    parser.add_argument("--preserve-obs-shape", action="store_true",
                      help="Preserve observation shape dimensions when using memory-efficient mode")
    
    # Early stopping and debugging
    parser.add_argument("--patience", type=int, default=5,
                       help="Number of evaluations without improvement before early stopping")
    parser.add_argument("--learning-rate", type=float, default=SAC_CONFIG["learning_rate"],
                       help="Override learning rate from config")
    
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
        from src.envs.wrappers import StableBaselinesFix
        
        # Create memory-efficient environment
        env = create_memory_efficient_env(
            env_path=env_path,
            worker_id=worker_id,
            stack_frames=config["stack_frames"],
            seed=config["seed"]
        )
        
        # Wrap the environment with GymVersionBridge
        env = StableBaselinesFix(env)
        
        return env
    
    return _init


def create_env_with_custom_wrappers(env_fn, config, use_reward_shaping=True, frame_skip=4, max_steps=1000,
                                  record=False, video_dir=None, video_length=3000):
    """Create environment with custom wrappers"""
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
    
    # No need for VecTransposeImage as observations are already in the right format
    
    return env


def main():
    """Main training function"""
    # Parse and validate arguments
    args = parse_args()
    args = validate_parameters(args)
    
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
    
    # Setup console logging if debug mode is enabled
    logger = None
    if args.debug:
        logger = ConsoleLogger(log_path, "debug_output.txt")
    
    try:
        print(f"Starting training run: {run_name}")
        print(f"Log path: {log_path}")
        print(f"Model path: {model_path}")
        
        # Create environment
        print("Creating environment...")
        env_fn = make_env(
            env_path=args.env_path,
            worker_id=args.workers,
            realtime_mode=False,
            config=ENV_CONFIG,
            no_graphics=not args.graphics
        )
        
        # Create the environment with selected wrapper approach
        use_wrappers = False
        if args.use_custom_wrappers:
            env = create_env_with_custom_wrappers(
                env_fn=env_fn,
                config=ENV_CONFIG,
                use_reward_shaping=args.reward_shaping,
                frame_skip=args.frame_skip,
                max_steps=TRAIN_CONFIG["max_episode_steps"]
            )
            use_wrappers = True
        else:
            # Use standard environment with basic preprocessing
            env = env_fn()
        
        # Apply memory-efficient wrappers if requested
        if args.memory_efficient:
            print("Applying memory-efficient wrappers...")
            env = create_memory_efficient_env(
                env, 
                width=args.obs_width, 
                height=args.obs_height, 
                debug=args.debug,
                preserve_obs_shape=args.preserve_obs_shape
            )
            use_wrappers = True
            
        # Apply Gym compatibility wrapper - only if other wrappers are used
        if use_wrappers:
            print("Applying Gym compatibility wrapper...")
            env = GymCompatibilityWrapper(env)
        
        print(f"Created environment with observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # Create evaluation environment (with different worker ID)
        eval_env_fn = make_env(
            env_path=args.env_path,
            worker_id=args.workers + 10,
            realtime_mode=False,
            config=ENV_CONFIG,
            no_graphics=not args.graphics
        )
        
        # Use same wrapper approach for eval env
        use_eval_wrappers = False
        if args.use_custom_wrappers:
            eval_env = create_env_with_custom_wrappers(
                env_fn=eval_env_fn,
                config=ENV_CONFIG,
                use_reward_shaping=args.reward_shaping,
                frame_skip=args.frame_skip,
                max_steps=TRAIN_CONFIG["max_episode_steps"]
            )
            use_eval_wrappers = True
        else:
            eval_env = eval_env_fn()
        
        # Apply same memory-efficient wrappers to eval env if needed
        if args.memory_efficient:
            eval_env = create_memory_efficient_env(
                eval_env, 
                width=args.obs_width, 
                height=args.obs_height, 
                debug=args.debug,
                preserve_obs_shape=args.preserve_obs_shape
            )
            use_eval_wrappers = True
            
        # Apply Gym compatibility wrapper to eval env - only if other wrappers are used
        if use_eval_wrappers:
            eval_env = GymCompatibilityWrapper(eval_env)
        
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
                if success and demo_buffer.total_transitions > 0:
                    print(f"Loaded {demo_buffer.total_transitions} demonstration transitions")
                else:
                    print("Warning: Failed to load demonstrations or buffer is empty")
                    if args.require_demos:
                        print("Exiting because demonstrations are required")
                        return
            else:
                print(f"Error: Demonstration file not found: {args.demo_path}")
                if args.require_demos:
                    print("Exiting because demonstrations are required")
                    return
                print("Creating new empty buffer. Note: Learning from scratch without demos")
        
        # Create RND model for intrinsic rewards
        rnd_model = None
        if not args.no_rnd and not args.force_no_rnd:
            print("Creating RND model...")
            try:
                rnd_config = RND_CONFIG.copy()
                rnd_config["intrinsic_reward_coef"] = args.rnd_coef
                
                # If using memory-efficient mode, we need to be careful with RND
                if args.memory_efficient:
                    print("WARNING: Using RND with memory-efficient mode. This may cause compatibility issues.")
                    print("If you encounter errors, use --force-no-rnd to disable RND when using memory-efficient mode.")
                    
                    # Check if we need to modify the RND input shape based on the wrapped environment
                    if len(obs_shape) != 3 or obs_shape[0] < 3:
                        print("ERROR: RND requires frame stacked observations in shape (frames, height, width)")
                        print("Current shape:", obs_shape)
                        print("Disabling RND for compatibility...")
                        args.force_no_rnd = True
                        rnd_model = None
                    else:
                        rnd_model = RNDModel(observation_shape=obs_shape, config=rnd_config)
                else:
                    rnd_model = RNDModel(observation_shape=obs_shape, config=rnd_config)
                    
            except Exception as e:
                print(f"Error creating RND model: {e}")
                traceback.print_exc()
                print("Disabling RND due to initialization error.")
                args.force_no_rnd = True
                rnd_model = None
        else:
            print("RND is disabled.")
        
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
            if not args.no_rnd and not args.force_no_rnd and rnd_model is not None:
                rnd_kwargs = {
                    "rnd_model": rnd_model,
                    "intrinsic_reward_coef": args.rnd_coef
                }
            
            # Set up policy kwargs with custom network
            policy_kwargs = setup_network_architecture(args.network, obs_shape)
            
            # Adjust buffer size if in memory-efficient mode
            if args.memory_efficient:
                print("Using memory-efficient settings...")
                # Reduce buffer size dramatically for memory efficiency
                buffer_size = min(10000, args.buffer_size) 
            else:
                buffer_size = args.buffer_size
            
            # Override learning rate if specified
            learning_rate = args.learning_rate if args.learning_rate != SAC_CONFIG["learning_rate"] else SAC_CONFIG["learning_rate"]
            if learning_rate != SAC_CONFIG["learning_rate"]:
                print(f"Overriding learning rate: {learning_rate} (config: {SAC_CONFIG['learning_rate']})")
            
            # Create model
            model = DemoSAC(
                policy=SAC_CONFIG["policy_type"],
                env=env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                learning_starts=min(buffer_size // 10, SAC_CONFIG["learning_starts"]),
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
        
        # Progress monitoring
        callbacks.append(TrainingProgressCallback(
            log_interval=TRAIN_CONFIG["log_interval"],
            verbose=1 if args.debug else 0
        ))
        
        # RND update callback
        if not args.no_rnd and not args.force_no_rnd and rnd_model is not None:
            callbacks.append(RNDUpdateCallback(
                rnd_model=rnd_model,
                update_freq=args.rnd_update_freq,
                verbose=1 if args.debug else 0
            ))
        
        # Demonstration recording callback
        if not args.no_demos and demo_buffer is not None:
            callbacks.append(DemonstrationRecorderCallback(
                demo_buffer=demo_buffer,
                min_reward_percentile=80.0,
                min_floor=1,
                verbose=1 if args.debug else 0
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
        
        # Early stopping callback
        callbacks.append(EarlyStoppingCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=EVAL_CONFIG["n_eval_episodes"],
            patience=args.patience,
            verbose=1
        ))
        
        # Checkpoint saving callback
        callbacks.append(SaveCheckpointCallback(
            save_path=model_path,
            save_freq=args.save_freq,
            save_demo_buffer=not args.no_demos,
            save_rnd_model=not args.no_rnd and not args.force_no_rnd,
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
        print(f"- Random Network Distillation: {'Disabled' if args.no_rnd or args.force_no_rnd else 'Enabled'}")
        if not args.no_rnd and not args.force_no_rnd:
            print(f"  - RND Coefficient: {args.rnd_coef}")
            print(f"  - RND Update Frequency: {args.rnd_update_freq}")
        print(f"- Training Timesteps: {args.timesteps}")
        print(f"- Buffer Size: {buffer_size}")
        print(f"- Memory-Efficient Mode: {'Enabled' if args.memory_efficient else 'Disabled'}")
        if args.memory_efficient:
            print(f"  - Observation Size: {args.obs_width}x{args.obs_height}")
            print(f"  - Preserve Shape: {'Yes' if args.preserve_obs_shape else 'No'}")
        print(f"- Learning Starts: {min(buffer_size // 10, SAC_CONFIG['learning_starts']) if args.memory_efficient else SAC_CONFIG['learning_starts']}")
        print(f"- Batch Size: {SAC_CONFIG['batch_size']}")
        print(f"- Learning Rate: {learning_rate}")
        print(f"- Gamma: {SAC_CONFIG['gamma']}")
        print(f"- Tau: {SAC_CONFIG['tau']}")
        print(f"- Early Stopping Patience: {args.patience}")
        print(f"- Debug Mode: {'Enabled' if args.debug else 'Disabled'}")
        print()
        
        # Train the agent
        print(f"Starting training for {args.timesteps} timesteps...")
        
        # In debug mode, check environment behavior first
        if args.debug:
            print("DEBUG: Testing environment reset and step...")
            try:
                obs = env.reset()
                print(f"Reset observation shape: {obs.shape}, dtype: {obs.dtype}")
                print(f"Observation min/max: {np.min(obs)}, {np.max(obs)}")
                
                action = env.action_space.sample()
                print(f"Sampled action: {action}")
                
                next_obs, reward, done, info = env.step(action)
                print(f"Step observation shape: {next_obs.shape}, dtype: {next_obs.dtype}")
                print(f"Reward: {reward}, Done: {done}")
                print(f"Info: {info}")
            except Exception as e:
                print(f"Error during environment testing: {e}")
                traceback.print_exc()
        
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                log_interval=TRAIN_CONFIG["log_interval"],
            )
            training_success = True
        except Exception as e:
            print(f"Error during training: {e}")
            print("Detailed traceback:")
            traceback.print_exc()
            
            # Write error to file for later analysis
            error_file = os.path.join(model_path, "error_log.txt")
            with open(error_file, "w") as f:
                f.write(f"Error at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error message: {str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
                f.write("\n\nConfiguration:\n")
                for key, value in vars(args).items():
                    f.write(f"{key}: {value}\n")
            
            print(f"Error details written to {error_file}")
            training_success = False
        
        # Save the final model
        if training_success:
            final_model_path = os.path.join(model_path, "final_model")
            print(f"\nTraining completed!")
            print(f"Final model saved to: {final_model_path}")
            model.save(final_model_path)
            
            # Save the final demonstration buffer
            if not args.no_demos and demo_buffer is not None:
                final_demo_path = os.path.join(model_path, "final_demonstrations.pkl")
                print(f"Final demonstration buffer saved to: {final_demo_path}")
                demo_buffer.save(final_demo_path)
            
            print("\nTraining completed successfully!")
        else:
            print("\nTraining did not complete successfully. Attempting to save checkpoint...")
            try:
                checkpoint_path = os.path.join(model_path, "interrupted_checkpoint")
                model.save(checkpoint_path)
                print(f"Checkpoint saved to: {checkpoint_path}")
                
                # Save demo buffer if available
                if not args.no_demos and demo_buffer is not None:
                    demo_path = os.path.join(model_path, "interrupted_demonstrations.pkl")
                    demo_buffer.save(demo_path)
                    print(f"Demonstration buffer saved to: {demo_path}")
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")
    
    finally:
        # Close the logger if it was created
        if logger is not None:
            logger.close()
        
        # Always close environments properly
        try:
            if 'env' in locals() and env is not None:
                env.close()
            if 'eval_env' in locals() and eval_env is not None:
                eval_env.close()
        except Exception as e:
            print(f"Error closing environments: {e}")


if __name__ == "__main__":
    main()