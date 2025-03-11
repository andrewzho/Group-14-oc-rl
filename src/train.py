import gym
from obstacle_tower_env import ObstacleTowerEnv
from src.model import PPONetwork
from src.ppo import PPO
from src.utils import normalize, save_checkpoint, load_checkpoint, ActionFlattener, MetricsTracker, TrainingLogger
import torch
import numpy as np
import argparse
from collections import deque
import os
from mlagents_envs.exception import UnityCommunicatorStoppedException
import time
import logging
from src.create_env import create_obstacle_tower_env
import torch.optim as optim
from src.detection import detect_key_visually, detect_door_visually
from src.memory import EnhancedKeyDoorMemory
import traceback
import datetime
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Add a simple episodic memory mechanism
class EpisodicMemory:
    def __init__(self, capacity=1000):
        self.key_locations = {}  # Floor -> list of estimated positions where keys were found
        self.door_locations = {}  # Floor -> list of estimated positions where doors were used
        self.successful_floors = set()  # Set of floors successfully completed
        self.capacity = capacity
        
    def add_key_location(self, floor, position):
        if floor not in self.key_locations:
            self.key_locations[floor] = []
        self.key_locations[floor].append(position)
        # Limit size
        if len(self.key_locations[floor]) > self.capacity:
            self.key_locations[floor] = self.key_locations[floor][-self.capacity:]
            
    def add_door_location(self, floor, position):
        if floor not in self.door_locations:
            self.door_locations[floor] = []
        self.door_locations[floor].append(position)
        # Limit size
        if len(self.door_locations[floor]) > self.capacity:
            self.door_locations[floor] = self.door_locations[floor][-self.capacity:]
            
    def mark_floor_complete(self, floor):
        self.successful_floors.add(floor)
        
    def is_key_location_nearby(self, floor, position, threshold=3.0):
        if floor not in self.key_locations:
            return False
        for key_pos in self.key_locations[floor]:
            dist = sum((position[i] - key_pos[i])**2 for i in range(len(position)))**0.5
            if dist < threshold:
                return True
        return False
        
    def is_door_location_nearby(self, floor, position, threshold=3.0):
        if floor not in self.door_locations:
            return False
        for door_pos in self.door_locations[floor]:
            dist = sum((position[i] - door_pos[i])**2 for i in range(len(position)))**0.5
            if dist < threshold:
                return True
        return False

# Set Unity to run in headless mode
os.environ['DISPLAY'] = ''

# Add at the top of your train.py file
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('obstacle_tower')

def make_env(env_path, seed=None, realtime_mode=False, rank=0, no_graphics=True, curriculum_config=None):
    """Factory function to create environments with proper seeding for parallel execution."""
    def _init():
        # Create a unique worker ID for each environment to avoid port conflicts
        worker_id = rank + int(time.time() * 1000) % 10000
        
        env = create_obstacle_tower_env(
            executable_path=env_path,
            realtime_mode=realtime_mode,
            timeout=300,
            worker_id=worker_id,
            no_graphics=no_graphics,
            config=curriculum_config
        )
        
        if seed is not None:
            env.seed(seed + rank)
            
        # Apply curriculum settings if provided
        if curriculum_config:
            try:
                for key, value in curriculum_config.items():
                    env.reset_parameters.set_float_parameter(key, float(value))
            except Exception as e:
                print(f"Error applying curriculum settings in env {rank}: {e}")
                traceback.print_exc()
                
        return env
    return _init

def main(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Check if we're just running evaluation mode
    if args.evaluate_only and args.checkpoint:
        try:
            print("Running in evaluation-only mode")
            from src.evaluate import evaluate_main
            
            # Create namespace with all necessary arguments for evaluation
            eval_args = argparse.Namespace(
                env_path=args.env_path,
                seed=args.seed,
                floor=args.starting_floor,
                checkpoint=args.checkpoint,
                device=args.device,
                deterministic=args.deterministic if hasattr(args, 'deterministic') else False,
                episodes=args.eval_episodes,
                max_ep_steps=args.max_ep_steps if hasattr(args, 'max_ep_steps') else 1000,
                render=args.render if hasattr(args, 'render') else False,
                realtime_mode=args.realtime_mode if hasattr(args, 'realtime_mode') else False,
                video_path=args.video_path if hasattr(args, 'video_path') else None
            )
            
            # Print evaluation settings
            print("Evaluation settings:")
            print(f"  Render: {eval_args.render}")
            print(f"  Realtime mode: {eval_args.realtime_mode}")
            
            # Run evaluation
            evaluate_main(eval_args)
            return
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return
        
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create environment
    print("INFO:obstacle_tower:Setting up environment...")
    env = create_obstacle_tower_env(
        executable_path=args.env_path,
        realtime_mode=args.realtime_mode,
        no_graphics=not args.render
    )
    print("INFO:obstacle_tower:Environment created")
    
    # Setup curriculum learning if enabled
    if args.curriculum:
        curriculum_config = {
            "starting-floor": 0,
            "tower-seed": args.seed
        }
        curriculum_attempts = 0
        curriculum_successes = 0
        curriculum_success_streak = 0
        current_curriculum_floor = 0
        curriculum_success_threshold = 5  # Increased from 3 - require more successes
        curriculum_attempt_threshold = 15  # Increased from 10 - give more attempts
        curriculum_floor_success_history = {}  # Track success rate per floor
    else:
        curriculum_config = None

    # Initialize training logger early, before using it
    training_logger = TrainingLogger(args.log_dir, model=None, log_frequency=10)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(args.log_dir)
    
    # Initialize episodic memory
    episodic_memory = EpisodicMemory()
    
    # Initialize enhanced memory
    key_door_memory = EnhancedKeyDoorMemory()
    
    # Environment setup
    logger.info("Setting up environment...")
    # Set initial curriculum if enabled
    if args.curriculum:
        training_logger.log(f"Starting curriculum learning at floor {current_curriculum_floor}", "CURRICULUM")
    
    # Create environment without passing config directly
    env = create_obstacle_tower_env(realtime_mode=False)
    
    # Apply curriculum settings after creation if enabled
    if args.curriculum and curriculum_config:
        try:
            for key, value in curriculum_config.items():
                env.reset_parameters.set_float_parameter(key, float(value))
            training_logger.log(f"Applied curriculum settings: {curriculum_config}", "ENV")
        except Exception as e:
            error_msg = f"Error applying curriculum settings: {e}"
            print(error_msg)
            training_logger.log(error_msg, "ERROR")
            
    logger.info("Environment created")
    action_flattener = ActionFlattener(env.action_space.nvec)
    num_actions = action_flattener.action_space.n
    print(f"Number of actions: {num_actions}")

    # Model and PPO setup
    model = PPONetwork(input_shape=(12, 84, 84), num_actions=num_actions).to(device)
    
    # Update training logger with the model
    training_logger.update_model(model)
    
    # Log hyperparameters
    hyperparams = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_eps': args.clip_eps,
        'entropy_reg': args.entropy_reg,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'num_steps': args.num_steps,
        'device': str(device),
        'action_space': num_actions,
        'input_shape': (12, 84, 84)
    }
    training_logger.log_hyperparameters(hyperparams)
    
    # Improved PPO parameters
    ppo = PPO(
        model, 
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.epochs,
        batch_size=args.batch_size,
        vf_coef=0.5,
        ent_reg=args.entropy_reg,
        max_grad_norm=0.7,
        target_kl=0.02,
        lr_scheduler='cosine'
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        training_logger.log(f"Loading checkpoint from {args.checkpoint}", "CHECKPOINT")
        model, loaded_metrics, loaded_update_count = load_checkpoint(model, args.checkpoint, ppo.optimizer, ppo.scheduler)
        if loaded_metrics:
            # Restore metrics
            metrics_tracker.metrics = loaded_metrics
            
        if loaded_update_count is not None:
            update_count = loaded_update_count
            training_logger.log(f"Restored update_count to {update_count}", "CHECKPOINT")
        else:
            # Fallback: estimate update count based on checkpoint filename
            checkpoint_name = os.path.basename(args.checkpoint)
            if checkpoint_name.startswith("step_") and checkpoint_name.endswith(".pth"):
                try:
                    steps_done = int(checkpoint_name[5:-4])
                    update_count = steps_done // 4096  # Estimate update count based on steps
                    training_logger.log(f"Estimated update_count as {update_count} based on steps_done {steps_done}", "CHECKPOINT")
                except ValueError:
                    pass

    # Initialize experience replay buffer if enabled
    replay_buffer = {
        'states': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'dones': []
    }
    max_replay_size = args.replay_size
    
    # Function to add batch to replay buffer
    def add_to_replay(replay_buffer, states, actions, rewards, log_probs, values, dones):
        replay_buffer['states'].extend(states)
        replay_buffer['actions'].extend(actions)
        replay_buffer['rewards'].extend(rewards)
        replay_buffer['log_probs'].extend(log_probs)
        replay_buffer['values'].extend(values)
        replay_buffer['dones'].extend(dones)
        
        # Keep buffer size limited
        if len(replay_buffer['states']) > max_replay_size:
            excess = len(replay_buffer['states']) - max_replay_size
            for key in replay_buffer.keys():
                replay_buffer[key] = replay_buffer[key][excess:]
    
    # Function to sample from replay buffer and compute returns and advantages
    def sample_from_replay(replay_buffer, batch_size):
        # Ensure we have enough samples
        buffer_size = len(replay_buffer['states'])
        if buffer_size == 0:
            return [], [], [], [], []  # Return empty lists if buffer is empty
            
        # Create priority weights based on rewards, floors, and recency
        priorities = np.ones(buffer_size)
        
        # 1. Prioritize based on rewards - higher rewards get higher priority
        abs_rewards = np.abs(np.array(replay_buffer['rewards']))
        if abs_rewards.max() > 0:
            reward_priorities = abs_rewards / abs_rewards.max()
            priorities *= (1.0 + reward_priorities)
        
        # 2. Prioritize recent experiences - more recent samples get higher priority
        recency_factor = 2.0
        recency_priorities = np.linspace(1.0, recency_factor, buffer_size)
        priorities *= recency_priorities
        
        # 3. Try to extract floor information and prioritize higher floors
        # This is a simplified approximation - adapt to your state representation
        try:
            # Extract floor info if available
            floor_info = []
            for i in range(buffer_size):
                # For this example, we're using a simple heuristic
                # In practice, you should extract actual floor info if available
                floor = 0  # Default floor
                
                # Estimate floor from reward magnitude as a heuristic
                if replay_buffer['rewards'][i] > 5.0:
                    floor = min(5, int(replay_buffer['rewards'][i] / 5.0))
                
                floor_info.append(floor)
            
            if len(floor_info) > 0 and max(floor_info) > 0:
                floor_priorities = np.array(floor_info) / max(1, max(floor_info))
                # Higher floors get much higher priority
                floor_boost = 1.0 + 2.0 * floor_priorities
                priorities *= floor_boost
        except Exception as e:
            print(f"Error calculating floor priorities: {e}")
        
        # Normalize priorities to create a valid probability distribution
        priorities = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            buffer_size, 
            min(buffer_size, batch_size), 
            replace=False,
            p=priorities
        )
        
        # Extract batch
        batch_states = [replay_buffer['states'][i] for i in indices]
        batch_actions = [replay_buffer['actions'][i] for i in indices]
        batch_rewards = [replay_buffer['rewards'][i] for i in indices]
        batch_log_probs = [replay_buffer['log_probs'][i] for i in indices]
        batch_values = [replay_buffer['values'][i] for i in indices]
        batch_dones = [replay_buffer['dones'][i] for i in indices]
        
        # Compute returns and advantages
        returns = []
        advantages = []
        
        # Get the last state's value or use 0 if not available
        if len(batch_values) > 0 and not batch_dones[-1]:
            # Here we'd need the value of the next state, but it's not stored
            # Instead, we'll use 0 which assumes episode boundaries
            next_value = 0.0
        else:
            next_value = 0.0
            
        # Compute returns using GAE
        gae_advantages = ppo.compute_gae(
            batch_rewards, batch_values, next_value, batch_dones
        )
        
        # Calculate returns as advantage + value
        for adv, val in zip(gae_advantages, batch_values):
            returns.append(adv + val)
            
        # Return exactly 5 values: states, actions, returns, advantages, old_log_probs
        return batch_states, batch_actions, returns, gae_advantages, batch_log_probs
        
    # Evaluation function to assess policy without exploration
    def evaluate_policy(n_episodes=5):
        eval_rewards = []
        eval_floors = []
        eval_lengths = []
        
        with torch.no_grad():
            for _ in range(n_episodes):
                frame_stack = deque(maxlen=4)
                eval_done = False
                eval_reward = 0
                eval_length = 0
                eval_floor = 0
                
                # Reset environment
                eval_obs = env.reset()
                eval_obs = np.transpose(eval_obs[0], (2, 0, 1)) / 255.0
                for _ in range(4):
                    frame_stack.append(eval_obs)
                eval_state = np.concatenate(frame_stack, axis=0)
                eval_obs = torch.tensor(eval_state, dtype=torch.float32).to(device)
                
                while not eval_done:
                    # Select action with less randomness
                    obs_batched = eval_obs.unsqueeze(0)
                    policy_logits, _ = model(obs_batched)
                    
                    # Use more deterministic policy for evaluation
                    if np.random.random() < 0.9:  # 90% greedy actions
                        action_idx = torch.argmax(policy_logits, dim=1).item()
                    else:
                        dist = torch.distributions.Categorical(logits=policy_logits)
                        action_idx = dist.sample().item()
                        
                    action = action_flattener.lookup_action(action_idx)
                    action = np.array(action)
                    
                    # Take step in environment
                    next_obs, reward, eval_done, info = env.step(action)
                    
                    # Update evaluation metrics
                    eval_reward += reward
                    eval_length += 1
                    if info["current_floor"] > eval_floor:
                        eval_floor = info["current_floor"]
                        
                    # Process next observation
                    next_obs = np.transpose(next_obs[0], (2, 0, 1)) / 255.0
                    frame_stack.append(next_obs)
                    next_state = np.concatenate(frame_stack, axis=0)
                    eval_obs = torch.tensor(next_state, dtype=torch.float32).to(device)
                    
                # Record episode results
                eval_rewards.append(eval_reward)
                eval_floors.append(eval_floor)
                eval_lengths.append(eval_length)
                
        # Log evaluation results
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        avg_floor = sum(eval_floors) / len(eval_floors)
        avg_length = sum(eval_lengths) / len(eval_lengths)
        max_floor = max(eval_floors)
        
        evaluation_msg = (f"Evaluation over {n_episodes} episodes - "
                         f"Avg Reward: {avg_reward:.2f}, Avg Floor: {avg_floor:.2f}, "
                         f"Max Floor: {max_floor}, Avg Length: {avg_length:.2f}")
        print(evaluation_msg)
        training_logger.log(evaluation_msg, "EVAL")
        
        # Update metrics
        metrics_tracker.update('eval_avg_reward', avg_reward)
        metrics_tracker.update('eval_avg_floor', avg_floor)
        metrics_tracker.update('eval_max_floor', max_floor)
        
        # Update learning rate scheduler if using reward-based plateau scheduler
        if hasattr(ppo.scheduler, 'step') and isinstance(ppo.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            ppo.update_lr_scheduler(avg_reward)
            
        return avg_reward, max_floor

    # Initialize variables for training loop
    total_frames = 0
    steps_done = 0
    update_count = 0  # Add update counter initialization
    episode = 0
    
    # Training loop
    max_steps = args.num_steps
    steps_done = 0
    episode_count = 0
    current_floor = 0
    max_floor_reached = 0
    
    start_time = time.time()
    last_save_time = start_time
    last_log_time = start_time
    
    key_collections = 0
    door_openings = 0
    
    training_logger.log("Starting training loop", "INFO")
    
    last_eval_time = 0
    
    # Initialize tracking for each environment
    env_episodes = [0] * args.n_envs
    env_rewards = [0] * args.n_envs
    env_steps = [0] * args.n_envs
    env_floors = [0] * args.n_envs
    env_dones = [False] * args.n_envs
    
    # Initialize frame stacks for each environment
    frame_stacks = [deque(maxlen=4) for _ in range(args.n_envs)]
    
    # Initialize previous keys and positions for each environment
    env_previous_keys = [None] * args.n_envs
    env_previous_positions = [None] * args.n_envs
    env_previous_obs = [None] * args.n_envs
    env_last_key_positions = [None] * args.n_envs
    
    # Reset all environments
    logger.info(f"Resetting environments...")
    try:
        obs = env.reset()
        logger.info(f"Environment reset complete. Got {len(obs)} observations.")
    except UnityCommunicatorStoppedException as e:
        error_msg = f"Error during initial reset: {e}"
        print(error_msg)
        training_logger.log(error_msg, "ERROR")
        env.close()
        return
    
    # Initialize frame stacks with initial observations
    for i in range(args.n_envs):
        env_obs = obs[i]
        if isinstance(env_obs, tuple):
            img_obs = env_obs[0]
        else:
            img_obs = env_obs
            
        img_obs = np.transpose(img_obs, (2, 0, 1)) / 255.0
        
        for _ in range(4):
            frame_stacks[i].append(img_obs)
    
    # Create initial states for all environments
    states = []
    for i in range(args.n_envs):
        state = np.concatenate(list(frame_stacks[i]), axis=0)
        states.append(state)
    
    while steps_done < max_steps:
        # Collect trajectories across all environments
        trajectory_states = [[] for _ in range(args.n_envs)]
        trajectory_actions = [[] for _ in range(args.n_envs)]
        trajectory_rewards = [[] for _ in range(args.n_envs)]
        trajectory_log_probs = [[] for _ in range(args.n_envs)]
        trajectory_values = [[] for _ in range(args.n_envs)]
        trajectory_dones = [[] for _ in range(args.n_envs)]
        
        # Track reward components for detailed logging
        reward_components = {
            'base': 0,
            'forward_movement': 0,
            'stay_still_penalty': 0,
            'rotation_penalty': 0,
            'jump_penalty': 0,
            'time_penalty': 0,
            'floor_bonus': 0,
            'key_bonus': 0,
            'door_bonus': 0,
            'exploration_bonus': 0
        }
        
        # Collect trajectory
        steps_in_trajectory = 0
        trajectory_length = 1024  # Shorter trajectory length for more frequent updates
        max_episode_steps = 4000  # Set a maximum number of steps per episode to prevent getting stuck
        
        for step in range(trajectory_length):
            # Process state input for each environment
            model_states = []
            for i in range(args.n_envs):
                model_input = torch.tensor(states[i], dtype=torch.float32).to(device)
                model_states.append(model_input)
                
            # Batch all states for the model
            if len(model_states) > 0:
                batched_states = torch.stack(model_states).to(device)
            else:
                break  # No active environments
                
            # Get actions for all environments
            with torch.no_grad():
                policy_logits, values = model(batched_states)
                dists = [torch.distributions.Categorical(logits=policy_logits[i]) for i in range(len(policy_logits))]
                action_indices = [dist.sample().item() for dist in dists]
                log_probs = [dist.log_prob(torch.tensor([idx], device=device)).item() for dist, idx in zip(dists, action_indices)]
                value_list = values.cpu().numpy().tolist()
                
            # Convert action indices to actual actions
            action_arrays = []
            for idx in action_indices:
                action = action_flattener.lookup_action(idx)
                action_arrays.append(action)
                
            # Step all environments
            next_obs, rewards, dones, infos = env.step(action_arrays)
            
            # Process each environment's result
            next_states = []
            for i in range(args.n_envs):
                # Process observation for frame stack
                if isinstance(next_obs[i], tuple):
                    img_obs = next_obs[i][0]
                else:
                    img_obs = next_obs[i]
                    
                img_obs = np.transpose(img_obs, (2, 0, 1)) / 255.0
                frame_stacks[i].append(img_obs)
                next_state = np.concatenate(list(frame_stacks[i]), axis=0)
                next_states.append(next_state)
                
                # Enhanced reward shaping
                move_idx, rot_idx, jump_idx, _ = action_arrays[i]
                shaped_reward = rewards[i]
                reward_components['base'] += rewards[i]
                
                # Track key usage and door opening
                current_keys = infos[i].get("total_keys", 0)
                if env_previous_keys[i] is not None:
                    # If keys decreased without collecting new ones, a door was opened
                    if current_keys < env_previous_keys[i]:
                        door_bonus = 0.5  # Increased reward for door opening
                        shaped_reward += door_bonus
                        door_msg = f"Env {i} - Door opened! Reward bonus added: +{door_bonus}"
                        print(door_msg)
                        training_logger.log_significant_event("DOOR", door_msg)
                        door_openings += 1
                        metrics_tracker.update('door_openings', door_openings)
                        reward_components['door_bonus'] += door_bonus
                        
                        # Record door location in episodic memory
                        episodic_memory.add_door_location(
                            infos[i].get("current_floor", 0), 
                            (infos[i].get("x_pos", 0), infos[i].get("y_pos", 0), infos[i].get("z_pos", 0))
                        )
                        
                    # If keys increased, a key was collected
                    elif current_keys > env_previous_keys[i]:
                        key_bonus = 0.5  # Increased reward for key collection
                        shaped_reward += key_bonus
                        key_msg = f"Env {i} - Key collected! Reward bonus added: +{key_bonus}"
                        print(key_msg)
                        training_logger.log_significant_event("KEY", key_msg)
                        key_collections += 1
                        metrics_tracker.update('key_collections', key_collections)
                        reward_components['key_bonus'] += key_bonus
                        
                        # Record key location in episodic memory
                        episodic_memory.add_key_location(
                            infos[i].get("current_floor", 0), 
                            (infos[i].get("x_pos", 0), infos[i].get("y_pos", 0), infos[i].get("z_pos", 0))
                        )
                env_previous_keys[i] = current_keys
                
                # Apply the rest of the reward shaping (movement, exploration, etc.)
                # This is just like your original code, but track for each environment
                
                # ... (rest of reward shaping logic) ...
                
                # Store trajectory data
                trajectory_states[i].append(states[i])
                trajectory_actions[i].append(action_indices[i])
                trajectory_rewards[i].append(shaped_reward)
                trajectory_log_probs[i].append(log_probs[i])
                trajectory_values[i].append(value_list[i])
                trajectory_dones[i].append(dones[i])
                
                # Update environment tracking
                env_rewards[i] += rewards[i]  # Track original rewards for metrics
                env_steps[i] += 1
                
                # Track floor level
                current_floor_i = infos[i].get("current_floor", 0)
                if current_floor_i > env_floors[i]:
                    env_floors[i] = current_floor_i
                    if current_floor_i > max_floor_reached:
                        max_floor_reached = current_floor_i
                    floor_msg = f"Env {i} - New floor reached: {current_floor_i}"
                    print(floor_msg)
                    training_logger.log_significant_event("FLOOR", floor_msg)
                    
                    # Add direct floor checkpoint saving here
                    try:
                        floor_checkpoint_path = os.path.join(args.log_dir, f"floor_{current_floor_i}.pth")
                        if not os.path.exists(floor_checkpoint_path):
                            save_checkpoint(
                                model, 
                                floor_checkpoint_path,
                                optimizer=ppo.optimizer,
                                scheduler=ppo.scheduler,
                                metrics=None,  # Don't save metrics to keep file smaller
                                update_count=update_count
                            )
                            print(f"CHECKPOINT: Saved floor checkpoint to {floor_checkpoint_path}")
                    except Exception as e:
                        print(f"ERROR saving floor checkpoint: {e}")
                        traceback.print_exc()
                
                # Handle episode completion
                if dones[i]:
                    episode_count += 1
                    
                    # Log episode statistics
                    metrics_tracker.update('episode_rewards', env_rewards[i])
                    metrics_tracker.update('episode_lengths', env_steps[i])
                    metrics_tracker.update('episode_floors', env_floors[i])
                    
                    print(f"Env {i} - Episode {env_episodes[i]+1} - Reward: {env_rewards[i]:.2f}, "
                          f"Length: {env_steps[i]}, Floor: {env_floors[i]}, Steps: {steps_done}")
                    
                    # Reset tracking for this environment
                    env_episodes[i] += 1
                    env_rewards[i] = 0
                    env_steps[i] = 0
                
            # Update states
            states = next_states
            
            # Increment global step counter
            steps_done += args.n_envs
            steps_in_trajectory += 1
            
            # Check if we should terminate the trajectory collection
            if steps_in_trajectory >= trajectory_length:
                break
                
        # After collecting trajectories from all environments, flatten and prepare for PPO update
        all_states = []
        all_actions = []
        all_rewards = []
        all_log_probs = []
        all_values = []
        all_dones = []
        
        # Flatten trajectories from all environments
        for i in range(args.n_envs):
            all_states.extend(trajectory_states[i])
            all_actions.extend(trajectory_actions[i])
            all_rewards.extend(trajectory_rewards[i])
            all_log_probs.extend(trajectory_log_probs[i])
            all_values.extend(trajectory_values[i])
            all_dones.extend(trajectory_dones[i])
            
        # Make sure we have data to update
        if len(all_states) > 0:
            # Calculate returns and advantages
            returns = []
            advantages = []
            
            # Process each environment separately for proper GAE calculation
            for i in range(args.n_envs):
                if len(trajectory_rewards[i]) > 0:
                    # Get final value for this environment
                    if not env_dones[i]:
                        # Get value of final state
                        final_state = states[i]
                        final_state_tensor = torch.tensor(final_state, dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            _, next_value = model(final_state_tensor)
                        next_value = next_value.item()
                    else:
                        next_value = 0.0
                        
                    # Compute GAE for this environment
                    env_gae = ppo.compute_gae(
                        rewards=trajectory_rewards[i],
                        values=trajectory_values[i],
                        next_value=next_value,
                        dones=trajectory_dones[i]
                    )
                    
                    # Compute returns (advantage + value)
                    env_returns = []
                    for j in range(len(env_gae)):
                        env_returns.append(env_gae[j] + trajectory_values[i][j])
                        
                    # Add to overall returns and advantages
                    returns.extend(env_returns)
                    advantages.extend(env_gae)
            
            # Update the policy with PPO
            if len(returns) > 0:
                # Convert to tensors
                states_tensor = torch.tensor(np.array(all_states), dtype=torch.float32).to(device)
                actions_tensor = torch.tensor(all_actions, dtype=torch.int64).to(device)
                returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
                old_log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float32).to(device)
                
                # Update the policy
                metrics = ppo.update(
                    states=states_tensor,
                    actions=actions_tensor,
                    old_log_probs=old_log_probs_tensor,
                    returns=returns_tensor,
                    advantages=advantages_tensor
                )
                
                update_count += 1
                
                # Log metrics
                update_summary = (
                    f"Update {update_count}: " +
                    f"Policy Loss: {metrics['policy_loss']:.4f}, " +
                    f"Value Loss: {metrics['value_loss']:.4f}, " +
                    f"Entropy: {metrics['entropy']:.4f}, " +
                    f"KL: {metrics['approx_kl']:.4f}, " +
                    f"LR: {metrics['learning_rate']:.6f}"
                )
                print(update_summary)
                training_logger.log_update(metrics)
                
                # Add to experience replay if enabled
                if args.experience_replay:
                    add_to_replay(replay_buffer, all_states, all_actions, all_rewards, all_log_probs, all_values, all_dones)

        # Save metrics and checkpoints periodically
        current_time = time.time()
        
        # Log training summary every minute
        if current_time - last_log_time > 60:  # Every minute
            elapsed_time = current_time - start_time
            steps_per_sec = steps_done / max(1, elapsed_time)
            metrics_tracker.update('steps_per_second', steps_per_sec)
            
            training_logger.log_training_summary(
                metrics_tracker=metrics_tracker,
                elapsed_time=elapsed_time,
                steps_done=steps_done
            )
            
            # Calculate and log environment stats across all envs
            active_envs = sum(1 for i in range(args.n_envs) if not env_dones[i])
            parallelism_efficiency = active_envs / args.n_envs if args.n_envs > 0 else 1.0
            
            training_logger.log(
                f"Parallel training stats: {active_envs}/{args.n_envs} active envs, " +
                f"efficiency: {parallelism_efficiency:.2f}, steps/sec: {steps_per_sec:.1f}",
                "PARALLELISM"
            )
            
            last_log_time = current_time
        
        # Run evaluation periodically using a separate environment
        if steps_done - last_eval_time >= args.eval_interval:
            # Create a separate environment for evaluation
            eval_env = create_obstacle_tower_env(
                executable_path=args.env_path,
                realtime_mode=args.realtime_mode,
                no_graphics=not args.render
            )
            
            eval_reward, eval_max_floor = evaluate_policy(eval_env)
            eval_env.close()  # Close evaluation environment
            
            last_eval_time = steps_done
            
            # Save a checkpoint after evaluation
            checkpoint_path = os.path.join(args.log_dir, f"eval_step_{steps_done}.pth")
            save_checkpoint(
                model, 
                checkpoint_path, 
                optimizer=ppo.optimizer, 
                scheduler=ppo.scheduler, 
                metrics=metrics_tracker.metrics
            )
            
            training_logger.log(f"Saved post-evaluation checkpoint at step {steps_done}", "CHECKPOINT")
        
        # Save checkpoints and metrics every 5 minutes
        if current_time - last_save_time > 300:  # Every 5 minutes
            # Update metrics tracker with PPO metrics
            metrics_tracker.update_from_ppo(ppo)
            
            # Save metrics without plotting
            metrics_tracker.save(plot=False)
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.log_dir, f"step_{steps_done}.pth")
            save_checkpoint(
                model, 
                checkpoint_path, 
                optimizer=ppo.optimizer, 
                scheduler=ppo.scheduler, 
                metrics=metrics_tracker.metrics,
                update_count=update_count
            )
            
            training_logger.log(f"Saved checkpoint at step {steps_done}", "CHECKPOINT")
            last_save_time = current_time

    # Final save
    metrics_tracker.update_from_ppo(ppo)
    metrics_tracker.save(plot=False)  # Don't generate plots
    
    final_checkpoint_path = os.path.join(args.log_dir, "final_model.pth")
    save_checkpoint(
        model, 
        final_checkpoint_path,
        optimizer=ppo.optimizer,
        scheduler=ppo.scheduler,
        metrics=metrics_tracker.metrics,
        update_count=update_count
    )
    
    # Log final training summary
    training_logger.log("=== FINAL TRAINING SUMMARY ===", "SUMMARY")
    training_logger.log(f"Total episodes: {episode_count}", "SUMMARY")
    training_logger.log(f"Total steps: {steps_done}", "SUMMARY")
    training_logger.log(f"Max floor reached: {max_floor_reached}", "SUMMARY")
    training_logger.log(f"Total training time: {time.time() - start_time:.2f} seconds", "SUMMARY")
    training_logger.log(f"Average steps per second: {steps_done / (time.time() - start_time):.2f}", "SUMMARY")
    training_logger.log(f"Environments: {args.n_envs}", "SUMMARY")
    
    # Close logger
    training_logger.close()
    
    # Close environment (will close all sub-environments)
    env.close()

if __name__ == "__main__":
    import argparse
    import datetime
    import traceback  # Add traceback import
    
    parser = argparse.ArgumentParser(description='Train PPO agent on Obstacle Tower')
    
    # Environment settings
    parser.add_argument('--env_path', type=str, default="./ObstacleTower/obstacletower.x86_64",
                        help='Path to Obstacle Tower executable')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_steps', type=int, default=1000000,
                        help='Number of training steps')
    parser.add_argument('--n_envs', type=int, default=1,
                        help='Number of parallel environments')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of PPO epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='PPO batch size')
    parser.add_argument('--entropy_reg', type=float, default=0.02,
                        help='Entropy regularization coefficient')
    
    # Replay and evaluation settings
    parser.add_argument('--experience_replay', action='store_true',
                        help='Use experience replay')
    parser.add_argument('--replay_size', type=int, default=10000,
                        help='Maximum size of replay buffer')
    parser.add_argument('--eval_interval', type=int, default=50000,
                        help='Number of steps between evaluations')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    
    # Training settings
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda or cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs and checkpoints')
    
    # Visualization settings
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--realtime_mode', action='store_true',
                        help='Run the environment in realtime mode')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to save video of agent evaluation')
    
    # Evaluation-only mode
    parser.add_argument('--evaluate_only', action='store_true',
                        help='Only run evaluation, no training (requires --checkpoint)')
    parser.add_argument('--starting_floor', type=int, default=None,
                        help='Starting floor for evaluation (only used with --evaluate_only)')
    
    args = parser.parse_args()
    
    # Create a timestamped log directory if not specified
    if args.log_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = f"./logs/run_{timestamp}"
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run main training loop
    main(args)