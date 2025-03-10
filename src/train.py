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

def main(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Check if we're just running evaluation mode
    if args.evaluate_only and args.checkpoint:
        print(f"Running in evaluation-only mode using checkpoint: {args.checkpoint}")
        try:
            # Import and run the evaluation code
            from src.evaluate import main as evaluate_main
            import argparse
            
            # Create a proper eval_args with all required fields
            eval_args = argparse.Namespace(
                env_path=args.env_path,
                seed=args.seed,
                floor=args.starting_floor,  # Map to the equivalent evaluate.py argument
                checkpoint=args.checkpoint,
                device=args.device,
                deterministic=0.9,  # Default for evaluation
                episodes=args.eval_episodes,
                max_ep_steps=2000,  # Reasonable default
                render=True,  # Always render in evaluation mode
                video_path=args.video_path
            )
            
            # Call the evaluation function
            evaluate_main(eval_args)
        except Exception as e:
            print(f"Error during evaluation: {e}")
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
    
    while steps_done < max_steps:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0
        episode_length = 0
        frame_stack = deque(maxlen=4)

        # Reset environment with current curriculum settings if enabled
        if args.curriculum and curriculum_config:
            try:
                # Update curriculum difficulty based on performance
                # Only update after an episode has completed (not at the very start)
                if 'done' in locals() and done:  # Check if 'done' exists and is True
                    curriculum_attempts += 1
                    
                    # Track floor-specific success rate
                    floor_reached = info["current_floor"]
                    if floor_reached not in curriculum_floor_success_history:
                        curriculum_floor_success_history[floor_reached] = {
                            'attempts': 0,
                            'successes': 0
                        }
                    curriculum_floor_success_history[floor_reached]['attempts'] += 1
                    
                    # Check if agent reached or exceeded target floor
                    if floor_reached >= current_curriculum_floor:
                        curriculum_successes += 1
                        curriculum_success_streak += 1
                        
                        # Record as success for this floor
                        curriculum_floor_success_history[floor_reached]['successes'] += 1
                        
                        success_rate = curriculum_floor_success_history[floor_reached]['successes'] / \
                                       curriculum_floor_success_history[floor_reached]['attempts']
                        
                        training_logger.log(
                            f"Curriculum success! {curriculum_successes}/{curriculum_success_threshold} " + 
                            f"successes at floor {current_curriculum_floor}. " +
                            f"Success streak: {curriculum_success_streak}. " +
                            f"Success rate: {success_rate:.2f}",
                            "CURRICULUM"
                        )
                        
                        # If agent has a strong success streak, increase difficulty faster
                        if curriculum_success_streak >= curriculum_success_threshold:
                            current_curriculum_floor += 1
                            curriculum_successes = 0
                            curriculum_attempts = 0
                            curriculum_success_streak = 0
                            curriculum_config["starting-floor"] = current_curriculum_floor
                            training_logger.log(
                                f"Curriculum difficulty increased to floor {current_curriculum_floor} " +
                                f"after {curriculum_success_threshold} consecutive successes!", 
                                "CURRICULUM"
                            )
                        # Otherwise, if accumulated enough total successes
                        elif curriculum_successes >= curriculum_success_threshold * 2:
                            current_curriculum_floor += 1
                            curriculum_successes = 0
                            curriculum_attempts = 0
                            curriculum_success_streak = 0
                            curriculum_config["starting-floor"] = current_curriculum_floor
                            training_logger.log(
                                f"Curriculum difficulty increased to floor {current_curriculum_floor} " +
                                f"after {curriculum_success_threshold * 2} total successes.", 
                                "CURRICULUM"
                            )
                    else:
                        # Reset streak if failed to reach target floor
                        curriculum_success_streak = 0
                        training_logger.log(
                            f"Curriculum failure. Only reached floor {floor_reached}, " +
                            f"target was {current_curriculum_floor}. Streak reset.",
                            "CURRICULUM"
                        )
                    
                    # If agent is struggling, decrease difficulty
                    if curriculum_attempts >= curriculum_attempt_threshold and current_curriculum_floor > 0:
                        # Calculate success rate for current floor
                        curr_floor_stats = curriculum_floor_success_history.get(current_curriculum_floor, 
                                                                              {'attempts': 0, 'successes': 0})
                        success_rate = curr_floor_stats['successes'] / max(1, curr_floor_stats['attempts'])
                        
                        # If success rate is low, decrease difficulty
                        if success_rate < 0.3:
                            current_curriculum_floor = max(0, current_curriculum_floor - 1)
                            curriculum_successes = 0
                            curriculum_attempts = 0
                            curriculum_success_streak = 0
                            curriculum_config["starting-floor"] = current_curriculum_floor
                            training_logger.log(
                                f"Curriculum difficulty decreased to floor {current_curriculum_floor} " +
                                f"due to low success rate ({success_rate:.2f}).", 
                                "CURRICULUM"
                            )
                        else:
                            # Keep current floor but reset counters
                            curriculum_attempts = 0
                            training_logger.log(
                                f"Maintaining curriculum difficulty at floor {current_curriculum_floor}. " +
                                f"Current success rate: {success_rate:.2f}", 
                                "CURRICULUM"
                            )
                
                # Apply curriculum settings to environment
                for key, value in curriculum_config.items():
                    env.reset_parameters.set_float_parameter(key, float(value))
                    
            except Exception as e:
                print(f"Error applying curriculum settings: {e}")
                traceback.print_exc()

        # Initialize frame stack
        training_logger.log_episode_start(episode_count + 1)  # Log new episode
        logger.info(f"Resetting environment...")
        try:
            obs = env.reset()
            logger.info(f"Environment reset complete. Observation shape: {obs[0].shape}")
        except UnityCommunicatorStoppedException as e:
            error_msg = f"Error during initial reset: {e}"
            print(error_msg)
            training_logger.log(error_msg, "ERROR")
            env.close()
            return
            
        obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
        for _ in range(4):
            frame_stack.append(obs)
        state = np.concatenate(frame_stack, axis=0)
        obs = torch.tensor(state, dtype=torch.float32).to(device)

        # Track key usage and door opening
        env._previous_keys = None
        env._previous_position = None
        
        # Collect trajectory
        steps_this_episode = 0
        trajectory_length = 1024  # Shorter trajectory length for more frequent updates
        max_episode_steps = 4000  # Set a maximum number of steps per episode to prevent getting stuck
        
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
        
        for step in range(trajectory_length):
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
                if current_floor > max_floor_reached:
                    max_floor_reached = current_floor
                    floor_msg = f"New floor reached: {current_floor}"
                    print(floor_msg)
                    training_logger.log_significant_event("FLOOR", floor_msg)
                    
                    # Add direct floor checkpoint saving here
                    try:
                        floor_checkpoint_path = os.path.join(args.log_dir, f"floor_{current_floor}.pth")
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
            
            # Enhanced reward shaping
            move_idx, rot_idx, jump_idx, _ = action
            shaped_reward = reward
            reward_components['base'] += reward
            
            # Track key usage and door opening
            current_keys = info["total_keys"]
            if hasattr(env, '_previous_keys') and env._previous_keys is not None:
                # If keys decreased without collecting new ones, a door was opened
                if current_keys < env._previous_keys:
                    door_bonus = 0.5  # Increased reward for door opening
                    shaped_reward += door_bonus
                    door_msg = f"Door opened! Reward bonus added: +{door_bonus}"
                    print(door_msg)
                    training_logger.log_significant_event("DOOR", door_msg)
                    door_openings += 1
                    metrics_tracker.update('door_openings', door_openings)
                    reward_components['door_bonus'] += door_bonus
                    
                    # Record door location in episodic memory
                    episodic_memory.add_door_location(
                        info["current_floor"], 
                        (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
                    )
                    
                # If keys increased, a key was collected
                elif current_keys > env._previous_keys:
                    key_bonus = 0.5  # Increased reward for key collection
                    shaped_reward += key_bonus
                    key_msg = f"Key collected! Reward bonus added: +{key_bonus}"
                    print(key_msg)
                    training_logger.log_significant_event("KEY", key_msg)
                    key_collections += 1
                    metrics_tracker.update('key_collections', key_collections)
                    reward_components['key_bonus'] += key_bonus
                    
                    # Record key location in episodic memory
                    episodic_memory.add_key_location(
                        info["current_floor"], 
                        (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
                    )
            env._previous_keys = current_keys
            
            # Use episodic memory to provide hints to the agent
            current_position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
            
            # If agent has a key and is near a door location, give a small hint
            if current_keys > 0 and episodic_memory.is_door_location_nearby(current_floor, current_position):
                door_hint_bonus = 0.05
                shaped_reward += door_hint_bonus
                reward_components['door_bonus'] += door_hint_bonus
                
            # If agent has no keys and is near a key location, give a small hint
            if current_keys == 0 and episodic_memory.is_key_location_nearby(current_floor, current_position):
                key_hint_bonus = 0.05
                shaped_reward += key_hint_bonus
                reward_components['key_bonus'] += key_hint_bonus
                
            # Encourage forward movement but with less extreme reward
            if move_idx == 1:  # Forward movement
                forward_bonus = 0.005  # Reduced from 0.01
                shaped_reward += forward_bonus
                reward_components['forward_movement'] += forward_bonus
            # Small penalty for staying still
            elif move_idx == 0:
                stay_penalty = -0.001  # Kept small
                shaped_reward += stay_penalty
                reward_components['stay_still_penalty'] += stay_penalty
                
            # MUCH smaller penalty for rotation - important for navigation
            if rot_idx != 0:
                rot_penalty = -0.0001  # Reduced from -0.0005
                shaped_reward += rot_penalty
                reward_components['rotation_penalty'] += rot_penalty
                
            # MUCH smaller penalty for jumping - critical for obstacle tower
            if jump_idx == 1:
                # Only penalize jumping if it's not being used to navigate
                # Check if there was significant movement after jump
                if hasattr(env, '_previous_position') and env._previous_position is not None:
                    distance = sum((current_position[i] - env._previous_position[i])**2 for i in range(3))**0.5
                    if distance < 0.1:  # If no significant movement after jump
                        jump_penalty = -0.0001  # Greatly reduced from -0.005
                        shaped_reward += jump_penalty
                        reward_components['jump_penalty'] += jump_penalty
                
            # Time penalty to encourage faster completion
            time_penalty = -0.0001
            shaped_reward += time_penalty
            reward_components['time_penalty'] += time_penalty
            
            # Enhanced exploration bonus based on visit count, with better scaling
            visit_count = info.get("visit_count", 0)
            current_floor = info.get("current_floor", 0)
            
            # Calculate adaptive exploration factors
            # 1. Base exploration factor starts high and decays over training, but maintains a minimum value
            progress = min(1.0, steps_done / (0.8 * max_steps))  # Slower decay over 80% of training
            base_exploration_factor = max(0.3, 1.0 - progress)  # Decay from 1.0 to 0.3
            
            # 2. Floor-specific factor - higher floors get higher exploration bonuses
            floor_factor = 1.0 + 0.2 * current_floor  # Each floor increases exploration bonus by 20%
            
            # 3. Uncertainty factor - areas visited less get higher bonuses
            if visit_count is not None:
                if visit_count == 0:  # Never visited before
                    uncertainty = 1.0  # Maximum uncertainty
                elif visit_count < 3:  # Visited only a few times
                    uncertainty = 0.7  # High uncertainty
                elif visit_count < 10:  # Visited several times
                    uncertainty = 0.4  # Medium uncertainty
                else:  # Visited many times
                    uncertainty = 0.2  # Low uncertainty
                
                # Calculate final exploration bonus with all factors
                exploration_bonus = 0.1 * base_exploration_factor * floor_factor * uncertainty
                
                if exploration_bonus > 0:
                    shaped_reward += exploration_bonus
                    reward_components['exploration_bonus'] += exploration_bonus
                    
                    # Log significant exploration bonuses
                    if exploration_bonus > 0.05 and steps_done % 100 == 0:
                        training_logger.log(
                            f"Exploration bonus: +{exploration_bonus:.3f} " +
                            f"(floor: {current_floor}, visits: {visit_count})",
                            "EXPLORATION"
                        )
            
            # Enhance movement reward with directional bias toward unexplored areas
            if hasattr(env, '_previous_position') and env._previous_position is not None:
                # Calculate distance moved
                distance = sum((current_position[i] - env._previous_position[i])**2 for i in range(3))**0.5
                
                # Higher reward for significant movement
                if distance > 0.5:
                    # Get movement direction
                    if distance > 0:
                        direction = [(current_position[i] - env._previous_position[i])/distance 
                                    for i in range(3)]
                    else:
                        direction = [0, 0, 0]
                    
                    # Check if agent is moving toward unexplored areas
                    target_info = key_door_memory.get_directions_to_target(
                        current_position, current_floor, current_keys > 0
                    )
                    
                    directional_bonus = 0.0
                    
                    if target_info:
                        # Calculate dot product to see if agent is moving toward target
                        target_dir = target_info['direction']
                        dot_product = sum(direction[i] * target_dir[i] for i in range(min(len(direction), len(target_dir))))
                        
                        # Higher reward for moving toward targets
                        if dot_product > 0:
                            # Scale by how aligned the movement is with target direction
                            alignment_factor = (dot_product + 1) / 2  # Scale from 0-1
                            directional_bonus = 0.02 * distance * alignment_factor * floor_factor
                    else:
                        # Basic movement bonus when no targets known
                        directional_bonus = 0.01 * distance * floor_factor
                    
                    if directional_bonus > 0:
                        shaped_reward += directional_bonus
                        reward_components['exploration_bonus'] += directional_bonus
            
            # Store current position for next step comparison
            env._previous_position = current_position
                
            # Enhanced floor completion bonus with progressive scaling
            if info["current_floor"] > current_floor:
                # Progressive floor bonus - higher floors get bigger rewards
                base_floor_bonus = 2.0
                floor_progression = info["current_floor"] - current_floor
                floor_bonus = base_floor_bonus * (1 + 0.5 * floor_progression)  # +50% per floor skipped
                
                shaped_reward += floor_bonus
                episodic_memory.mark_floor_complete(current_floor)
                key_door_memory.mark_floor_complete(current_floor)
                
                # Track floor achievement
                new_floor = info["current_floor"]
                if new_floor > max_floor_reached:
                    max_floor_reached = new_floor
                    metrics_tracker.update('max_floor_reached', max_floor_reached)
                    
                    # Additional milestone bonus for reaching a new max floor
                    if new_floor > current_floor + 1:  # Skip multiple floors
                        milestone_bonus = 5.0  # Substantial bonus for significant progress
                        shaped_reward += milestone_bonus
                        print(f"Milestone achieved! Reached new max floor {new_floor}! +{milestone_bonus} bonus")
                        training_logger.log_significant_event(
                            "NEW_MAX_FLOOR", 
                            f"Reached new maximum floor {new_floor} (skipped {new_floor - current_floor - 1} floors)"
                        )
                    else:
                        print(f"Reached new max floor {new_floor}!")
                        training_logger.log_significant_event(
                            "NEW_MAX_FLOOR", 
                            f"Reached new maximum floor {new_floor}"
                        )
                
                # Update current floor to the new floor
                current_floor = new_floor
                reward_components['floor_bonus'] += floor_bonus
                print(f"New floor reached: {current_floor}! Bonus reward added: +{floor_bonus}")
                
                # Reset exploration tracking for new floor
                if hasattr(env, '_visited_positions'):
                    env._visited_positions = {}
                    print("Reset exploration tracking for new floor")
                
                # Add extra hint reward if we have keys to indicate they might be needed
                if info["total_keys"] > 0:
                    key_usage_hint = 1.0
                    shaped_reward += key_usage_hint
                    print(f"Carrying keys to new floor! +{key_usage_hint} reward")
                    
                # Log the significant achievement
                training_logger.log_significant_event(
                    "FLOOR_COMPLETE", 
                    f"Completed floor {current_floor-1}, moving to floor {current_floor}"
                )
            
            # Log detailed environment interaction if verbose
            if steps_done % 100 == 0:
                training_logger.log_environment_interaction(
                    action=action, 
                    reward=reward, 
                    shaped_reward=shaped_reward, 
                    info=info, 
                    step_num=steps_done
                )

            # Store previous observation for detection
            if hasattr(env, '_previous_obs'):
                previous_obs = env._previous_obs
            else:
                previous_obs = None
            env._previous_obs = next_obs[0]  # Assuming this is the visual observation

            # Check for key detection
            key_detected = detect_key_visually(next_obs[0], previous_obs)
            if key_detected:
                # Extra visual reward - increased from 2.0 to 3.0
                key_visual_bonus = 3.0
                shaped_reward += key_visual_bonus
                print(f"Key visually detected! +{key_visual_bonus} reward")
                
                # Update memory with current position
                current_position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
                key_door_memory.add_key_location(info["current_floor"], current_position)
                # Track key collection for metrics
                key_collections += 1
                metrics_tracker.update('key_collections', key_collections)
                training_logger.log_significant_event("KEY", f"Key detected at floor {info['current_floor']}, position {current_position}")

            # Check for key collection by comparing with previous state
            has_key = info["total_keys"] > 0
            if hasattr(env, '_previous_keys') and env._previous_keys is not None:
                if has_key and env._previous_keys == 0:
                    # Additional reward for actually collecting the key
                    key_collection_bonus = 4.0
                    shaped_reward += key_collection_bonus
                    print(f"Key collected! +{key_collection_bonus} reward")
                    training_logger.log_significant_event("KEY_COLLECTED", f"Key collected at floor {info['current_floor']}")
            # Update the key count tracker
            env._previous_keys = info["total_keys"]

            # Enhance key possession awareness
            if has_key:
                # If agent has keys, provide a consistent small bonus
                has_key_bonus = 0.1  # Increased from 0.05
                shaped_reward += has_key_bonus
                
                # Check for doors visually when agent has keys
                door_detected = detect_door_visually(next_obs[0])
                if door_detected:
                    door_visual_bonus = 0.5  # Increased from 0.2
                    shaped_reward += door_visual_bonus
                    print(f"Door visually detected with key! +{door_visual_bonus} reward")
                    
                    # Store door position in memory
                    door_position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
                    key_door_memory.add_door_location(info["current_floor"], door_position)
                    
                # Check for door opening by key loss
                if hasattr(env, '_previous_keys') and env._previous_keys > info["total_keys"]:
                    # Key was used - likely opened a door
                    door_open_bonus = 5.0  # New substantial reward for using a key
                    shaped_reward += door_open_bonus
                    print(f"Door opened with key! +{door_open_bonus} reward")
                    door_openings += 1
                    metrics_tracker.update('door_openings', door_openings)
                    training_logger.log_significant_event("DOOR_OPENED", f"Door opened at floor {info['current_floor']}")
                    
                    # Store successful key-door interaction
                    if hasattr(env, '_last_key_position') and env._last_key_position is not None:
                        key_door_memory.store_key_door_sequence(
                            env._last_key_position, 
                            current_position, 
                            info["current_floor"]
                        )
                
                # Add proximity bonus based on memory - enhanced version
                door_proximity_bonus = key_door_memory.get_proximity_bonus(
                    current_position, info["current_floor"], has_key=True
                )
                # Apply a multiplier to make proximity more significant
                door_proximity_bonus *= 1.5
                shaped_reward += door_proximity_bonus
                
                # Store position where key was obtained
                if not hasattr(env, '_last_key_position') or env._last_key_position is None:
                    env._last_key_position = current_position
            else:
                # Add key proximity bonus when agent has no keys - enhanced
                key_proximity_bonus = key_door_memory.get_proximity_bonus(
                    current_position, info["current_floor"], has_key=False
                )
                # Apply a multiplier to make proximity more significant
                key_proximity_bonus *= 1.5
                shaped_reward += key_proximity_bonus
                
                # Reset key position tracking
                env._last_key_position = None

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
            steps_this_episode += 1

            obs = next_obs
            state = next_state
            
            # Add termination condition for extremely long episodes
            if steps_this_episode >= max_episode_steps and not done:
                print(f"Terminating episode after {steps_this_episode} steps to prevent getting stuck")
                training_logger.log_significant_event("TIMEOUT", f"Episode terminated after {steps_this_episode} steps")
                done = True  # Force termination
            
            if done or steps_this_episode >= trajectory_length:
                if done:
                    # Log episode statistics
                    episode_count += 1
                    metrics_tracker.update('episode_rewards', episode_reward)
                    metrics_tracker.update('episode_lengths', episode_length)
                    metrics_tracker.update('episode_floors', info['current_floor'])
                    
                    elapsed_time = time.time() - start_time
                    steps_per_sec = steps_done / elapsed_time
                    metrics_tracker.update('steps_per_second', steps_per_sec)
                    
                    # Log episode completion with detailed stats
                    episode_stats = {
                        'reward': episode_reward,
                        'length': episode_length,
                        'floor': info['current_floor'],
                        'max_floor': max_floor_reached,
                        'steps': steps_done,
                        'steps_per_sec': steps_per_sec
                    }
                    training_logger.log_episode_complete(episode_stats)
                    
                    # Log reward breakdown
                    training_logger.log_reward_breakdown(
                        base_reward=reward_components['base'],
                        shaped_components={k: v for k, v in reward_components.items() if k != 'base'}
                    )
                    
                    print(f"Episode {episode_count} - Reward: {episode_reward:.2f}, Length: {episode_length}, "
                          f"Floor: {info['current_floor']}, Max Floor: {max_floor_reached}, Steps: {steps_done}, "
                          f"Steps/sec: {steps_per_sec:.2f}")
                    
                break
        
        # If episode was cut off, we need to compute the value of the final state
        if not done:
            with torch.no_grad():
                obs_batched = obs.unsqueeze(0)
                _, next_value = model(obs_batched)
                next_value = next_value.item()
        else:
            next_value = 0.0  # Terminal state has value 0
            
            # Reset for next episode
            try:
                obs = env.reset()
            except UnityCommunicatorStoppedException as e:
                error_msg = f"Error during reset after episode: {e}"
                print(error_msg)
                training_logger.log(error_msg, "ERROR")
                env.close()
                return
                
            obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
            frame_stack = deque(maxlen=4)
            for _ in range(4):
                frame_stack.append(obs)
            state = np.concatenate(frame_stack, axis=0)
            obs = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Reset key tracking
            env._previous_keys = None
            env._previous_position = None

        # Add current trajectory to experience replay if enabled
        if args.experience_replay:
            add_to_replay(replay_buffer, states, actions, rewards, log_probs, values, dones)
            
            # Log replay buffer size occasionally
            if steps_done % 10000 == 0:
                buffer_size = len(replay_buffer['states'])
                training_logger.log(f"Experience replay buffer size: {buffer_size}/{max_replay_size}", "REPLAY")
        
        # Perform PPO update after collecting enough steps
        if len(replay_buffer['states']) >= ppo.batch_size:
            # Log replay buffer stats before sampling
            buffer_size = len(replay_buffer['states'])
            
            # Calculate floor distribution in the buffer for adaptive sampling
            floor_distribution = {}
            for i in range(buffer_size):
                # Extract floor info if available in state or use default
                obs_floor = 0  # Default floor
                if isinstance(replay_buffer['states'][i], np.ndarray) and replay_buffer['states'][i].shape[0] > 0:
                    # Try to extract floor from observation if available
                    # This is a simplified example - adapt to your state representation
                    obs_floor = int(np.mean(replay_buffer['states'][i][:10]) * 10) % 10  # Simple heuristic
                
                if obs_floor not in floor_distribution:
                    floor_distribution[obs_floor] = 0
                floor_distribution[obs_floor] += 1
            
            # Log floor distribution when significant
            if steps_done % 10000 == 0:
                floor_dist_str = ", ".join([f"Floor {f}: {c}" for f, c in sorted(floor_distribution.items())])
                training_logger.log(f"Replay buffer floor distribution: {floor_dist_str}", "REPLAY")
            
            # Sample from replay buffer with priority for higher floors
            states, actions, returns, advantages, old_log_probs = sample_from_replay(
                replay_buffer, ppo.batch_size
            )
            
            # Check if we have enough samples
            if len(states) >= ppo.batch_size // 2:  # Allow for smaller batch if needed
                # Update policy
                metrics = ppo.update(states, actions, old_log_probs, returns, advantages)
                update_count += 1  # Increment update counter
                
                # Log metrics with more detail
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
                
                # Adaptive entropy regulation - increase entropy if stuck on a floor
                if current_floor <= 2 and update_count % 50 == 0:
                    # Check if we've been stuck on the same floor for a while
                    if len(metrics_tracker.metrics.get('episode_floors', [])) > 10:
                        recent_floors = metrics_tracker.metrics['episode_floors'][-10:]
                        max_recent_floor = max(recent_floors) if recent_floors else 0
                        
                        if max_recent_floor <= 2:
                            # We're stuck on early floors - increase entropy temporarily
                            old_entropy = ppo.ent_reg
                            ppo.ent_reg = min(0.1, ppo.ent_reg * 1.5)  # Increase entropy up to 0.1
                            training_logger.log(
                                f"Increasing entropy from {old_entropy:.4f} to {ppo.ent_reg:.4f} to escape floor {max_recent_floor}",
                                "TUNING"
                            )
                
                # Clear replay buffer after update
                replay_buffer = {
                    'states': [], 'actions': [], 'rewards': [],
                    'log_probs': [], 'values': [], 'dones': []
                }
            else:
                training_logger.log(f"Skipping update - not enough samples ({len(states)}/{ppo.batch_size})", "UPDATE")
        
        # Run evaluation periodically
        if steps_done - last_eval_time >= args.eval_interval:
            eval_reward, eval_max_floor = evaluate_policy(args.eval_episodes)
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
            
        # Save metrics and checkpoints periodically
        current_time = time.time()
        
        # Log training summary every minute
        if current_time - last_log_time > 60:  # Every minute
            training_logger.log_training_summary(
                metrics_tracker=metrics_tracker,
                elapsed_time=current_time - start_time,
                steps_done=steps_done
            )
            last_log_time = current_time
        
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

        # Add this check after PPO updates
        if update_count % 200 == 0 or ppo.optimizer.param_groups[0]['lr'] < 5e-5:
            ppo.reset_optimizer_state()

        # Apply floor-specific reward scaling to prioritize progression
        # Lower floors get reduced reward scale over time to encourage moving higher
        floor_scaling_factor = 1.0
        
        # Gradually reduce reward scale for easier floors as training progresses
        if steps_done > 100000:  # After initial learning period
            current_max_floor = max_floor_reached
            progress_ratio = min(1.0, steps_done / max_steps)
            
            # Calculate floor-specific scaling
            # Floors far below the max get diminished rewards
            if current_floor < current_max_floor - 1:
                # Scale down rewards for floors more than 1 below max
                floor_gap = current_max_floor - current_floor
                # Stronger reduction as training progresses and gap increases
                reduction = 0.3 * progress_ratio * min(floor_gap, 3)  # Cap at 3 floors difference
                floor_scaling_factor = max(0.5, 1.0 - reduction)  # Don't go below 50%
                
                # Log when we apply significant scaling
                if floor_scaling_factor < 0.8 and step % 100 == 0:
                    training_logger.log(
                        f"Applying floor scaling factor {floor_scaling_factor:.2f} " +
                        f"(floor {current_floor} vs max {current_max_floor})",
                        "REWARD_SHAPING"
                    )
            elif current_floor >= current_max_floor:
                # Bonus scaling for pushing to new floors
                floor_scaling_factor = 1.2  # 20% bonus for being at the frontier
            
            # Apply the scaling factor
            shaped_reward *= floor_scaling_factor

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
    
    # Close logger
    training_logger.close()
    
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