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
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create environment
    print("INFO:obstacle_tower:Setting up environment...")
    env = create_obstacle_tower_env(
        executable_path=args.env_path,
        realtime_mode=False,
        no_graphics=True
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
        current_curriculum_floor = 0
        curriculum_success_threshold = 3
        curriculum_attempt_threshold = 10
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
            
        # Sample indices
        indices = np.random.choice(buffer_size, min(buffer_size, batch_size), replace=False)
        
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
                    
                    # Check if agent completed the floor
                    if info["current_floor"] > current_curriculum_floor:
                        curriculum_successes += 1
                        training_logger.log(f"Curriculum success! {curriculum_successes}/{curriculum_success_threshold} successes at floor {current_curriculum_floor}", "CURRICULUM")
                        
                        # If agent has had enough successes, increase difficulty
                        if curriculum_successes >= curriculum_success_threshold:
                            current_curriculum_floor += 1
                            curriculum_successes = 0
                            curriculum_attempts = 0
                            curriculum_config["starting-floor"] = current_curriculum_floor
                            training_logger.log(f"Curriculum difficulty increased to floor {current_curriculum_floor}", "CURRICULUM")
                    
                    # If agent is struggling, decrease difficulty
                    elif curriculum_attempts >= curriculum_attempt_threshold and current_curriculum_floor > 0:
                        current_curriculum_floor = max(0, current_curriculum_floor - 1)
                        curriculum_successes = 0
                        curriculum_attempts = 0
                        curriculum_config["starting-floor"] = current_curriculum_floor
                        training_logger.log(f"Curriculum difficulty decreased to floor {current_curriculum_floor}", "CURRICULUM")
                
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
            
            # Enhanced exploration bonus based on visit count, but with decay
            visit_count = info.get("visit_count", 0)
            # Calculate training progress for decaying exploration
            progress = min(1.0, steps_done / (0.5 * max_steps))  # Decay over first half of training
            exploration_decay = max(0.2, 1.0 - progress)  # Decay from 1.0 to 0.2
            
            if visit_count is not None:
                if visit_count == 0:  # Never visited before
                    exploration_bonus = 0.05 * exploration_decay  # Reduced and decaying over time
                    shaped_reward += exploration_bonus
                    reward_components['exploration_bonus'] += exploration_bonus
                elif visit_count < 3:  # Visited only a few times
                    exploration_bonus = 0.02 * exploration_decay  # Reduced and decaying over time
                    shaped_reward += exploration_bonus
                    reward_components['exploration_bonus'] += exploration_bonus
            
            # Also reward distance moved but with lower weight
            if hasattr(env, '_previous_position') and env._previous_position is not None:
                # Calculate distance moved
                distance = sum((current_position[i] - env._previous_position[i])**2 for i in range(3))**0.5
                if distance > 0.5:  # If moved significantly
                    distance_bonus = 0.005 * distance * exploration_decay  # Reduced and decaying
                    shaped_reward += distance_bonus
                    reward_components['exploration_bonus'] += distance_bonus
            env._previous_position = current_position
                
            # Floor completion bonus - already handled by environment but ensure it's significant
            if info["current_floor"] > current_floor:
                floor_bonus = 2.0  # Doubled from 1.0 to prioritize floor progression
                shaped_reward += floor_bonus
                episodic_memory.mark_floor_complete(current_floor)
                current_floor = info["current_floor"]
                reward_components['floor_bonus'] += floor_bonus
                print(f"New floor reached: {current_floor}! Bonus reward added: +{floor_bonus}")
            
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
                # Extra visual reward
                key_visual_bonus = 2.0
                shaped_reward += key_visual_bonus
                print(f"Key visually detected! +{key_visual_bonus} reward")
                
                # Update memory with current position
                current_position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
                key_door_memory.add_key_location(info["current_floor"], current_position)

            # Enhance key possession awareness
            if info["total_keys"] > 0:
                # If agent has keys, provide a consistent small bonus
                has_key_bonus = 0.05
                shaped_reward += has_key_bonus
                
                # Check for doors visually when agent has keys
                door_detected = detect_door_visually(next_obs[0])
                if door_detected:
                    door_visual_bonus = 0.2
                    shaped_reward += door_visual_bonus
                    print(f"Door visually detected with key! +{door_visual_bonus} reward")
                    
                # Add proximity bonus based on memory
                door_proximity_bonus = key_door_memory.get_proximity_bonus(
                    current_position, info["current_floor"], has_key=True
                )
                shaped_reward += door_proximity_bonus
            else:
                # Add key proximity bonus when agent has no keys
                key_proximity_bonus = key_door_memory.get_proximity_bonus(
                    current_position, info["current_floor"], has_key=False
                )
                shaped_reward += key_proximity_bonus

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
                    metrics_tracker.update('episode_floors', info["current_floor"])
                    
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
            # Sample from replay buffer
            states, actions, returns, advantages, old_log_probs = sample_from_replay(
                replay_buffer, ppo.batch_size
            )
            
            # Update policy
            metrics = ppo.update(states, actions, old_log_probs, returns, advantages)
            update_count += 1  # Increment update counter
            
            # Log metrics
            training_logger.log_update(metrics)
            
            # Clear replay buffer after update
            replay_buffer = {
                'states': [], 'actions': [], 'rewards': [],
                'log_probs': [], 'values': [], 'dones': []
            }
        
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
            
            # Save metrics and plot
            metrics_tracker.save()
            
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

    # Final save
    metrics_tracker.update_from_ppo(ppo)
    metrics_tracker.save()
    
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