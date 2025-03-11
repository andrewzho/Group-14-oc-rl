import argparse
import numpy as np
import torch
import os
import time
import sys
import matplotlib.pyplot as plt
from collections import deque
import datetime
import json
import cv2
import traceback

from src.model import PPONetwork
from src.ppo import PPO
from src.utils import normalize, save_checkpoint, load_checkpoint, ActionFlattener, MetricsTracker, TrainingLogger
from src.create_env import create_obstacle_tower_env
from src.detection import detect_key_visually, detect_door_visually, detect_key_collection, detect_door_opening
from src.icm import ICM
from src.memory import EnhancedKeyDoorMemory

# Track key and door interactions to improve exploration
class ObstacleTracker:
    def __init__(self, capacity=1000):
        # Memory for keys and doors to help guide exploration
        self.key_door_memory = EnhancedKeyDoorMemory(decay_factor=0.95, horizon=1000)
        
        # Current state
        self.has_key = False
        self.prev_observation = None
        self.prev_keys = 0
        self.current_floor = 0
        self.visited_positions = {}  # Track visited positions (floor, x, z) -> count
        self.position_resolution = 1.0  # Grid resolution for position tracking
        
        # Stats
        self.keys_collected = 0
        self.doors_opened = 0
        self.floors_completed = 0
        
    def reset(self):
        """Reset tracker for new episode but maintain memory across episodes."""
        self.has_key = False
        self.prev_observation = None
        self.prev_keys = 0
        self.current_floor = 0
        # Don't reset key_door_memory or visited_positions to maintain knowledge
    
    def update(self, observation, info):
        """Update tracker state and detect key-door interactions."""
        detected_key = False
        detected_door = False
        floor_changed = False
        
        # Get current position estimate (x, z coordinates)
        position = (
            info.get('x_pos', 0),
            info.get('z_pos', 0)
        )
        
        # Track floor changes
        new_floor = info.get('current_floor', 0)
        if new_floor > self.current_floor:
            self.current_floor = new_floor
            floor_changed = True
            self.key_door_memory.mark_floor_complete(new_floor - 1)  # Mark previous floor complete
            self.floors_completed += 1
            # Reset visited positions for new floor
            self.visited_positions = {k: v for k, v in self.visited_positions.items() 
                                     if not k.startswith(f"{new_floor}_")}
        
        # Detect key collection
        current_keys = info.get('total_keys', 0)
        if current_keys > self.prev_keys:
            detected_key = True
            self.has_key = True
            self.keys_collected += 1
            # When a key is collected, remember its location
            self.key_door_memory.add_key_location(self.current_floor, position)
            print(f"Key collected at floor {self.current_floor}, position {position}")
        self.prev_keys = current_keys
        
        # Detect door opening with OpenCV-based detection
        # Only try to detect doors if agent has a key
        if self.has_key and self.prev_observation is not None:
            if detect_door_opening(self.prev_observation, observation, self.has_key, True):
                detected_door = True
                self.has_key = False
                self.doors_opened += 1
                # When a door is opened, remember its location
                self.key_door_memory.add_door_location(self.current_floor, position)
                print(f"Door opened at floor {self.current_floor}, position {position}")
        
        # Visual detection for keys and doors as backup
        if not detected_key and self.prev_observation is not None:
            detected_key = detect_key_visually(observation, self.prev_observation)
            if detected_key:
                print(f"Key visually detected at floor {self.current_floor}")
                self.key_door_memory.add_key_location(self.current_floor, position)
        
        if not detected_door and detect_door_visually(observation):
            print(f"Door visually detected at floor {self.current_floor}")
            self.key_door_memory.add_door_location(self.current_floor, position)
        
        # Track visited positions for exploration bonuses
        pos_key = f"{self.current_floor}_{round(position[0]/self.position_resolution)}_{round(position[1]/self.position_resolution)}"
        self.visited_positions[pos_key] = self.visited_positions.get(pos_key, 0) + 1
        
        # Update key-door interaction tracking
        self.key_door_memory.update_key_detection(position, self.has_key)
        
        # Remember observation for next time
        self.prev_observation = observation
        
        return {
            'detected_key': detected_key,
            'detected_door': detected_door,
            'floor_changed': floor_changed,
            'has_key': self.has_key,
            'visit_count': self.visited_positions.get(pos_key, 0)
        }
    
    def get_exploration_bonus(self, observation, info):
        """Calculate exploration bonus based on novelty and memory."""
        bonus = 0.0
        
        # Get position
        position = (
            info.get('x_pos', 0),
            info.get('z_pos', 0)
        )
        
        # 1. Bonus for rarely visited locations (inverse of visit count)
        pos_key = f"{self.current_floor}_{round(position[0]/self.position_resolution)}_{round(position[1]/self.position_resolution)}"
        visit_count = self.visited_positions.get(pos_key, 0)
        if visit_count > 0:
            bonus += 0.01 / np.sqrt(visit_count)  # Diminishing returns for repeated visits
        
        # 2. Bonus for proximity to keys or doors based on memory
        proximity_bonus = self.key_door_memory.get_proximity_bonus(
            position, self.current_floor, self.has_key, threshold=10.0)
        bonus += proximity_bonus
        
        # 3. Directional guidance toward keys/doors (simplified here)
        directions = self.key_door_memory.get_directions_to_target(
            position, self.current_floor, self.has_key)
        if directions is not None:
            # Small constant bonus for having a target direction
            bonus += 0.02
        
        return bonus


def preprocess_observation(obs):
    """Process observation for the neural network."""
    # For ObstacleTower environment
    if isinstance(obs, tuple):
        # Get only the visual observation
        img_obs = obs[0]
    else:
        img_obs = obs
    
    # Convert to float32 and normalize to [0,1]
    img_obs = img_obs.astype(np.float32) / 255.0
    
    # Convert to correct format for the model
    # Expected shape: (C, H, W) for PyTorch
    if img_obs.shape[2] == 3:  # Check if channels is last dimension
        img_obs = np.transpose(img_obs, (2, 0, 1))
    
    return img_obs


def create_frame_stack(obs, frame_stack_size=4):
    """Create a stack of frames from a single observation."""
    if isinstance(obs, tuple):
        img_obs = obs[0]
    else:
        img_obs = obs
    
    # Normalize and convert to channels-first format
    processed = preprocess_observation(obs)
    
    # Create stack with repeated initial frame
    stack = np.tile(processed, (frame_stack_size, 1, 1, 1))
    
    # Reshape to expected format for model: [C*stack_size, H, W]
    return stack.reshape(-1, *processed.shape[1:])


def update_frame_stack(frame_stack, obs):
    """Update frame stack with new observation."""
    # Process new observation
    processed = preprocess_observation(obs)
    
    # Shift stack and add new frame
    # Assuming frame_stack shape is [C*stack_size, H, W]
    channels_per_frame = processed.shape[0]
    frame_stack = np.roll(frame_stack, -channels_per_frame, axis=0)
    frame_stack[-channels_per_frame:] = processed
    
    return frame_stack


def calculate_returns_and_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Calculate returns and advantage estimates.
    """
    returns = []
    advantages = []
    gae = 0
    
    # Convert to numpy arrays for easier handling
    rewards = np.array(rewards)
    values = np.array(values)
    dones = np.array(dones)
    
    # Calculate advantages in reverse order
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Value of terminal state is 0
        else:
            next_value = values[t + 1]
            
        # If episode ended, next value is 0
        if dones[t]:
            next_value = 0
            
        # TD error: r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        
        # Generalized Advantage Estimation (GAE)
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        
        # Insert at beginning since we're going backwards
        advantages.insert(0, gae)
        
        # Return = advantage + value
        returns.insert(0, gae + values[t])
    
    return np.array(returns), np.array(advantages)


def main(args):
    """Main training function for Obstacle Tower with enhanced exploration."""
    
    # Create log directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_dir is None:
        args.log_dir = f"./logs/explore_{timestamp}"
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Initialize logging
    metrics = MetricsTracker(args.log_dir)
    logger = TrainingLogger(args.log_dir, log_frequency=5)
    logger.log_hyperparameters(vars(args))
    
    # Save args to file
    with open(os.path.join(args.log_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seeds
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
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
    
    # Set environment seed
    if args.seed is not None:
        env.seed(args.seed)
    
    # Track floor mastery for robust curriculum learning
    floor_completions = {}  # floor -> list of recent success/failure (1/0)
    consecutive_completions = {}  # floor -> count of consecutive completions
    
    # Set starting floor if curriculum learning is enabled
    if args.curriculum:
        # Start at floor 0 or a specific starting floor
        starting_floor = args.starting_floor if args.starting_floor is not None else 0
        env.floor(starting_floor)
        print(f"Starting at floor {starting_floor} (curriculum learning)")
        
        # Initialize floor tracking
        for f in range(starting_floor + 1):
            floor_completions[f] = []
            consecutive_completions[f] = 0
    
    # Get environment information
    action_flattener = ActionFlattener(env.action_space.nvec)
    action_size = action_flattener.action_space.n
    print(f"Action space size: {action_size}")
    
    # Initialize obstacle tracker for key-door interactions
    obstacle_tracker = ObstacleTracker()
    
    # Initialize frame stack
    obs = env.reset()
    frame_stack = create_frame_stack(obs)
    state_shape = frame_stack.shape
    print(f"State shape: {state_shape}")
    
    # Create neural network
    model = PPONetwork(input_shape=state_shape, num_actions=action_size).to(device)
    logger.update_model(model)
    
    # Create ICM network for intrinsic motivation
    icm = ICM(input_shape=state_shape, action_dim=action_size, 
             feature_dim=256, forward_scale=0.2, inverse_scale=0.8).to(device)
    
    # Create PPO agent
    agent = PPO(
        model=model,
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        vf_coef=args.vf_coef,
        ent_reg=args.entropy_reg,  # Higher entropy for better exploration
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        lr_scheduler=args.lr_scheduler,
        adaptive_entropy=True,  # Enable adaptive entropy
        min_entropy=0.01,      # Keep some minimum entropy
        entropy_decay_factor=0.9999  # Slow decay for entropy
    )
    
    # Create optimizer for ICM
    icm_optimizer = torch.optim.Adam(icm.parameters(), lr=args.icm_lr)
    
    # Load checkpoint if provided
    update_counter = 0
    if args.checkpoint:
        model, metrics_dict, update_counter = load_checkpoint(model, args.checkpoint)
        if update_counter is None:
            update_counter = 0
        print(f"Loaded checkpoint, starting from update {update_counter}")
        
        # Try to load ICM if available
        try:
            icm_checkpoint = torch.load(args.checkpoint.replace('.pth', '_icm.pth'))
            icm.load_state_dict(icm_checkpoint['model_state_dict'])
            icm_optimizer.load_state_dict(icm_checkpoint['optimizer_state_dict'])
            print("Loaded ICM checkpoint")
        except:
            print("No ICM checkpoint found, starting with fresh ICM")
    
    # Initialize training variables
    num_steps = 0
    episode = 0
    max_floor_reached = 0
    training_start_time = time.time()
    last_save_time = training_start_time
    last_plot_time = training_start_time
    
    # For tracking FPS
    fps_start_time = time.time()
    fps_step_counter = 0
    fps_history = []
    
    # For tracking episodic stats
    total_extrinsic_rewards = 0
    total_intrinsic_rewards = 0
    episode_step_counter = 0
    episode_start_time = time.time()
    
    # Experience collection for PPO update
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    
    # Main training loop
    try:
        while num_steps < args.num_steps:
            # Reset at the start of an episode
            obs = env.reset()
            frame_stack = create_frame_stack(obs)
            done = False
            obstacle_tracker.reset()
            
            # Print episode start information
            episode += 1
            logger.log_episode_start(episode)
            print(f"\nStarting episode {episode}")
            
            # Episode loop
            while not done:
                # Increment step counters
                num_steps += 1
                episode_step_counter += 1
                fps_step_counter += 1
                
                # Get action from policy
                state_tensor = torch.FloatTensor(frame_stack).unsqueeze(0).to(device)
                with torch.no_grad():
                    policy_logits, value = model(state_tensor)
                    value = value.item()
                    dist = torch.distributions.Categorical(logits=policy_logits)
                    action_index = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action_index, device=device)).item()
                
                # Convert action index to environment action
                action = action_flattener.lookup_action(action_index)
                
                # Convert action list to numpy array before passing to environment
                action = np.array(action)
                
                # Step the environment
                next_obs, reward, done, info = env.step(action)
                
                # Update obstacle tracker and get exploration info
                tracker_info = obstacle_tracker.update(next_obs, info)
                
                # Calculate exploration bonus
                exploration_bonus = obstacle_tracker.get_exploration_bonus(next_obs, info)
                
                # Update frame stack with new observation
                next_frame_stack = update_frame_stack(frame_stack.copy(), next_obs)
                
                # Calculate intrinsic reward from ICM
                if args.use_icm:
                    state_tensor_next = torch.FloatTensor(next_frame_stack).unsqueeze(0).to(device)
                    action_tensor = torch.tensor([action_index], device=device)
                    
                    with torch.no_grad():
                        icm_results = icm.forward(state_tensor, state_tensor_next, action_tensor)
                        intrinsic_reward = icm_results['intrinsic_reward'].item()
                else:
                    intrinsic_reward = 0.0
                
                # Combine rewards: extrinsic (environment) + intrinsic (ICM) + exploration (memory-based)
                combined_reward = (
                    args.extrinsic_coef * reward + 
                    args.intrinsic_coef * intrinsic_reward +
                    args.exploration_coef * exploration_bonus
                )
                
                # Add key-door rewards from detection
                if tracker_info['detected_key']:
                    combined_reward += args.key_reward
                    print(f"Key reward: +{args.key_reward}")
                    
                if tracker_info['detected_door']:
                    combined_reward += args.door_reward
                    print(f"Door reward: +{args.door_reward}")
                
                if tracker_info['floor_changed']:
                    floor = info.get('current_floor', 0)
                    combined_reward += args.floor_reward
                    max_floor_reached = max(max_floor_reached, floor)
                    print(f"Floor changed reward: +{args.floor_reward} (Floor {floor})")
                
                # Track rewards
                total_extrinsic_rewards += reward
                total_intrinsic_rewards += intrinsic_reward
                
                # Store experience for PPO update
                states.append(frame_stack)
                actions.append(action_index)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(combined_reward)
                dones.append(done)
                
                # Update state for next iteration
                frame_stack = next_frame_stack
                
                # Render if requested
                if args.render:
                    env.render()
                
                # Print step information occasionally
                if num_steps % 100 == 0:
                    elapsed = time.time() - fps_start_time
                    fps = fps_step_counter / elapsed if elapsed > 0 else 0
                    fps_history.append(fps)
                    fps_step_counter = 0
                    fps_start_time = time.time()
                    
                    print(f"Step {num_steps}/{args.num_steps} | "
                          f"Episode {episode} | "
                          f"Floor {info.get('current_floor', 0)} | "
                          f"Reward {reward:.2f} + {intrinsic_reward:.2f} = {combined_reward:.2f} | "
                          f"FPS: {fps:.1f}")
                
                # Update policy if enough steps have been collected
                if len(states) >= args.horizon or done:
                    # Calculate returns and advantages
                    returns, advantages = calculate_returns_and_advantages(
                        rewards, values, dones, args.gamma, args.gae_lambda)
                    
                    # Convert data to tensors
                    b_states = torch.FloatTensor(np.array(states)).to(device)
                    b_actions = torch.LongTensor(actions).to(device)
                    b_log_probs = torch.FloatTensor(log_probs).to(device)
                    b_returns = torch.FloatTensor(returns).to(device)
                    b_advantages = torch.FloatTensor(advantages).to(device)
                    
                    # Update PPO
                    update_metrics = agent.update(b_states, b_actions, b_log_probs, b_returns, b_advantages)
                    update_counter += 1
                    
                    # Log PPO update
                    logger.log_update(update_metrics)
                    
                    # Update ICM if enabled
                    if args.use_icm:
                        # Create pairs of consecutive observations for ICM update
                        for i in range(len(states) - 1):
                            icm_optimizer.zero_grad()
                            
                            s_t = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                            s_t1 = torch.FloatTensor(states[i+1]).unsqueeze(0).to(device)
                            a_t = torch.LongTensor([actions[i]]).to(device)
                            
                            icm_results = icm.forward(s_t, s_t1, a_t)
                            
                            # ICM loss is a combination of forward and inverse model losses
                            icm_loss = (
                                args.forward_loss_coef * icm_results['forward_loss'] + 
                                args.inverse_loss_coef * icm_results['inverse_loss']
                            )
                            
                            icm_loss.backward()
                            icm_optimizer.step()
                    
                    # Clear experience buffers
                    states = []
                    actions = []
                    log_probs = []
                    values = []
                    rewards = []
                    dones = []
                    
                    # Save checkpoint periodically
                    if time.time() - last_save_time > args.save_interval:
                        chkpt_path = os.path.join(args.log_dir, f"step_{num_steps}.pth")
                        save_checkpoint(model, chkpt_path, agent.optimizer, agent.scheduler, 
                                      metrics=metrics.metrics, update_count=update_counter)
                        
                        if args.use_icm:
                            icm_path = os.path.join(args.log_dir, f"step_{num_steps}_icm.pth")
                            torch.save({
                                'model_state_dict': icm.state_dict(),
                                'optimizer_state_dict': icm_optimizer.state_dict()
                            }, icm_path)
                        
                        last_save_time = time.time()
                    
                    # Plot metrics periodically
                    if time.time() - last_plot_time > args.plot_interval:
                        metrics.update_from_ppo(agent)
                        metrics.save(plot=True)
                        last_plot_time = time.time()
            
            # Episode complete, log stats
            episode_time = time.time() - episode_start_time
            episode_start_time = time.time()
            
            floor = info.get('current_floor', 0)
            if floor > max_floor_reached:
                max_floor_reached = floor
                floor_path = os.path.join(args.log_dir, f"floor_{floor}.pth")
                save_checkpoint(model, floor_path, agent.optimizer, agent.scheduler, 
                               metrics=metrics.metrics, update_count=update_counter)
                print(f"New max floor reached: {floor}! Saved checkpoint to {floor_path}")
                
                if args.use_icm:
                    icm_floor_path = os.path.join(args.log_dir, f"floor_{floor}_icm.pth")
                    torch.save({
                        'model_state_dict': icm.state_dict(),
                        'optimizer_state_dict': icm_optimizer.state_dict()
                    }, icm_floor_path)
                
                # Initialize tracking for the new floor
                if floor not in floor_completions:
                    floor_completions[floor] = []
                    consecutive_completions[floor] = 0
            
            # Curriculum learning - track floor mastery and adjust difficulty
            if args.curriculum:
                current_training_floor = args.starting_floor
                
                # Record whether the agent completed the current floor
                # Success means agent reached a higher floor than it started on
                success = (floor > current_training_floor)
                
                # Update floor completion tracking
                if current_training_floor not in floor_completions:
                    floor_completions[current_training_floor] = []
                
                # Add the result (success=1, failure=0)
                floor_completions[current_training_floor].append(1 if success else 0)
                
                # Keep only the most recent 10 attempts
                if len(floor_completions[current_training_floor]) > 10:
                    floor_completions[current_training_floor].pop(0)
                
                # Calculate consecutive completions
                if success:
                    consecutive_completions[current_training_floor] += 1
                else:
                    consecutive_completions[current_training_floor] = 0
                
                # Check if we should advance to the next floor
                if consecutive_completions[current_training_floor] >= args.mastery_threshold:
                    next_floor = current_training_floor + 1
                    
                    # Only advance if we haven't exceeded max_starting_floor
                    if next_floor <= args.max_starting_floor:
                        args.starting_floor = next_floor
                        consecutive_completions[current_training_floor] = 0  # Reset counter
                        print(f"CURRICULUM: Floor {current_training_floor} mastered with {args.mastery_threshold} consecutive completions!")
                        print(f"CURRICULUM: Starting floor increased to {args.starting_floor}")
                        
                        # Initialize tracking for the next floor if needed
                        if next_floor not in floor_completions:
                            floor_completions[next_floor] = []
                            consecutive_completions[next_floor] = 0
                
                # Log curriculum progress
                success_rate = np.mean(floor_completions[current_training_floor]) if floor_completions[current_training_floor] else 0
                print(f"CURRICULUM: Floor {current_training_floor} - Success rate: {success_rate:.2f}, Consecutive completions: {consecutive_completions[current_training_floor]}/{args.mastery_threshold}")
            
            # Log episode metrics
            episode_stats = {
                'episode_reward': total_extrinsic_rewards,
                'episode_length': episode_step_counter,
                'episode_time': episode_time,
                'episode_intrinsic_reward': total_intrinsic_rewards,
                'episode_floor': floor,
                'episode_fps': episode_step_counter / episode_time if episode_time > 0 else 0,
                'episode_keys': obstacle_tracker.keys_collected,
                'episode_doors': obstacle_tracker.doors_opened
            }
            
            metrics.update('episode_rewards', total_extrinsic_rewards)
            metrics.update('episode_lengths', episode_step_counter)
            metrics.update('episode_floors', floor)
            metrics.update('intrinsic_rewards', total_intrinsic_rewards)
            metrics.update('door_openings', obstacle_tracker.doors_opened)
            metrics.update('key_collections', obstacle_tracker.keys_collected)
            
            print(f"\nEpisode {episode} completed:")
            print(f"  Length: {episode_step_counter} steps")
            print(f"  Reward: {total_extrinsic_rewards:.2f} (extrinsic) + {total_intrinsic_rewards:.2f} (intrinsic)")
            print(f"  Floor: {floor}")
            print(f"  Keys collected: {obstacle_tracker.keys_collected}")
            print(f"  Doors opened: {obstacle_tracker.doors_opened}")
            print(f"  Time: {episode_time:.2f}s ({episode_stats['episode_fps']:.1f} FPS)")
            
            logger.log_episode_complete(episode_stats)
            
            # Reset episode counters
            total_extrinsic_rewards = 0
            total_intrinsic_rewards = 0
            episode_step_counter = 0
    
    except (KeyboardInterrupt, Exception) as e:
        print(f"Training interrupted: {e}")
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
    
    finally:
        # Save final checkpoint
        final_path = os.path.join(args.log_dir, f"final.pth")
        save_checkpoint(model, final_path, agent.optimizer, agent.scheduler, 
                       metrics=metrics.metrics, update_count=update_counter)
        
        if args.use_icm:
            final_icm_path = os.path.join(args.log_dir, f"final_icm.pth")
            torch.save({
                'model_state_dict': icm.state_dict(),
                'optimizer_state_dict': icm_optimizer.state_dict()
            }, final_icm_path)
        
        # Save metrics
        metrics.update_from_ppo(agent)
        metrics.save(plot=True)
        
        # Log training summary
        elapsed_time = time.time() - training_start_time
        logger.log_training_summary(metrics, elapsed_time, num_steps)
        
        # Close environment
        env.close()
        
        print(f"\nTraining completed. Ran for {num_steps} steps ({elapsed_time:.2f}s)")
        print(f"Max floor reached: {max_floor_reached}")
        print(f"Results saved to {args.log_dir}")
        
        return max_floor_reached


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO with enhanced exploration on Obstacle Tower')
    
    # Environment parameters
    parser.add_argument('--env_path', type=str, default='./ObstacleTower/obstacletower.x86_64',
                        help='Path to Obstacle Tower executable')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    parser.add_argument('--realtime_mode', action='store_true',
                        help='Run environment in realtime mode')
    
    # Training parameters
    parser.add_argument('--num_steps', type=int, default=5000000,
                        help='Total number of training steps')
    parser.add_argument('--horizon', type=int, default=2048,
                        help='Number of steps to collect before updating policy')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate for PPO')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clip epsilon')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='Number of PPO epochs per update')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Minibatch size for PPO updates')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value function loss coefficient')
    parser.add_argument('--entropy_reg', type=float, default=0.03,
                        help='Entropy regularization coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--target_kl', type=float, default=0.05,
                        help='Target KL divergence for early stopping')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=[None, 'linear', 'cosine', 'plateau'],
                        help='Learning rate scheduler')
    
    # Exploration parameters
    parser.add_argument('--use_icm', action='store_true', default=True,
                        help='Use Intrinsic Curiosity Module')
    parser.add_argument('--icm_lr', type=float, default=1e-4,
                        help='Learning rate for ICM')
    parser.add_argument('--forward_loss_coef', type=float, default=0.2,
                        help='Forward model loss coefficient for ICM')
    parser.add_argument('--inverse_loss_coef', type=float, default=0.8,
                        help='Inverse model loss coefficient for ICM')
    parser.add_argument('--intrinsic_coef', type=float, default=0.5,
                        help='Intrinsic reward coefficient')
    parser.add_argument('--extrinsic_coef', type=float, default=1.0,
                        help='Extrinsic reward coefficient')
    parser.add_argument('--exploration_coef', type=float, default=0.1,
                        help='Exploration bonus coefficient')
    
    # Reward shaping
    parser.add_argument('--key_reward', type=float, default=1.0,
                        help='Bonus reward for collecting a key')
    parser.add_argument('--door_reward', type=float, default=2.0,
                        help='Bonus reward for opening a door')
    parser.add_argument('--floor_reward', type=float, default=10.0,
                        help='Bonus reward for reaching a new floor')
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning')
    parser.add_argument('--starting_floor', type=int, default=0,
                        help='Starting floor for curriculum learning')
    parser.add_argument('--max_starting_floor', type=int, default=10,
                        help='Maximum starting floor for curriculum learning')
    parser.add_argument('--curriculum_interval', type=int, default=10,
                        help='Episodes between curriculum difficulty increases')
    parser.add_argument('--mastery_threshold', type=int, default=5,
                        help='Number of consecutive completions required to advance to the next floor')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--save_interval', type=int, default=900,
                        help='Seconds between checkpoint saves')
    parser.add_argument('--plot_interval', type=int, default=300,
                        help='Seconds between metrics plots')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    main(args) 