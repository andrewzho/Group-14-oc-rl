# Import the numpy patch first to fix np.bool deprecation
from src.np_patch import *

import gym
from obstacle_tower_env import ObstacleTowerEnv
import warnings
# Filter out the spammy gym_unity warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym_unity")

from src.model import PPONetwork, RecurrentPPONetwork
from src.ppo import PPO
from src.utils import normalize, save_checkpoint, load_checkpoint, ActionFlattener, MetricsTracker, TrainingLogger
import torch.nn.functional as F
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
import random
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import cv2

# New imports for TensorBoard
from torch.utils.tensorboard import SummaryWriter

# TensorBoard Logger
class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0
        self.episode_count = 0
        
    def log_scalar(self, tag, value, step=None):
        """Log a scalar value to TensorBoard"""
        if step is None:
            step = self.step_count
        self.writer.add_scalar(tag, value, step)
    
    def log_episode(self, reward, length, floor, max_floor, step_count=None):
        """Log episode metrics to TensorBoard"""
        if step_count is not None:
            self.step_count = step_count
        else:
            self.step_count += length
            
        self.episode_count += 1
        self.writer.add_scalar('episode/reward', reward, self.episode_count)
        self.writer.add_scalar('episode/length', length, self.episode_count)
        self.writer.add_scalar('episode/floor', floor, self.episode_count)
        self.writer.add_scalar('episode/max_floor', max_floor, self.episode_count)
        
    def log_update(self, metrics):
        """Log policy update metrics to TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'update/{key}', value, self.step_count)
            
    def log_image(self, tag, image, step=None):
        """Log an image to TensorBoard"""
        if step is None:
            step = self.step_count
            
        # Convert image to proper format if needed
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            if len(image.shape) == 3 and image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
                
            self.writer.add_image(tag, image, step, dataformats='HWC')
        
    def log_video(self, tag, frames, fps=30, step=None):
        """Log a video to TensorBoard"""
        if step is None:
            step = self.step_count
            
        # Convert frames to proper format
        video_frames = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.dtype == np.float32 and frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                    
                if len(frame.shape) == 3 and frame.shape[0] == 3:  # CHW format
                    frame = np.transpose(frame, (1, 2, 0))
                    
                video_frames.append(frame)
            
        # Convert to correct tensor format for TensorBoard
        if video_frames:
            video_tensor = np.stack(video_frames)
            self.writer.add_video(tag, video_tensor[None], step, fps=fps)
        
    def log_hyperparams(self, hyperparams):
        """Log hyperparameters"""
        # Convert hyperparams to proper format for TensorBoard
        hparam_dict = {}
        metric_dict = {}
        
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            else:
                # For non-primitive types, convert to string
                hparam_dict[key] = str(value)
        
        self.writer.add_hparams(hparam_dict, metric_dict)
        
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

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

# Add a new logging system that minimizes console output
class EnhancedLogger:
    """Enhanced logger with visualization capabilities."""
    
    def __init__(self, log_dir, console_level='INFO', file_level='DEBUG', 
                 console_freq=100, visualize_freq=1000, 
                 verbosity=1, log_intrinsic_rewards=False, intrinsic_log_freq=500):
        """Initialize logger with visualization capabilities."""
        self.log_dir = log_dir
        self.console_level = console_level
        self.file_level = file_level
        self.console_freq = console_freq
        self.visualize_freq = visualize_freq
        self.verbosity = verbosity  # Store verbosity level
        self.log_intrinsic_rewards = log_intrinsic_rewards
        self.intrinsic_log_freq = intrinsic_log_freq
        self.step_counter = 0  # Add step counter initialization
        
        # Create subdirectories
        self.metrics_dir = os.path.join(log_dir, 'metrics')
        self.figures_dir = os.path.join(log_dir, 'figures')
        self.checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        # Setup logging
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_{self.timestamp}.log')
        self.metrics_file = os.path.join(log_dir, f"metrics_{self.timestamp}.json")
        
        # Initialize metrics
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'floors': [],
            'max_floor': 0,
            'policy_losses': [],
            'value_losses': [],
            'entropy': [],
            'learning_rate': [],
            'steps_per_second': [],
            'total_steps': 0,
            'update_counts': 0,
            'floors_by_time': [],
            'key_collections': 0,
            'door_openings': 0
        }
        
        # Set up file logger
        self._setup_file_logger()
        
        # Print header to console
        print(f"=== OBSTACLE TOWER TRAINING ===")
        print(f"Logs and metrics saved to: {log_dir}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Full logs written to: {self.log_file}")
        if verbosity == 0:
            print(f"Verbosity level: {verbosity} (MINIMAL - Episode results always shown, debug info suppressed)")
        elif verbosity == 1:
            print(f"Verbosity level: {verbosity} (NORMAL - Episode results always shown, some debug info shown)")
        else:
            print(f"Verbosity level: {verbosity} (VERBOSE - All information shown)")
            
        print(f"Visualizations generated every {visualize_freq} episodes")
        if not log_intrinsic_rewards:
            print("Intrinsic reward logs disabled for console (still recorded in log file)")
        print(f"===============================")
    
    def _setup_file_logger(self):
        """Set up file logger for detailed logs."""
        # Clear any existing handlers
        logger = logging.getLogger('obstacle_tower')
        logger.setLevel(logging.DEBUG)  # Capture all logs
        
        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Configure file handler to capture all logs
        file_handler = logging.FileHandler(self.log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, self.file_level))
        logger.addHandler(file_handler)
        
        # Only add console handler if verbosity > 1 (we'll manage console output ourselves)
        if self.verbosity > 1:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(getattr(logging, self.console_level))
            logger.addHandler(console_handler)
        
        # This prevents messages from propagating to the root logger 
        # which can cause duplicate messages in some setups
        logger.propagate = False

    def log_episode(self, episode, reward, length, floor, max_floor, steps, steps_per_sec):
        """Log episode metrics and always print to console, regardless of settings."""
        # Always log to file
        logger = logging.getLogger('obstacle_tower')
        logger.debug(f"Episode {episode}: Reward={reward:.2f}, Length={length}, " +
                     f"Floor={floor}, MaxFloor={max_floor}, Steps={steps}, SPS={steps_per_sec:.2f}")
        
        # Update metrics
        self.metrics['episodes'].append(episode)
        self.metrics['rewards'].append(float(reward))
        self.metrics['lengths'].append(int(length))
        self.metrics['floors'].append(int(floor))
        self.metrics['max_floor'] = max(self.metrics['max_floor'], int(max_floor))
        self.metrics['total_steps'] = int(steps)  # Make sure we update the total_steps in metrics
        self.metrics['steps_per_second'].append(float(steps_per_sec))
        self.metrics['floors_by_time'].append((int(steps), int(max_floor)))
        
        # ALWAYS print to console for every episode, regardless of verbosity or console_freq
        # Calculate recent statistics for a more meaningful update
        recent_rewards = self.metrics['rewards'][-min(10, len(self.metrics['rewards'])):]
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        
        print(f"Episode {episode} | Reward: {reward:.2f} (Avg10: {avg_reward:.2f}) | " +
              f"Floor: {floor} | Max Floor: {max_floor} | " +
              f"Steps: {steps} | SPS: {steps_per_sec:.2f}")
        
        # Generate visualizations periodically
        if episode % self.visualize_freq == 0 and episode > 0:
            self.generate_visualizations()
            self.save_metrics()
    
    def log_update(self, update_num, metrics):
        """Log policy update metrics."""
        # Always log to file
        logger = logging.getLogger('obstacle_tower')
        logger.debug(f"Update {update_num}: " +
                     f"PolicyLoss={metrics.get('policy_loss', 0):.4f}, " +
                     f"ValueLoss={metrics.get('value_loss', 0):.4f}, " +
                     f"Entropy={metrics.get('entropy', 0):.4f}, " +
                     f"LR={metrics.get('learning_rate', 0):.6f}")
        
        # Update metrics
        self.metrics['update_counts'] = update_num
        
        if 'policy_loss' in metrics:
            self.metrics['policy_losses'].append(float(metrics['policy_loss']))
        
        if 'value_loss' in metrics:
            self.metrics['value_losses'].append(float(metrics['value_loss']))
            
        if 'entropy' in metrics:
            self.metrics['entropy'].append(float(metrics['entropy']))
            
        if 'learning_rate' in metrics:
            self.metrics['learning_rate'].append(float(metrics['learning_rate']))
        
        # Only print detailed update info if verbosity > 0
        if self.verbosity > 0 and update_num % max(1, 10//self.verbosity) == 0:
            print(f"Update {update_num} | "
                  f"Policy Loss: {metrics.get('policy_loss', 0):.4f} | "
                  f"Value Loss: {metrics.get('value_loss', 0):.4f} | "
                  f"Entropy: {metrics.get('entropy', 0):.4f}")
    
    def log_event(self, event_type, message):
        """Log significant events."""
        logger = logging.getLogger('obstacle_tower')
        
        # Always log to file, but avoid direct console output from the logger
        logger.info(f"[{event_type}] {message}")
        
        # Control console output based on event type and verbosity 
        if event_type in ['NEW_FLOOR', 'MILESTONE', 'ACHIEVEMENT']:
            # Always print important achievements
            if self.verbosity >= 0:  # Print at all verbosity levels
                print(f"🏆 {event_type}: {message}")
        elif event_type == 'INTRINSIC_REWARD':
            # Intrinsic rewards are extremely frequent, only log if enabled and at specified frequency
            self.step_counter += 1
            if self.log_intrinsic_rewards and self.step_counter % self.intrinsic_log_freq == 0:
                if self.verbosity > 1:  # Only with high verbosity
                    print(f"🧠 {message}")
        elif event_type in ['DOOR', 'KEY', 'KEY_COLLECTED'] and self.verbosity > 1:
            # Medium priority events
            print(f"🔑 {event_type}: {message}")
        elif event_type == 'ENV_INTERACTION':
            # Only print with highest verbosity
            if self.verbosity > 1:
                print(f"🎮 {message}")
        elif event_type == 'ERROR':
            # Always print errors
            print(f"❌ ERROR: {message}")
        elif event_type == 'EVAL':
            # Always print evaluation results
            print(f"📊 EVAL: {message}")
        elif event_type in ['TRAINING_SUMMARY', 'CHECKPOINT'] and self.verbosity > 0:
            # Print summaries with normal verbosity
            print(f"📈 {message}")
        elif self.verbosity > 1:
            # Print all other events only with highest verbosity
            print(f"{event_type}: {message}")
    
    def track_item_collection(self, item_type):
        """Track collection of keys and door openings."""
        if item_type.lower() == 'key':
            self.metrics['key_collections'] += 1
        elif item_type.lower() == 'door':
            self.metrics['door_openings'] += 1
    
    def generate_visualizations(self):
        """Generate visualizations of training progress."""
        if len(self.metrics['episodes']) < 2:
            return  # Not enough data to visualize
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Reward plot
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(self.metrics['episodes'], self.metrics['rewards'], 'b-')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        
        # Add moving average
        if len(self.metrics['rewards']) >= 10:
            N = 10  # Window size for moving average
            cumsum = np.cumsum(np.insert(self.metrics['rewards'], 0, 0)) 
            ma = (cumsum[N:] - cumsum[:-N]) / float(N)
            ax1.plot(self.metrics['episodes'][N-1:], ma, 'r-', linewidth=2, 
                    label='10-episode MA')
            ax1.legend()
        
        # 2. Floor progression
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(self.metrics['episodes'], self.metrics['floors'], 'g-')
        ax2.set_title('Floor Reached')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Floor')
        
        # 3. Episode length
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(self.metrics['episodes'], self.metrics['lengths'], 'm-')
        ax3.set_title('Episode Length')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        
        # 4. Policy loss
        if len(self.metrics['policy_losses']) > 0:
            ax4 = fig.add_subplot(2, 3, 4)
            ax4.plot(self.metrics['policy_losses'], 'r-')
            ax4.set_title('Policy Loss')
            ax4.set_xlabel('Update')
            ax4.set_ylabel('Loss')
        
        # 5. Entropy
        if len(self.metrics['entropy']) > 0:
            ax5 = fig.add_subplot(2, 3, 5)
            ax5.plot(self.metrics['entropy'], 'c-')
            ax5.set_title('Entropy')
            ax5.set_xlabel('Update')
            ax5.set_ylabel('Entropy')
        
        # 6. Floor progression by steps
        if len(self.metrics['floors_by_time']) > 0:
            ax6 = fig.add_subplot(2, 3, 6)
            steps, floors = zip(*self.metrics['floors_by_time'])
            ax6.plot(steps, floors, 'y-')
            ax6.set_title('Floor Progression by Steps')
            ax6.set_xlabel('Total Steps')
            ax6.set_ylabel('Max Floor')
        
        # Save figure
        plt.tight_layout()
        vis_path = os.path.join(self.figures_dir, f"training_vis_{self.timestamp}.png")
        plt.savefig(vis_path)
        plt.close()
    
    def save_metrics(self):
        """Save metrics to a JSON file for later analysis."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def debug(self, message):
        """Log debug message only at high verbosity."""
        if self.verbosity > 1:
            logger = logging.getLogger('obstacle_tower')
            logger.debug(message)
            if self.verbosity > 1:  # Only print to console at highest verbosity
                print(f"DEBUG: {message}")
    
    def info(self, message):
        """Log info message with medium verbosity."""
        logger = logging.getLogger('obstacle_tower')
        logger.info(message)
        if self.verbosity > 0:  # Print at normal and high verbosity
            print(f"INFO: {message}")
    
    def warning(self, message):
        """Log warning message."""
        logger = logging.getLogger('obstacle_tower')
        logger.warning(message)
        # Always print warnings
        print(f"⚠️ WARNING: {message}")
    
    def error(self, message):
        """Log error message."""
        logger = logging.getLogger('obstacle_tower')
        logger.error(message)
        # Always print errors
        print(f"❌ ERROR: {message}")
    
    def close(self):
        """Generate final visualizations and save metrics."""
        self.generate_visualizations()
        self.save_metrics()
        
        # Final summary
        print("\n=== TRAINING COMPLETE ===")
        print(f"Total episodes: {len(self.metrics['episodes'])}")
        print(f"Total steps: {self.metrics['total_steps']}")
        print(f"Max floor reached: {self.metrics['max_floor']}")
        print(f"Key collections: {self.metrics['key_collections']}")
        print(f"Door openings: {self.metrics['door_openings']}")
        print(f"Metrics saved to: {self.metrics_file}")
        print("=========================\n")

# Replace debug_print with a function that uses the logger
def debug_print(logger, *args, **kwargs):
    """Only print if verbosity level is high enough."""
    if logger and logger.verbosity > 1:
        message = " ".join(map(str, args))
        logger.debug(message)

# Simplified reward shaping function
def shape_reward(reward, info, action, prev_info=None, prev_keys=None, episodic_memory=None, current_floor=0):
    """Apply simplified reward shaping"""
    shaped_reward = reward
    
    # Basic structure for tracking reward components
    reward_components = {
        'base': reward,
        'floor_bonus': 0,
        'key_bonus': 0,
        'door_bonus': 0,
        'exploration_bonus': 0,
        'movement_bonus': 0
    }
    
    # 1. Floor completion bonus - the most important signal
    info_floor = info.get("current_floor", 0)
    if prev_info and info_floor > prev_info.get("current_floor", 0):
        floor_bonus = 5.0  # Significant bonus for reaching a new floor
        shaped_reward += floor_bonus
        reward_components['floor_bonus'] = floor_bonus
    
    # 2. Key collection bonus
    current_keys = info.get("total_keys", 0)
    if prev_keys is not None and current_keys > prev_keys:
        key_bonus = 1.0  # Substantial bonus for collecting keys
        shaped_reward += key_bonus
        reward_components['key_bonus'] = key_bonus
        
        # Track key location in memory
        if episodic_memory:
            position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
            episodic_memory.add_key_location(current_floor, position)
    
    # 3. Door opening bonus (key usage)
    if prev_keys is not None and current_keys < prev_keys:
        door_bonus = 2.0  # Significant bonus for using keys effectively
        shaped_reward += door_bonus
        reward_components['door_bonus'] = door_bonus
        
        # Track door location in memory
        if episodic_memory:
            position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
            episodic_memory.add_door_location(current_floor, position)
    
    # 4. Simple exploration bonus - avoid revisits
    if 'visit_count' in info:
        visit_count = info['visit_count']
        if visit_count == 0:  # First visit to this state
            exploration_bonus = 0.1
            shaped_reward += exploration_bonus
            reward_components['exploration_bonus'] = exploration_bonus
    
    # 5. Basic movement bonuses
    # Extract movement from action
    if isinstance(action, (list, tuple, np.ndarray)) and len(action) >= 3:
        move_idx, rot_idx, jump_idx = action[0], action[1], action[2]
        
        # Reward forward movement
        if move_idx == 1:  # Forward
            movement_bonus = 0.002  # Small positive reward
            shaped_reward += movement_bonus
            reward_components['movement_bonus'] = movement_bonus
    
    return shaped_reward, reward_components

def record_eval_video(model, env, device, action_flattener, max_steps=500, use_lstm=False):
    """Record a video of the agent's performance"""
    frames = []
    obs = env.reset()
    done = False
    steps = 0
    
    # Initialize frame stack
    frame_stack = deque(maxlen=4)
    obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
    for _ in range(4):
        frame_stack.append(obs)
    state = np.concatenate(frame_stack, axis=0)
    
    # Initialize LSTM state if needed
    lstm_state = None
    if use_lstm and hasattr(model, 'init_lstm_state'):
        lstm_state = model.init_lstm_state(batch_size=1, device=device)
    
    while not done and steps < max_steps:
        # Capture frame
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # Get action from policy
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if use_lstm:
                policy_logits, _, lstm_state = model(state_tensor, lstm_state)
            else:
                policy_logits, _ = model(state_tensor)
            
            action_idx = torch.argmax(policy_logits, dim=1).cpu().numpy()[0]
            action = action_flattener.lookup_action(action_idx)
            
        # Step environment
        next_obs, reward, done, info = env.step(action)
        
        # Update state
        next_obs = np.transpose(next_obs[0], (2, 0, 1)) / 255.0
        frame_stack.append(next_obs)
        state = np.concatenate(frame_stack, axis=0)
        
        steps += 1
    
    return frames

def main(args):
    """Main training function."""
    
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    logger = EnhancedLogger(
        log_dir=log_dir,
        console_freq=1 if args.verbosity >= 1 else args.console_log_freq,  # Always show episodes at verbosity 1+
        visualize_freq=args.visualize_freq,
        verbosity=args.verbosity,
        log_intrinsic_rewards=args.use_icm and args.verbosity >= 2  # Only log intrinsic rewards at verbosity 2+
    )
    
    # Setup TensorBoard logger
    tb_logger = TensorboardLogger(os.path.join(log_dir, 'tensorboard'))
    
    # Set environment variables for other modules
    os.environ['DEBUG'] = '1' if args.verbosity >= 2 else '0'
    os.environ['VERBOSITY'] = str(args.verbosity)  # Add this line to set VERBOSITY env var
    
    # Suppress TensorFlow warnings if low verbosity
    if args.verbosity < 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device for PyTorch
    device = torch.device(args.device)
    logger.log_event("SETUP", f"Using device: {device}")
    
    # Create environment
    try:
        env = create_obstacle_tower_env(
            executable_path=args.env_path,
            realtime_mode=args.realtime_mode,
            timeout=300,
            no_graphics=not args.graphics,  # Flip the graphics flag since our arg is positive but function takes negative
            config=None,
            worker_id=None
        )
        
        # Seed environment for reproducibility
        env.seed(args.seed)
        logger.log_event("SETUP", "Environment created successfully")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        return
    
    # Create ActionFlattener for MultiDiscrete action space
    if hasattr(env.action_space, 'n'):
        action_flattener = None
        action_dim = env.action_space.n
    else:
        action_flattener = ActionFlattener(env.action_space.nvec)
        action_dim = action_flattener.action_space.n
    
    logger.log_event("SETUP", f"Action space: {env.action_space}, Action dimension: {action_dim}")
    
    # Create PPO network - updated to support LSTM-based policies
    input_shape = (12, 84, 84)  # 4 stacked RGB frames (4 * 3 channels)
    
    if args.use_lstm:
        # Import RecurrentPPONetwork for LSTM-based policy
        from src.model import RecurrentPPONetwork
        model = RecurrentPPONetwork(
            input_shape=input_shape, 
            num_actions=action_dim,
            lstm_hidden_size=args.lstm_hidden_size,
            use_lstm=True
        ).to(device)
        logger.log_event("SETUP", f"Created recurrent model with LSTM hidden size {args.lstm_hidden_size}")
    else:
        # Use standard model without LSTM
        model = PPONetwork(input_shape=input_shape, num_actions=action_dim).to(device)
        logger.log_event("SETUP", f"Created standard model with input shape {input_shape}")
    
    # Initialize PPO agent with updated parameters
    ppo_agent = PPO(
        model=model,
        lr=args.lr if hasattr(args, 'lr') and args.lr is not None else 1e-4,  # Reduced learning rate for stability
        clip_eps=args.clip_eps if hasattr(args, 'clip_eps') and args.clip_eps is not None else 0.2,
        gamma=args.gamma if hasattr(args, 'gamma') and args.gamma is not None else 0.99,
        gae_lambda=args.gae_lambda if hasattr(args, 'gae_lambda') and args.gae_lambda is not None else 0.95,
        epochs=args.epochs if hasattr(args, 'epochs') and args.epochs is not None else 4,
        batch_size=args.batch_size if hasattr(args, 'batch_size') and args.batch_size is not None else 256,  # Larger batch size
        vf_coef=args.vf_coef if hasattr(args, 'vf_coef') and args.vf_coef is not None else 0.5,
        ent_reg=0.01,                 # More conservative entropy regularization
        max_grad_norm=0.5,           
        target_kl=0.025,              # Reduced target KL for more conservative updates
        lr_scheduler='linear',       
        adaptive_entropy=True,       
        min_entropy=0.005,            # Higher minimum entropy to ensure exploration
        entropy_decay_factor=0.9999,  # Slower entropy decay
        update_adv_batch_norm=True,  
        entropy_boost_threshold=0.001,
        lr_reset_interval=100,        # More frequent optimizer resets
        use_icm=args.use_icm,        
        icm_lr=5e-5,                  # Lower ICM learning rate
        icm_reward_scale=0.01,        # Reduced ICM reward scale for better balance
        icm_forward_weight=0.2,
        icm_inverse_weight=0.8,
        icm_feature_dim=256,
        device=device,
        use_recurrent=args.use_lstm,
        recurrent_seq_len=args.sequence_length if hasattr(args, 'sequence_length') and args.sequence_length is not None else 16,  # Longer sequences
    )
    
    # Log hyperparameters to TensorBoard
    hyperparams = {
        'lr': ppo_agent.lr,
        'gamma': ppo_agent.gamma,
        'gae_lambda': ppo_agent.gae_lambda,
        'clip_eps': ppo_agent.clip_eps,
        'epochs': ppo_agent.epochs,
        'batch_size': ppo_agent.batch_size,
        'ent_reg': ppo_agent.ent_reg,
        'use_lstm': args.use_lstm,
        'use_icm': args.use_icm,
        'seed': args.seed,
    }
    tb_logger.log_hyperparams(hyperparams)
    
    # Initialize ICM if enabled
    if args.use_icm:
        logger.log_event("SETUP", "Initializing Intrinsic Curiosity Module (ICM)")
        # We need to initialize ICM with 12 channels (4 stacked frames) input shape
        icm_input_shape = (12, 84, 84)  # Modified to use 12 channels
        ppo_agent.initialize_icm(icm_input_shape, action_dim)
        logger.log_event("SETUP", f"ICM initialized with input shape {icm_input_shape} and action dimension {action_dim}")
    
    # For backward compatibility, keep the old metric tracker but we'll primarily use enhanced_logger
    metrics_tracker = MetricsTracker(args.log_dir)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            model, optimizer_state, scheduler_state, saved_metrics, update_count = load_checkpoint(
                model, args.checkpoint, ppo_agent.optimizer, ppo_agent.scheduler
            )
            logger.log_event("CHECKPOINT", f"Loaded checkpoint from {args.checkpoint}")
            # Update optimizer and scheduler if states were loaded
            if optimizer_state:
                ppo_agent.optimizer.load_state_dict(optimizer_state)
            if scheduler_state and ppo_agent.scheduler:
                ppo_agent.scheduler.load_state_dict(scheduler_state)
            optimization_steps = update_count if update_count else 0
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            optimization_steps = 0
    else:
        optimization_steps = 0
    
    # Initialize variables for reward shaping
    last_position = None
    episode_floor_reached = 0
    episode_keys = 0
    
    # Initialize key/door memory
    if args.reward_shaping:
        if 'EnhancedKeyDoorMemory' in globals():
            key_door_memory = EnhancedKeyDoorMemory()
        else:
            key_door_memory = EpisodicMemory()
    else:
        key_door_memory = None
    
    # For storing episode rewards
    episode_rewards = []
    episode_lengths = []
    episode_floors = []
    
    # For tracking progress
    running_reward = 0
    best_reward = float('-inf')
    max_floor_reached = 0
    steps_without_progress = 0
    
    # Create episodic memory for tracking key and door locations
    episodic_memory = EpisodicMemory()
    
    # For intrinsic reward calculation
    prev_obs = None
    prev_has_key = False
    
    # Initialize curriculum learning variables if enabled
    curriculum_config = None
    if args.curriculum:
        # Initial curriculum settings
        curriculum_config = {
            'starting_floor': 0,
            'max_floor': 0,  # Start with just floor 0
            'difficulty': 0  # Start with easiest difficulty
        }
        # Tracking for curriculum adjustment
        curriculum_attempts = 0
        curriculum_successes = 0
        curriculum_success_streak = 0
        curriculum_floor_success_history = {}
        current_curriculum_floor = 0
        
        logger.log_event("CURRICULUM", f"Curriculum learning enabled with initial config: {curriculum_config}")

    # Add debugging info at the beginning of the main training loop
    # Main training loop
    start_time = time.time()
    steps_done = 0
    episode_count = 0
    current_floor = 0
    max_floor_reached = 0
    last_log_time = start_time
    last_save_time = start_time
    total_steps = 0  # Initialize total step counter ONCE, before defining report_progress
    
    # Initialize episode variables
    truncated_episodes = 0  # Keep track of episodes that were cut off
    max_episode_steps = 4000  # Maximum steps per episode to prevent getting stuck
    
    logger.log_event("TRAINING", f"Starting training loop with {args.num_steps} target steps")
    
    # Update the progress reporting to include debug info
    def report_progress():
        """Helper function to report current training progress."""
        nonlocal total_steps, steps_done  # Explicitly tell Python to use the outer scope's variables
        
        elapsed = time.time() - start_time
        fps = total_steps / max(1.0, elapsed)  # Avoid division by zero
        
        # Also check and report if steps_done and total_steps are different
        if steps_done != total_steps:
            logger.warning(f"Counter mismatch: steps_done={steps_done}, total_steps={total_steps}")
            # Fix the discrepancy (using max to ensure we don't lose progress)
            total_steps = max(steps_done, total_steps)
        
        # Add detailed step debugging for metrics
        logger.debug(
            f"Detailed metrics - episode_count: {episode_count}, "
            f"steps_done: {steps_done}, total_steps: {total_steps}, "
            f"max_floor: {max_floor_reached}"
        )
        
        # Build a detailed progress message
        if total_steps == 0:
            message = (
                f"Starting training - environment initialized\n"
                f"Total episodes: {episode_count}\n"
                f"Ready to begin collecting experience\n"
                f"Total setup time: {elapsed:.2f} seconds"
            )
        else:
            message = (
                f"Training progress at {elapsed:.2f} seconds:\n"
                f"Total episodes: {episode_count}\n"
                f"Total steps: {total_steps} (steps_done: {steps_done})\n"
                f"Max floor reached: {max_floor_reached}\n" 
                f"Total training time: {elapsed:.2f} seconds\n"
                f"Average steps per second: {fps:.2f}"
            )
            
        logger.log_event("TRAINING_SUMMARY", message)
    
    # Log initial state
    report_progress()
    
    # Initialize replay buffer
    replay_buffer = {
        'states': [],
        'next_states': [],
        'actions': [],
        'rewards': [],
        'log_probs': [],
        'values': [],
        'dones': []
    }
    max_replay_size = args.replay_buffer_size
    
    key_collections = 0
    door_openings = 0
    
    logger.log_event("TRAINING", "Starting training loop")
    
    last_eval_time = 0
    
    # LSTM state handling
    lstm_states = None
    
    # Setup metrics tracker - use the correct directory attribute
    metrics = MetricsTracker(log_dir=os.path.join(logger.log_dir, 'metrics'))
    
    # Store previous info for reward shaping
    prev_info = None
    
    while steps_done < args.num_steps:
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        episode_reward = 0
        episode_length = 0
        frame_stack = deque(maxlen=4)

        # Reset environment with current curriculum settings if enabled
        if args.curriculum and curriculum_config:
            try:
                # Update curriculum difficulty based on performance
                # Only update after an episode has completed (not at the very start)
                if episode_count > 0 and 'done' in locals() and done:
                    curriculum_attempts += 1
                    
                    # Track floor-specific success rate
                    floor_reached = info.get("current_floor", 0)
                    if floor_reached not in curriculum_floor_success_history:
                        curriculum_floor_success_history[floor_reached] = {
                            'attempts': 0,
                            'successes': 0
                        }
                    curriculum_floor_success_history[floor_reached]['attempts'] += 1
                    
                    # More aggressive floor advancement
                    # Check if agent reached or exceeded target floor
                    if floor_reached >= current_curriculum_floor:
                        curriculum_successes += 1
                        curriculum_success_streak += 1
                        
                        # Record as success for this floor
                        curriculum_floor_success_history[floor_reached]['successes'] += 1
                        
                        success_rate = curriculum_floor_success_history[floor_reached]['successes'] / \
                                       curriculum_floor_success_history[floor_reached]['attempts']
                        
                        # Make curriculum advancement more generous
                        # If success rate is good or we have a success streak, increase difficulty
                        if optimization_steps == 5:
                            logger.log_event("DIAGNOSTIC", "Testing policy after initial updates")
                            with torch.no_grad():
                                # Sample a few actions and print probabilities
                                test_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                                policy_logits, _ = model(test_state)
                                probs = F.softmax(policy_logits, dim=1)
                                logger.log_event("DIAGNOSTIC", f"Action probs: {probs.cpu().numpy()}")

                        if (success_rate > 0.5 and curriculum_floor_success_history[floor_reached]['attempts'] >= 3) or \
                           (curriculum_success_streak >= 2):
                            if curriculum_config['max_floor'] < 10:  # Cap at floor 10 for now
                                curriculum_config['max_floor'] += 1
                            current_curriculum_floor += 1
                            
                            # Log curriculum advancement
                            curriculum_msg = f"Advancing curriculum to floor {current_curriculum_floor}"
                            logger.log_event("CURRICULUM", curriculum_msg)
                            
                            # Boost entropy when curriculum advances to encourage exploration
                            old_entropy = ppo_agent.ent_reg
                            ppo_agent.ent_reg = min(0.1, ppo_agent.ent_reg * 1.5)
                            logger.log_event("ENTROPY", 
                                f"Boosting entropy from {old_entropy:.4f} to {ppo_agent.ent_reg:.4f} for new floor exploration")
                    else:
                        curriculum_success_streak = 0
                        
                    # If consistently failing at current floor, occasionally practice easier floors
                    if curriculum_attempts % 5 == 0 and curriculum_success_streak == 0:
                        # Temporarily reduce floor to build skills
                        practice_floor = max(0, current_curriculum_floor - 1)
                        logger.log_event("CURRICULUM", 
                            f"Temporarily practicing on floor {practice_floor} to build skills")
                        if hasattr(env, 'floor'):
                            env.floor(practice_floor)
                
                # Set floor based on curriculum
                # For simplicity, just use the current curriculum floor
                # A more sophisticated approach would sample from available floors
                if hasattr(env, 'floor'):
                    floor_to_use = min(current_curriculum_floor, curriculum_config['max_floor'])
                    env.floor(floor_to_use)
                    
            except Exception as e:
                error_msg = f"Error in curriculum logic: {e}"
                logger.log_event("ERROR", error_msg)
                # Continue without using curriculum for this episode
        
        # Reset the environment
        try:
            obs = env.reset()
        except Exception as e:
            error_msg = f"Error during environment reset: {e}"
            logger.log_event("ERROR", error_msg)
            
            # Try to recreate the environment
            try:
                env.close()
                env = create_obstacle_tower_env(
                    executable_path=args.env_path,
                    realtime_mode=False,
                    timeout=300
                )
                env.seed(args.seed)
                obs = env.reset()
            except Exception as e2:
                fatal_error_msg = f"Fatal error recreating environment: {e2}"
                logger.log_event("FATAL", fatal_error_msg)
                break  # Exit training loop

        # Initialize frame stack
        try:
            obs = env.reset()
        except UnityCommunicatorStoppedException as e:
            error_msg = f"Error during initial reset: {e}"
            logger.log_event("ERROR", error_msg)
            env.close()
            return
            
        obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
        for _ in range(4):
            frame_stack.append(obs)
        state = np.concatenate(frame_stack, axis=0)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        # Track key usage and door opening
        env._previous_keys = None
        env._previous_position = None
        
        # Reset previous info
        prev_info = None
        
        # Reset LSTM state at the beginning of each episode if using recurrent policy
        if args.use_lstm:
            ppo_agent.reset_lstm_state()
        
        # Collect trajectory
        steps_this_episode = 0
        trajectory_length = 1024  # Shorter trajectory length for more frequent updates
        
        # Track reward components for detailed logging
        reward_components = {
            'base': 0,
            'floor_bonus': 0,
            'key_bonus': 0,
            'door_bonus': 0,
            'exploration_bonus': 0,
            'movement_bonus': 0,
        }
        
        for step in range(trajectory_length):
            # Check if we've reached the maximum allowed steps for this episode
            if steps_this_episode >= max_episode_steps:
                logger.log_event("EPISODE", f"Episode truncated after reaching {max_episode_steps} steps")
                truncated_episodes += 1
                # Important: need to log the episode before setting done=True
                
                # Log the truncated episode just like we would a completed one
                episode_count += 1
                logger.debug(f"Episode {episode_count} truncated after {steps_this_episode} steps")
                
                # Log episode completion even though it was truncated
                logger.log_episode(
                    episode=episode_count,
                    reward=episode_reward,
                    length=episode_length,
                    floor=current_floor,
                    max_floor=max_floor_reached,
                    steps=total_steps,
                    steps_per_sec=total_steps / (time.time() - start_time)
                )
                
                # Log to TensorBoard
                tb_logger.log_episode(
                    reward=episode_reward,
                    length=episode_length,
                    floor=current_floor,
                    max_floor=max_floor_reached,
                    step_count=total_steps
                )
                
                # Now mark as done to exit the loop
                done = True
                break
                
            with torch.no_grad():
                # Use PPO agent's select_action method which handles LSTM states
                action_idx, log_prob, value, _ = ppo_agent.select_action(state)
                action = action_flattener.lookup_action(action_idx)
                action = np.array(action)
                
                # Store trajectory data
                states.append(state.copy())
                actions.append(action_idx)
                values.append(value)
                log_probs.append(log_prob)

            next_obs, reward, done, info = env.step(action)
            
            # Get current position from info
            current_position = [
                info.get("x_pos", 0),
                info.get("y_pos", 0),
                info.get("z_pos", 0)
            ]
            
            # Track highest floor reached
            info_floor = info.get("current_floor", 0)
            if info_floor > current_floor:
                current_floor = info_floor
                if current_floor > max_floor_reached:
                    max_floor_reached = current_floor
                    floor_msg = f"New floor reached: {current_floor}"
                    logger.log_event("NEW_FLOOR", floor_msg)
                    
                    # Add direct floor checkpoint saving here
                    try:
                        floor_checkpoint_path = os.path.join(log_dir, f"floor_{current_floor}.pth")
                        save_checkpoint(
                            model, 
                            floor_checkpoint_path,
                            optimizer=ppo_agent.optimizer,
                            scheduler=ppo_agent.scheduler,
                            metrics=None,  # Don't save metrics to keep file smaller
                            update_count=optimization_steps
                        )
                        # Replace direct print with logging call
                        logger.log_event("CHECKPOINT", f"Saved floor checkpoint to {floor_checkpoint_path}")
                    except Exception as e:
                        # Replace direct print with logging call
                        logger.log_event("ERROR", f"ERROR saving floor checkpoint: {e}")
                        traceback.print_exc()
            
            # Apply simplified reward shaping
            current_keys = info.get("total_keys", 0)
            previous_keys = env._previous_keys if hasattr(env, '_previous_keys') else None
            
            # Create shaped rewards using the new simplified function
            shaped_reward, reward_comps = shape_reward(
                reward, 
                info, 
                action, 
                prev_info=prev_info, 
                prev_keys=previous_keys, 
                episodic_memory=episodic_memory,
                current_floor=current_floor
            )
            
            # Store previous info for next step
            prev_info = info.copy()
            
            # Update reward components tracking
            for key, value in reward_comps.items():
                if key in reward_components:
                    reward_components[key] += value
            
            # Track door openings and key collections for metrics
            if previous_keys is not None and current_keys > previous_keys:
                key_collections += 1
                logger.track_item_collection('key')
            
            if previous_keys is not None and current_keys < previous_keys:
                door_openings += 1
                logger.track_item_collection('door')
            
            # Update key count tracker for next step
            env._previous_keys = current_keys
            
            # Apply intrinsic motivation if ICM is enabled
            if args.use_icm and ppo_agent.icm is not None:
                # Convert observation to tensor for ICM
                if prev_obs is not None:
                    # Process current and next observation using frame stacking
                    state_tensor = preprocess_observation_for_icm(prev_obs, frame_stack).to(device)
                    next_state_tensor = preprocess_observation_for_icm(next_obs, frame_stack).to(device)
                    
                    # Add batch dimension if needed
                    if len(state_tensor.shape) == 3:
                        state_tensor = state_tensor.unsqueeze(0)
                    if len(next_state_tensor.shape) == 3:
                        next_state_tensor = next_state_tensor.unsqueeze(0)
                    
                    # Get intrinsic reward from ICM
                    if isinstance(action_idx, (int, np.integer)):
                        action_tensor = torch.tensor([action_idx], device=device)
                    else:
                        action_tensor = torch.tensor(action_idx, device=device)
                        
                    intrinsic_reward = ppo_agent.get_intrinsic_reward(
                        state_tensor, next_state_tensor, action_tensor)
                    
                    # Scale and add intrinsic reward
                    if isinstance(intrinsic_reward, torch.Tensor):
                        intrinsic_reward = intrinsic_reward.item()
                    
                    # Add intrinsic reward to shaped reward
                    shaped_reward += intrinsic_reward
                    
                    # Log intrinsic reward
                    logger.log_event(
                        "INTRINSIC_REWARD",
                        f"Extrinsic: {shaped_reward - intrinsic_reward:.4f}, Intrinsic: {intrinsic_reward:.4f}"
                    )
            else:
                    # Skip ICM calculation for the first step when previous_obs is None
                    intrinsic_reward = 0.0

            # Store the current state for next step ICM calculation
            prev_obs = state.copy()

            # Store reward and done flag
            rewards.append(shaped_reward)
            dones.append(done)
            
            # Update episode tracking
            episode_reward += shaped_reward
            episode_length += 1
            steps_done += 1
            steps_this_episode += 1
            total_steps += 1
            
            # Log step counting every 1000 steps for debugging
            if total_steps % 1000 == 0:
                logger.debug(f"Step count update: total_steps={total_steps}, steps_done={steps_done}")

            # Process next observation for next step
            if isinstance(next_obs, tuple):
                next_obs_img = np.transpose(next_obs[0], (2, 0, 1)) / 255.0
            else:
                next_obs_img = np.transpose(next_obs, (2, 0, 1)) / 255.0
                
            frame_stack.append(next_obs_img)
            state = np.concatenate(list(frame_stack), axis=0)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Process done flag for LSTM state management
            if done:
                # Reset LSTM state when episode ends
                if args.use_lstm:
                    ppo_agent.process_done(done)
                
                # Log to TensorBoard periodically
                if total_steps % 1000 == 0:
                    # Capture a sample state for visualization
                    if args.verbosity > 1:
                        tb_logger.log_image('state/observation', next_obs[0], step=total_steps)
                
                # ... existing episode completion code ...
                break
        
        # If episode was cut off, we need to compute the value of the final state
        if not done:
            with torch.no_grad():
                policy_logits, next_value = model(state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0  # Terminal state has value 0
            
            # Increment episode counter when episode is complete
            episode_count += 1
            logger.debug(f"Episode {episode_count} completed after {episode_length} steps")
            
            # Log episode completion
            logger.log_episode(
                episode=episode_count,
                reward=episode_reward,
                length=episode_length,
                floor=current_floor,
                max_floor=max_floor_reached,
                steps=total_steps,
                steps_per_sec=total_steps / (time.time() - start_time)
            )
            
            # Log to TensorBoard
            tb_logger.log_episode(
                reward=episode_reward,
                length=episode_length,
                floor=current_floor,
                max_floor=max_floor_reached,
                step_count=total_steps
            )
            
            # Also log reward component breakdown
            for component, value in reward_components.items():
                tb_logger.log_scalar(f"reward_components/{component}", value, step=total_steps)
            
            # Reset for next episode
            try:
                obs = env.reset()
            except UnityCommunicatorStoppedException as e:
                error_msg = f"Error during reset after episode: {e}"
                logger.log_event("ERROR", error_msg)
                env.close()
                return
                
            obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
            frame_stack = deque(maxlen=4)
            for _ in range(4):
                frame_stack.append(obs)
            state = np.concatenate(frame_stack, axis=0)
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Reset key tracking
            env._previous_keys = None
            env._previous_position = None
            prev_info = None

        # Add current trajectory to experience replay if enabled
        if args.experience_replay:
            add_to_replay(replay_buffer, states, actions, rewards, log_probs, values, dones, max_replay_size, logger)
            
            # Log replay buffer size occasionally
            if steps_done % 10000 == 0:
                buffer_size = len(replay_buffer['states'])
                logger.log_event("REPLAY", f"Experience replay buffer size: {buffer_size}/{max_replay_size}")
        
        # Perform PPO update after collecting enough steps
        if len(replay_buffer['states']) >= ppo_agent.batch_size:
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
                logger.log_event("REPLAY", f"Replay buffer floor distribution: {floor_dist_str}")
            
            # Sample from replay buffer with special handling for recurrent policy
            if args.use_lstm:
                try:
                    # We need to include dones in the sample for proper sequence handling
                    states, actions, returns, advantages, old_log_probs, dones_sampled = sample_from_replay(
                        replay_buffer, ppo_agent.batch_size, include_dones=True, logger=logger
                    )
                    
                    # Process data for recurrent policy using the improved sequence handling
                    sequenced_data = ppo_agent.update_sequence_data(
                        states, actions, returns, values, old_log_probs, dones_sampled
                    )
                    
                    # Update policy with sequence data
                    metrics = ppo_agent.update(
                        sequenced_data, actions, old_log_probs, returns, advantages, dones=dones_sampled
                    )
                    optimization_steps += 1
                except Exception as e:
                    logger.error(f"LSTM update failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try to fall back to non-recurrent mode
                    try:
                        logger.log_event("RECOVERY", "Attempting fallback to non-recurrent update")
                        states, actions, returns, advantages, old_log_probs = sample_from_replay(
                            replay_buffer, ppo_agent.batch_size, logger=logger
                        )
                        metrics = ppo_agent.update(states, actions, old_log_probs, returns, advantages)
                        optimization_steps += 1
                    except Exception as e2:
                        logger.error(f"Fallback update also failed: {e2}")
                        # Skip this update to avoid terminating training
            else:
                # Standard update for non-recurrent policy
                try:
                    states, actions, returns, advantages, old_log_probs = sample_from_replay(
                        replay_buffer, ppo_agent.batch_size, logger=logger
                    )
                    metrics = ppo_agent.update(states, actions, old_log_probs, returns, advantages)
                    optimization_steps += 1
                except Exception as e:
                    logger.error(f"Update failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Skip this update
            
            # Update sequence length dynamically after successful updates
            if args.use_lstm and 'early_stopped' in metrics and metrics['early_stopped']:
                # If updates are stopping early, reduce sequence length
                original_seq_len = ppo_agent.recurrent_seq_len
                ppo_agent.recurrent_seq_len = max(2, ppo_agent.recurrent_seq_len // 2)
                if original_seq_len != ppo_agent.recurrent_seq_len:
                    logger.log_event("TUNING", 
                        f"Reduced sequence length from {original_seq_len} to {ppo_agent.recurrent_seq_len}")
                    
            # Log update regardless of whether it succeeded
            logger.log_update(optimization_steps, metrics if 'metrics' in locals() else {"error": "update_failed"})
            
            # Log to TensorBoard
            if 'metrics' in locals() and metrics:
                tb_logger.log_update(metrics)
            
            # Add diagnostic code to check learning
            if optimization_steps in [1, 5, 10, 20]:
                logger.log_event("DIAGNOSTIC", f"Testing policy after {optimization_steps} updates")
                with torch.no_grad():
                    test_state = torch.FloatTensor(state).unsqueeze(0).to(device)
                    policy_logits, value = model(test_state)
                    probs = F.softmax(policy_logits, dim=1)
                    prob_info = {f"action_{i}": float(p) for i, p in enumerate(probs[0].cpu().numpy()) if p > 0.01}
                    logger.log_event("DIAGNOSTIC", f"Action probs: {prob_info}")
                    logger.log_event("DIAGNOSTIC", f"Value estimate: {float(value.item())}")
                    
                    # Check if policy is becoming non-random
                    max_prob = float(probs.max().item())
                    if max_prob > 0.9:
                        logger.log_event("WARNING", f"Policy too deterministic (max prob: {max_prob})")
                    elif max_prob < 0.2:
                        logger.log_event("WARNING", f"Policy too random (max prob: {max_prob})")
                
            # Adaptive entropy regulation - increase entropy if stuck on a floor
            if current_floor <= 2 and optimization_steps % 50 == 0:
                # Check if we've been stuck on the same floor for a while
                if len(metrics_tracker.metrics.get('episode_floors', [])) > 10:
                    recent_floors = metrics_tracker.metrics['episode_floors'][-10:]
                    max_recent_floor = max(recent_floors) if recent_floors else 0
                    
                    if max_recent_floor <= 2:
                        # We're stuck on early floors - increase entropy temporarily
                        old_entropy = ppo_agent.ent_reg
                        ppo_agent.ent_reg = min(0.1, ppo_agent.ent_reg * 1.5)  # Increase entropy up to 0.1
                        logger.log_event("TUNING", f"Increasing entropy from {old_entropy:.4f} to {ppo_agent.ent_reg:.4f} to escape floor {max_recent_floor}")
                
            # Clear replay buffer after update
            replay_buffer = {
                'states': [], 'next_states': [], 'actions': [], 'rewards': [],
                'log_probs': [], 'values': [], 'dones': []
            }
        
        # Run evaluation periodically
        if steps_done - last_eval_time >= args.eval_interval:
            logger.log_event("EVAL", f"Starting evaluation at step {steps_done}")
            
            # Record and log video
            try:
                eval_frames = record_eval_video(
                    model, 
                    env, 
                    device, 
                    action_flattener, 
                    max_steps=500, 
                    use_lstm=args.use_lstm
                )
                
                # Log the video to TensorBoard
                if eval_frames:
                    tb_logger.log_video('evaluation/video', eval_frames, step=steps_done)
                    logger.log_event("EVAL", f"Recorded and saved evaluation video at step {steps_done}")
            except Exception as e:
                logger.error(f"Error recording evaluation video: {e}")
            
            # Save a checkpoint after evaluation
            checkpoint_path = os.path.join(log_dir, f"eval_step_{steps_done}.pth")
            save_checkpoint(
                model, 
                checkpoint_path, 
                optimizer=ppo_agent.optimizer, 
                scheduler=ppo_agent.scheduler, 
                metrics=metrics_tracker.metrics,
                update_count=optimization_steps
            )
            
            logger.log_event("CHECKPOINT", f"Saved post-evaluation checkpoint at step {steps_done}")
            last_eval_time = steps_done
            
        # Save metrics and checkpoints periodically
        current_time = time.time()
        
        # Log training summary every minute
        if current_time - last_log_time > 60:  # Every minute
            logger.log_event("TRAINING_SUMMARY", f"Training summary at {current_time - start_time:.2f} seconds")
            logger.log_event("TRAINING_SUMMARY", f"Total episodes: {episode_count}")
            logger.log_event("TRAINING_SUMMARY", f"Total steps: {steps_done}")
            logger.log_event("TRAINING_SUMMARY", f"Max floor reached: {max_floor_reached}")
            logger.log_event("TRAINING_SUMMARY", f"Total training time: {current_time - start_time:.2f} seconds")
            logger.log_event("TRAINING_SUMMARY", f"Average steps per second: {steps_done / (current_time - start_time):.2f}")
            last_log_time = current_time
        
        # Save checkpoints and metrics every 5 minutes
        if current_time - last_save_time > 300:  # Every 5 minutes
            # Update metrics tracker with PPO metrics
            metrics_tracker.update_from_ppo(ppo_agent)
            
            # Save metrics without plotting
            metrics_tracker.save(plot=False)
            
            # Save checkpoint
            checkpoint_path = os.path.join(log_dir, f"step_{steps_done}.pth")
            save_checkpoint(
                model, 
                checkpoint_path, 
                optimizer=ppo_agent.optimizer, 
                scheduler=ppo_agent.scheduler, 
                metrics=metrics_tracker.metrics,
                update_count=optimization_steps
            )
            
            logger.log_event("CHECKPOINT", f"Saved checkpoint at step {steps_done}")
            last_save_time = current_time

        # Reset optimizer periodically
        if optimization_steps % 200 == 0 or ppo_agent.optimizer.param_groups[0]['lr'] < 5e-5:
            ppo_agent.reset_optimizer_state()
            logger.log_event("OPTIMIZATION", "Reset optimizer state")

        # Periodically update ICM with recent experience
        if args.use_icm and ppo_agent.icm is not None and len(replay_buffer['states']) >= args.batch_size and step % 10 == 0:
            try:
                # Sample batch from buffer
                indices = np.random.choice(len(replay_buffer['states']), 
                                          size=min(args.batch_size, len(replay_buffer['states'])),
                                          replace=False)
                
                # Convert states to proper format for ICM
                # Note: replay_buffer['states'] already contains stacked frames (12 channels)
                # so we can just convert them to tensors
                icm_states = torch.stack([
                    torch.as_tensor(replay_buffer['states'][i], dtype=torch.float32)
                    for i in indices
                ]).to(device)
                
                icm_actions = torch.tensor([replay_buffer['actions'][i] for i in indices], 
                                          dtype=torch.long).to(device)
                
                # Get next states, either from buffer or compute them
                if 'next_states' in replay_buffer and all(i < len(replay_buffer['next_states']) for i in indices):
                    icm_next_states = torch.stack([
                        torch.as_tensor(replay_buffer['next_states'][i], dtype=torch.float32)
                        for i in indices
                    ]).to(device)
                else:
                    # Fallback if next_states not available
                    icm_next_states = []
                    for i in indices:
                        if i + 1 < len(replay_buffer['states']) and not replay_buffer['dones'][i]:
                            icm_next_states.append(torch.as_tensor(
                                replay_buffer['states'][i + 1], dtype=torch.float32))
                        else:
                            # For terminal states or last state, use zeros
                            icm_next_states.append(torch.zeros_like(
                                torch.as_tensor(replay_buffer['states'][i], dtype=torch.float32)))
                            
                    icm_next_states = torch.stack(icm_next_states).to(device)
                
                # Make sure the tensors have the right shape for ICM
                # ICM expects [batch_size, 12, 84, 84]
                if icm_states.shape[1] != 12:
                    # Replace direct print with logging call
                    logger.log_event("ICM", f"Reshaping ICM states from {icm_states.shape} to match 12-channel requirement")
                    # Repeat each 3-channel state 4 times to get 12 channels
                    if icm_states.shape[1] == 3:
                        icm_states = torch.cat([icm_states] * 4, dim=1)
                        
                if icm_next_states.shape[1] != 12:
                    # Replace direct print with logging call
                    logger.log_event("ICM", f"Reshaping ICM next_states from {icm_next_states.shape} to match 12-channel requirement")
                    # Repeat each 3-channel state 4 times to get 12 channels
                    if icm_next_states.shape[1] == 3:
                        icm_next_states = torch.cat([icm_next_states] * 4, dim=1)
                
                # Update ICM
                icm_metrics = ppo_agent.update_icm(icm_states, icm_actions, icm_next_states)
                
                # Log ICM metrics to TensorBoard
                if icm_metrics and step % 100 == 0:
                    for key, value in icm_metrics.items():
                        metrics_tracker.update(f"icm/{key}", value)
                        tb_logger.log_scalar(f"icm/{key}", value, step=total_steps)
            except Exception as e:
                # Log error but don't crash training
                # Replace direct print with logging call
                logger.log_event("ERROR", f"Error updating ICM: {e}")

        # Periodically report training progress
        current_time = time.time()
        time_since_last_report = current_time - last_log_time
        
        # Log training summary every minute
        if time_since_last_report > 60:  # Every minute
            report_progress()  # Use our helper function for consistent reporting
            last_log_time = current_time

    # Final cleanup and logging
    logger.log_event("TRAINING", "Training loop completed")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(log_dir, "final_model.pth")
    save_checkpoint(
        model, 
        final_checkpoint_path, 
        optimizer=ppo_agent.optimizer, 
        scheduler=ppo_agent.scheduler, 
        metrics=metrics_tracker.metrics,
        update_count=optimization_steps
    )
    logger.log_event("CHECKPOINT", f"Saved final model to {final_checkpoint_path}")
    
    # Close loggers
    logger.close()
    tb_logger.close()
    
    # Close environment
    env.close()

    return model, metrics_tracker, logger.metrics

def add_to_replay(replay_buffer, states, actions, rewards, log_probs, values, dones, max_replay_size, logger=None):
    """Add trajectory to the replay buffer."""
    # Add all the standard entries
    replay_buffer['states'].extend(states)
    replay_buffer['actions'].extend(actions)
    replay_buffer['rewards'].extend(rewards)
    replay_buffer['log_probs'].extend(log_probs)
    replay_buffer['values'].extend(values)
    replay_buffer['dones'].extend(dones)
    
    # Compute and add next_states
    next_states = []
    for i in range(len(states)):
        if i + 1 < len(states) and not dones[i]:
            next_states.append(states[i+1])
        else:
            # For terminal states or the last state, use a copy of the current state
            # This is a placeholder since these states won't actually be used for learning
            # (due to the 'done' flag)
            next_states.append(np.zeros_like(states[i]))
    
    replay_buffer['next_states'].extend(next_states)
    
    # Ensure buffer doesn't exceed max size
    if len(replay_buffer['states']) > max_replay_size:
        excess = len(replay_buffer['states']) - max_replay_size
        for key in replay_buffer:
            replay_buffer[key] = replay_buffer[key][excess:]
        
        if logger and logger.verbosity > 1:
            logger.debug(f"Replay buffer trimmed, removed {excess} old entries")

def sample_from_replay(replay_buffer, batch_size, include_dones=False, logger=None):
    """Sample a batch of experiences from the replay buffer."""
    buffer_size = len(replay_buffer['states'])
    if buffer_size == 0:
        if include_dones:
            return [], [], [], [], [], []
        else:
            return [], [], [], [], []
    
    # Sample indices with priority for more recent experiences
    indices = np.random.choice(buffer_size, size=min(buffer_size, batch_size), replace=False)
    
    # Extract data
    states = [replay_buffer['states'][i] for i in indices]
    actions = [replay_buffer['actions'][i] for i in indices]
    rewards = [replay_buffer['rewards'][i] for i in indices]
    values = [replay_buffer['values'][i] for i in indices]
    old_log_probs = [replay_buffer['log_probs'][i] for i in indices]
    
    # Include dones if requested (for recurrent policy)
    if include_dones:
        dones = [replay_buffer['dones'][i] for i in indices]
    
    # Simple returns and advantages calculation
    returns = rewards.copy()  # Simple estimate: just use the reward
    advantages = [rewards[i] - values[i] for i in range(len(rewards))]  # Simple advantage
    
    # Normalize advantages for numerical stability
    if len(advantages) > 1:
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = [(advantages[i] - adv_mean) / adv_std for i in range(len(advantages))]
    
    # Ensure we specify dtypes to avoid object arrays
    if include_dones:
        try:
            return (np.array(states, dtype=np.float32), 
                    np.array(actions, dtype=np.int64), 
                    np.array(returns, dtype=np.float32), 
                    np.array(advantages, dtype=np.float32), 
                    np.array(old_log_probs, dtype=np.float32), 
                    np.array(dones, dtype=np.float32))
        except Exception as e:
            if logger:
                logger.debug(f"Error in sample_from_replay: {e}")
            # Fall back to lists if numpy conversion fails
            return states, actions, returns, advantages, old_log_probs, dones
    else:
        try:
            return (np.array(states, dtype=np.float32), 
                    np.array(actions, dtype=np.int64), 
                    np.array(returns, dtype=np.float32), 
                    np.array(advantages, dtype=np.float32), 
                    np.array(old_log_probs, dtype=np.float32))
        except Exception as e:
            if logger:
                logger.debug(f"Error in sample_from_replay: {e}")
            # Fall back to lists if numpy conversion fails
            return states, actions, returns, advantages, old_log_probs

def preprocess_observation_for_icm(observation, frame_stack):
    """
    Preprocess observation for ICM (Intrinsic Curiosity Module).
    
    Args:
        observation: Raw observation from environment
        frame_stack: List of previous observations for frame stacking
        
    Returns:
        Preprocessed observation with frame stacking as PyTorch tensor
    """
    # Process the current observation
    try:
        if isinstance(observation, tuple):
            obs_img = observation[0]
        else:
            obs_img = observation
            
        # Convert to channels-first format if needed
        if obs_img.shape[-1] == 3:  # [H, W, C] format
            processed_frame = np.transpose(obs_img, (2, 0, 1))
        else:
            processed_frame = obs_img
            
        # Normalize to [0, 1]
        if processed_frame.max() > 1.0:
            processed_frame = processed_frame / 255.0
            
        # Resize to 84x84 if needed
        if processed_frame.shape[1:] != (84, 84):
            # Save original format for reshaping
            channels = processed_frame.shape[0]
            # Reshape to HWC for cv2
            reshaped = np.transpose(processed_frame, (1, 2, 0))
            resized = cv2.resize(reshaped, (84, 84))
            # Back to CHW
            processed_frame = np.transpose(resized, (2, 0, 1))
    except Exception as e:
        print(f"Error in preprocess_observation_for_icm: {e}")
        # Emergency fallback - create a blank frame
        processed_frame = np.zeros((3, 84, 84), dtype=np.float32)
    
    # If frame_stack is empty or None, create a new stack with the current frame repeated
    if not frame_stack:
        stacked_frames = np.concatenate([processed_frame] * 4, axis=0)  # Repeat the same frame 4 times
        return torch.FloatTensor(stacked_frames)  # Convert to torch tensor
    
    # Otherwise, add the new frame to the stack and return the concatenated result
    frame_stack.append(processed_frame)
    if len(frame_stack) > 4:
        frame_stack.pop(0)  # Remove oldest frame if we have more than 4
        
    # Stack the frames along the channel dimension
    while len(frame_stack) < 4:
        frame_stack.append(processed_frame)  # Pad with copies if not enough frames
        
    # Verify the shapes before concatenating
    for i, frame in enumerate(frame_stack):
        if frame.shape != (3, 84, 84):
            print(f"Warning: Frame {i} has unexpected shape {frame.shape}, reshaping...")
            # Try to reshape if possible
            if frame.size == 3*84*84:
                frame_stack[i] = frame.reshape(3, 84, 84)
            else:
                # Replace with zeros if shape is wrong
                frame_stack[i] = np.zeros((3, 84, 84), dtype=np.float32)
    
    # Concatenate the frames
    try:
        stacked_frames = np.concatenate(frame_stack, axis=0)
        # Verify the final shape
        assert stacked_frames.shape == (12, 84, 84), f"Stacked frames have shape {stacked_frames.shape}, expected (12, 84, 84)"
        # Convert to PyTorch tensor
        return torch.FloatTensor(stacked_frames)
    except Exception as e:
        print(f"Error concatenating frames: {e}")
        # Emergency fallback - create a blank stacked frame
        stacked_frames = np.zeros((12, 84, 84), dtype=np.float32)
        return torch.FloatTensor(stacked_frames)

if __name__ == "__main__":
    import argparse
    import datetime
    import traceback
    
    parser = argparse.ArgumentParser(description='Train PPO agent on Obstacle Tower')
    
    # Environment settings
    parser.add_argument('--env_path', type=str, default="./ObstacleTower/obstacletower.x86_64",
                        help='Path to Obstacle Tower executable')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--num_steps', type=int, default=1000000,
                        help='Number of training steps')
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning to gradually increase difficulty')
    
    # PPO hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,  # Updated default learning rate
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--epochs', type=int, default=4,  # Increased from 3 to 4
                        help='Number of PPO epochs')
    parser.add_argument('--batch_size', type=int, default=256,  # Increased from 128 to 256
                        help='PPO batch size')
    parser.add_argument('--entropy_reg', type=float, default=0.01,  # Reduced from 0.02 to 0.01
                        help='Entropy regularization coefficient to encourage exploration')
    parser.add_argument('--vf_coef', type=float, default=0.5, 
                        help='Value function loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--target_kl', type=float, default=0.025,  # Updated from 0.01 to 0.025
                        help='Target KL divergence for early stopping')
                        
    # ICM parameters
    parser.add_argument('--use_icm', action='store_true',
                        help='Use Intrinsic Curiosity Module for exploration')
    parser.add_argument('--icm_reward_scale', type=float, default=0.01,  # Reduced from 0.05 to 0.01
                        help='Scale factor for intrinsic rewards from ICM')
    parser.add_argument('--icm_forward_weight', type=float, default=0.2,
                        help='Weight for ICM forward model loss')
    parser.add_argument('--icm_inverse_weight', type=float, default=0.8,
                        help='Weight for ICM inverse model loss')
    parser.add_argument('--icm_feature_dim', type=int, default=256,
                        help='Dimension of ICM feature representation')
    
    # Training settings
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to load agent checkpoint')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoints every N episodes')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log every N episodes')
    parser.add_argument('--update_interval', type=int, default=2048,
                        help='Update policy every N steps')
    parser.add_argument('--min_update_samples', type=int, default=1024,
                        help='Minimum samples before updating policy')
    parser.add_argument('--replay_buffer_size', type=int, default=10000,
                        help='Size of replay buffer for off-policy learning')
    parser.add_argument('--eval_interval', type=int, default=20000,
                        help='Evaluate policy every N steps')
    parser.add_argument('--eval_episodes', type=int, default=5,
                        help='Number of episodes for evaluation')
    parser.add_argument('--experience_replay', action='store_true',
                        help='Use experience replay for off-policy learning')
    parser.add_argument('--log_dir', type=str, default=f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help='Directory for logs and checkpoints')
    parser.add_argument('--reward_shaping', action='store_true',
                        help='Use reward shaping for better learning')
    parser.add_argument('--use_episodic_memory', action='store_true',
                        help='Use episodic memory for tracking key-door interactions')
    
    # Add verbosity controls
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity level: 0=minimal output, 1=show ALL episode results without debug info, 2=verbose with debug info')
    parser.add_argument('--console_log_freq', type=int, default=100,
                        help='How often to print episode results to console (every N episodes)')
    parser.add_argument('--visualize_freq', type=int, default=1000,
                        help='Generate visualizations every N episodes')
    parser.add_argument('--log_intrinsic_rewards', action='store_true',
                        help='Log intrinsic rewards to console (very verbose)')
    parser.add_argument('--intrinsic_log_freq', type=int, default=500,
                        help='Log intrinsic rewards every N steps (if enabled)')
    
    # Add LSTM-related arguments
    parser.add_argument('--use_lstm', action='store_true',
                        help='Use LSTM-based recurrent policy for better memory')
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help='Hidden size of LSTM layer')
    parser.add_argument('--sequence_length', type=int, default=16,  # Increased from 8 to 16
                        help='Sequence length for LSTM training')
    
    # Add realtime mode and graphics options
    parser.add_argument('--realtime_mode', action='store_true',
                        help='Run environment in realtime mode to watch training')
    parser.add_argument('--graphics', action='store_true',
                        help='Enable graphics (disable no_graphics) to visualize training')
    
    args = parser.parse_args()
    
    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup a simple logger for startup messages before main logger is created
    startup_logger = logging.getLogger('startup')
    startup_logger.setLevel(logging.INFO)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    startup_logger.addHandler(console_handler)
    
    try:
        # Create enhanced logger
        logger = EnhancedLogger(
            log_dir=log_dir,
            console_freq=1 if args.verbosity >= 1 else args.console_log_freq,
            visualize_freq=args.visualize_freq,
            verbosity=args.verbosity,
            log_intrinsic_rewards=args.use_icm and args.verbosity >= 2
        )
        
        # Create TensorBoard logger
        tb_logger = TensorboardLogger(os.path.join(log_dir, 'tensorboard'))
        
        logger.log_event("STARTUP", "Starting Obstacle Tower training")
        
        # Run main training function
        model, metrics_tracker, final_metrics = main(args)
        
        # Log completion
        logger.log_event("COMPLETION", "Training completed successfully")
        
        # Final cleanup
        logger.close()
        tb_logger.close()
        
    except KeyboardInterrupt:
        startup_logger.warning("Training interrupted by user")
        if 'logger' in locals():
            logger.warning("Training interrupted by user")
            logger.close()
            
        if 'tb_logger' in locals():
            tb_logger.close()
            
    except Exception as e:
        error_msg = f"Error: {e}"
        traceback_str = traceback.format_exc()
        
        # Log to startup logger if enhanced logger not created yet
        if 'logger' not in locals():
            startup_logger.error(error_msg)
            startup_logger.error(traceback_str)
            
            # Create a basic log file for the error
            error_log_path = os.path.join(log_dir, "error.log")
            with open(error_log_path, 'w') as f:
                f.write(f"{error_msg}\n\n{traceback_str}")
            print(f"Error details written to {error_log_path}")
        else:
            # Use enhanced logger if available
            logger.error(error_msg)
            logger.error(traceback_str)
            logger.close()
            
        if 'tb_logger' in locals():
            tb_logger.close()