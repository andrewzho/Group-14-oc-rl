import gym
from obstacle_tower_env import ObstacleTowerEnv
from src.model import PPONetwork, RecurrentPPONetwork
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
import random
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation

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
    """
    Enhanced logging system that minimizes console output while keeping
    detailed logs in files that can be analyzed later.
    """
    def __init__(self, log_dir, console_level='INFO', file_level='DEBUG', 
                 console_freq=100, visualize_freq=1000, 
                 verbosity=1, log_intrinsic_rewards=False, intrinsic_log_freq=500):
        """
        Initialize enhanced logger.
    
    Args:
            log_dir: Directory to save logs and visualizations
            console_level: Minimum level to print to console
            file_level: Minimum level to write to log file
            console_freq: Controls visualization frequency, episode results are always printed
            visualize_freq: Generate visualizations every N episodes
            verbosity: Control amount of debug console output (0=minimal, 1=normal, 2=verbose)
                       Note: Episode results are ALWAYS printed regardless of this setting
            log_intrinsic_rewards: Whether to log intrinsic rewards to console
            intrinsic_log_freq: Only log intrinsic rewards every N steps
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamp for this run
        self.timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"training_{self.timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"metrics_{self.timestamp}.json")
        
        # Configure logging levels
        self.console_level = console_level
        self.file_level = file_level
        
        # Configure frequency
        self.console_freq = console_freq
        self.visualize_freq = visualize_freq
        
        # Configure verbosity
        self.verbosity = verbosity
        self.log_intrinsic_rewards = log_intrinsic_rewards
        self.intrinsic_log_freq = intrinsic_log_freq
        self.step_counter = 0
        
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
        
        # Log header to console
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
        self.metrics['total_steps'] = int(steps)
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
        # Only print to console if appropriate based on verbosity and event type
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
        elif event_type in ['DOOR', 'KEY', 'KEY_COLLECTED'] and self.verbosity > 0:
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
        vis_path = os.path.join(self.log_dir, f"training_vis_{self.timestamp}.png")
        plt.savefig(vis_path)
        plt.close()
    
    def save_metrics(self):
        """Save metrics to a JSON file for later analysis."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
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

# Add a debug flag to control print statements
DEBUG_MODE = os.environ.get('DEBUG', '0') == '1'  # Set DEBUG=1 in environment to enable debug output

def debug_print(*args, **kwargs):
    """Print only when debug mode is enabled."""
    if DEBUG_MODE:
        print(*args, **kwargs)

def main(args):
    """Main training function."""
    global DEBUG_MODE
    
    # Set debug mode based on verbosity
    DEBUG_MODE = args.verbosity >= 2  # Only enable debug prints if verbosity is at least 2
    
    # Set environment variable for other modules
    os.environ['DEBUG'] = '1' if DEBUG_MODE else '0'
    
    # Suppress TensorFlow warnings if low verbosity
    if args.verbosity < 2:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
        
    # Create log directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    logger = EnhancedLogger(
        log_dir=log_dir,
        console_freq=args.console_log_freq,
        visualize_freq=args.visualize_freq,
        verbosity=args.verbosity,
        log_intrinsic_rewards=args.use_icm
    )
    
    # Log start message according to verbosity
    if args.verbosity >= 1:
        print(f"Starting training with verbosity={args.verbosity}")
        print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
        print(f"Logs will be saved to {log_dir}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set device for PyTorch
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create environment
    env = create_obstacle_tower_env(
        executable_path=args.env_path,
        realtime_mode=False,
        timeout=300
    )
    
    # Seed environment for reproducibility
    env.seed(args.seed)
    
    # Create ActionFlattener for MultiDiscrete action space
    if hasattr(env.action_space, 'n'):
        action_flattener = None
        action_dim = env.action_space.n
    else:
        action_flattener = ActionFlattener(env.action_space.nvec)
        action_dim = action_flattener.action_space.n
    
    print(f"Action space: {env.action_space}, Action dimension: {action_dim}")
    
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
        print(f"Created recurrent model with LSTM hidden size {args.lstm_hidden_size}")
    else:
        # Use standard model without LSTM
        model = PPONetwork(input_shape=input_shape, num_actions=action_dim).to(device)
    
    print(f"Created model with input shape {input_shape} and {action_dim} actions")
    
    # Initialize PPO agent with updated parameters for LSTM if needed
    ppo_agent = PPO(
        model=model,
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        epochs=args.epochs,
        batch_size=args.batch_size,
        vf_coef=args.vf_coef,
        ent_reg=0.03,                # Increased entropy regularization for better exploration
        max_grad_norm=0.5,           
        target_kl=0.05,              
        lr_scheduler='linear',       
        adaptive_entropy=True,       
        min_entropy=0.02,            # Higher minimum entropy to ensure exploration
        entropy_decay_factor=0.9995, # Slower entropy decay
        update_adv_batch_norm=True,  
        entropy_boost_threshold=0.001,
        lr_reset_interval=50,        
        use_icm=args.use_icm,        
        icm_lr=1e-4,                 
        icm_reward_scale=0.02,       # Higher ICM reward scale to encourage exploration
        icm_forward_weight=0.2,
        icm_inverse_weight=0.8,
        icm_feature_dim=256,
        device=device,
        use_recurrent=args.use_lstm,
        recurrent_seq_len=args.sequence_length,
    )
    
    # Initialize ICM if enabled
    if args.use_icm:
        print("Initializing Intrinsic Curiosity Module (ICM)")
        # We need to initialize ICM with 12 channels (4 stacked frames) input shape
        icm_input_shape = (12, 84, 84)  # Modified to use 12 channels
        ppo_agent.initialize_icm(icm_input_shape, action_dim)
        print(f"ICM initialized with input shape {icm_input_shape} and action dimension {action_dim}")
    
    # For backward compatibility, keep the old metric tracker but we'll primarily use enhanced_logger
    metrics_tracker = MetricsTracker(args.log_dir)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            model, optimizer_state, scheduler_state, saved_metrics, update_count = load_checkpoint(
                model, args.checkpoint, ppo_agent.optimizer, ppo_agent.scheduler
            )
            print(f"Loaded checkpoint from {args.checkpoint}")
            # Update optimizer and scheduler if states were loaded
            if optimizer_state:
                ppo_agent.optimizer.load_state_dict(optimizer_state)
            if scheduler_state and ppo_agent.scheduler:
                ppo_agent.scheduler.load_state_dict(scheduler_state)
            optimization_steps = update_count if update_count else 0
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            optimization_steps = 0
    else:
        optimization_steps = 0
    
    # Initialize variables for reward shaping
    last_position = None
    episode_floor_reached = 0
    episode_keys = 0
    movement_bonus = 0
    floor_bonus = 0
    key_bonus = 0
    door_bonus = 0
    proximity_bonus = 0
    
    # Initialize key/door memory
    if args.reward_shaping:
        if 'EnhancedKeyDoorMemory' in globals():
            key_locations = EnhancedKeyDoorMemory()
        else:
            key_locations = EpisodicMemory()
    
    # For storing episode rewards
    episode_rewards = []
    episode_lengths = []
    episode_floors = []
    
    # For tracking progress
    running_reward = 0
    best_reward = float('-inf')
    max_floor_reached = 0
    steps_without_progress = 0
    
    # Create memory system for keys and doors
    memory = EnhancedKeyDoorMemory()
    
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

    # Main training loop
    start_time = time.time()
    steps_done = 0
    episode_count = 0
    current_floor = 0
    max_floor_reached = 0
    last_log_time = start_time
    last_save_time = start_time
    
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
    
    # Map memory to key_door_memory for consistent naming
    key_door_memory = memory
    
    key_collections = 0
    door_openings = 0
    
    logger.log_event("TRAINING", "Starting training loop")
    
    last_eval_time = 0
    
    # LSTM state handling
    lstm_states = None
    
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
                if episode_count > 0 and 'done' in locals() and done:  # Check if 'done' exists and is True
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
        
        # Reset LSTM state at the beginning of each episode if using recurrent policy
        if args.use_lstm:
            ppo_agent.reset_lstm_state()
        
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
            
            # Update position tracking in environment wrapper
            if not hasattr(env, '_previous_position'):
                env._previous_position = [0, 0, 0]
            else:
                env._previous_position = current_position.copy()
            
            # Track highest floor reached
            if info["current_floor"] > current_floor:
                current_floor = info["current_floor"]
                if current_floor > max_floor_reached:
                    max_floor_reached = current_floor
                    floor_msg = f"New floor reached: {current_floor}"
                    logger.log_event("NEW_FLOOR", floor_msg)
                    
                    # Add direct floor checkpoint saving here
                    try:
                        floor_checkpoint_path = os.path.join(args.log_dir, f"floor_{current_floor}.pth")
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
                    logger.log_event("DOOR", door_msg)
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
                    logger.log_event("KEY", key_msg)
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
            
            # Enhanced door proximity detection and reward
            has_detected_door = False
            
            # Visual door detection - check if a door is visible in the observation
            if detect_door_visually(next_obs[0]):
                has_detected_door = True
                door_visual_bonus = 0.1
                shaped_reward += door_visual_bonus
                
                if current_keys > 0:  # Additional bonus if agent has keys and sees a door
                    shaped_reward += 0.2
                    # Replace direct print with logging call
                    logger.log_event("DOOR", f"Door visually detected with key! Bonus added: +0.3")
                else:
                    # Replace direct print with logging call
                    logger.log_event("DOOR", f"Door visually detected! Bonus added: +0.1") 
                
                # Record potential door location
                door_position = current_position
                key_door_memory.add_door_location(current_floor, door_position)
            
            # If agent has a key and is near a door location, give a progressive hint based on proximity
            if current_keys > 0:  
                # Check for doors in memory
                closest_door_distance = float('inf')
                for door_floor, door_positions in key_door_memory.door_locations.items():
                    if door_floor == current_floor:
                        for door_pos in door_positions:
                            dist = sum((current_position[i] - door_pos[i])**2 for i in range(3))**0.5
                            closest_door_distance = min(closest_door_distance, dist)
                
                # Progressive reward based on proximity to doors when agent has keys
                if closest_door_distance < float('inf'):
                    # Higher reward the closer we get
                    if closest_door_distance < 1.0:
                        proximity_bonus = 0.3
                    elif closest_door_distance < 2.0:
                        proximity_bonus = 0.2
                    elif closest_door_distance < 4.0:
                        proximity_bonus = 0.1
                    elif closest_door_distance < 8.0:
                        proximity_bonus = 0.05
                    else:
                        proximity_bonus = 0.01
                        
                    shaped_reward += proximity_bonus
                    reward_components['door_bonus'] += proximity_bonus
                
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
                
            # Time penalty to encourage faster completion
            time_penalty = -0.0001
            shaped_reward += time_penalty
            reward_components['time_penalty'] += time_penalty
            
            # Enhanced exploration bonus based on visit count, with better scaling
            visit_count = info.get("visit_count", 0)
            current_floor = info.get("current_floor", 0)
            
            # Calculate adaptive exploration factors
            # 1. Base exploration factor starts high and decays over training, but maintains a minimum value
            progress = min(1.0, steps_done / (0.8 * args.num_steps))  # Slower decay over 80% of training
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
                        logger.log_event(
                            "EXPLORATION",
                            f"Exploration bonus: +{exploration_bonus:.3f} " +
                            f"(floor: {current_floor}, visits: {visit_count})"
                        )
            
            # Enhance movement reward with directional bias toward unexplored areas
            if hasattr(env, '_previous_position') and env._previous_position is not None:
                # Calculate distance moved
                distance = sum((current_position[i] - env._previous_position[i])**2 for i in range(3))**0.5
                
                # Store movement history for temporal persistence
                if not hasattr(env, '_movement_history'):
                    env._movement_history = deque(maxlen=10)  # Track last 10 movements
                
                # Get current movement direction
                if distance > 0.1:  # Only consider significant movements
                    direction = [(current_position[i] - env._previous_position[i])/max(distance, 0.001) for i in range(3)]
                    env._movement_history.append((direction, current_position.copy(), shaped_reward > 0))
                
                # Higher reward for significant movement
                if distance > 0.5:
                    # Check if agent is moving toward unexplored areas
                    target_info = key_door_memory.get_directions_to_target(
                        current_position, current_floor, current_keys > 0
                    )
                    
                    # Initialize directional bonus
                    directional_bonus = 0.0
                    
                    # Calculate persistence bonus for maintaining direction
                    persistence_bonus = 0.0
                    if len(env._movement_history) >= 3:
                        # Get average direction over past few steps
                        past_dirs = [env._movement_history[-i-1][0] for i in range(min(3, len(env._movement_history)-1))]
                        avg_past_dir = [sum(d[i] for d in past_dirs)/len(past_dirs) for i in range(3)]
                        
                        # Dot product with current direction
                        persistence_dot = sum(direction[i] * avg_past_dir[i] for i in range(3))
                        
                        # Reward for consistent movement
                        if persistence_dot > 0.7:  # Roughly the same direction
                            persistence_bonus = 0.01 * distance * floor_factor
                        
                    # Calculate target-seeking bonus
                    target_bonus = 0.0
                    if target_info:
                        # Calculate dot product to see if agent is moving toward target
                        target_dir = target_info['direction']
                        dot_product = sum(direction[i] * target_dir[i] for i in range(min(len(direction), len(target_dir))))
                        
                        # Higher reward for moving toward targets
                        if dot_product > 0:
                            # Scale by how aligned the movement is with target direction
                            alignment_factor = (dot_product + 1) / 2  # Scale from 0-1
                            target_bonus = 0.03 * distance * alignment_factor * floor_factor
                            
                            # Extra bonus if agent has keys and is moving toward doors
                            if current_keys > 0 and target_info['type'] == 'door':
                                target_bonus *= 1.5  # 50% more reward for door-seeking with keys
                    else:
                        # Basic movement bonus when no targets known
                        target_bonus = 0.015 * distance * floor_factor
                        
                    # Total directional bonus combines persistence and target-seeking
                    directional_bonus = persistence_bonus + target_bonus
                    
                    if directional_bonus > 0:
                        shaped_reward += directional_bonus
                        reward_components['exploration_bonus'] += directional_bonus
                
                # Penalize going in circles or backtracking
                if hasattr(env, '_visited_positions'):
                    position_key = f"{current_floor}_{round(current_position[0], 1)}_{round(current_position[2], 1)}"
                    
                    if position_key in env._visited_positions:
                        revisit_count = env._visited_positions[position_key]
                        
                        # Progressive penalty for revisiting the same location repeatedly
                        if revisit_count > 5:
                            loop_penalty = -0.001 * min(revisit_count, 10)  # Cap penalty
                            shaped_reward += loop_penalty
                else:
                    env._visited_positions = {}
            
            # Store current position for next step comparison
            env._previous_position = current_position
                
            # Enhanced floor completion bonus with progressive scaling
            if info["current_floor"] > current_floor:
                # Progressive floor bonus - higher floors get bigger rewards
                base_floor_bonus = 5.0  # Increased from 2.0 to 5.0 for stronger signal
                floor_progression = info["current_floor"] - current_floor
                floor_bonus = base_floor_bonus * (1 + 0.5 * floor_progression)  # +50% per floor skipped
                
                # Add a significant one-time bonus for reaching a new max floor
                if info["current_floor"] > max_floor_reached:
                    new_floor_achievement_bonus = 10.0  # Major reward for reaching new floors
                    shaped_reward += new_floor_achievement_bonus
                    # Replace direct print with logging call
                    logger.log_event("NEW_ACHIEVEMENT", f"First time reaching floor {info['current_floor']}! +{new_floor_achievement_bonus} bonus!")
                
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
                        milestone_bonus = 15.0  # Substantial bonus for significant progress (increased from 5.0)
                        shaped_reward += milestone_bonus
                        # Replace direct print with logging call
                        logger.log_event("MILESTONE", f"Milestone achieved! Reached new max floor {new_floor}! +{milestone_bonus} bonus")
                        logger.log_event("NEW_MAX_FLOOR", 
                            f"Reached new maximum floor {new_floor} (skipped {new_floor - current_floor - 1} floors)"
                        )
                    else:
                        # Replace direct print with logging call
                        logger.log_event("NEW_MAX_FLOOR", f"Reached new max floor {new_floor}!")
                        logger.log_event("NEW_MAX_FLOOR", 
                            f"Reached new maximum floor {new_floor}"
                        )
                
                # Update current floor to the new floor
                current_floor = new_floor
                reward_components['floor_bonus'] += floor_bonus
                # Replace direct print with logging call
                logger.log_event("FLOOR", f"New floor reached: {current_floor}! Bonus reward added: +{floor_bonus}")
                
                # Reset exploration tracking for new floor
                if hasattr(env, '_visited_positions'):
                    env._visited_positions = {}
                    # Replace direct print with logging call
                    logger.log_event("EXPLORATION", "Reset exploration tracking for new floor")
                
                # Also reset movement history for new floor
                if hasattr(env, '_movement_history'):
                    env._movement_history.clear()
                
                # Add extra hint reward if we have keys to indicate they might be needed
                if info["total_keys"] > 0:
                    key_usage_hint = 2.0  # Increased from 1.0 for stronger signal
                    shaped_reward += key_usage_hint
                    
                # Log the significant achievement
                logger.log_event(
                    "FLOOR_COMPLETE", 
                    f"Completed floor {current_floor-1}, moving to floor {current_floor}"
                )
            
            # Log detailed environment interaction if verbose
            if steps_done % 100 == 0:
                logger.log_event(
                    "ENV_INTERACTION",
                    f"Action: {action}, Reward: {reward}, Shaped Reward: {shaped_reward}"
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
                # Replace direct print with logging call
                logger.log_event("KEY", f"Key visually detected! +{key_visual_bonus} reward")
                
                # Update memory with current position
                current_position = (info.get("x_pos", 0), info.get("y_pos", 0), info.get("z_pos", 0))
                key_door_memory.add_key_location(info["current_floor"], current_position)
                # Track key collection for metrics
                key_collections += 1
                metrics_tracker.update('key_collections', key_collections)
                logger.log_event("KEY", f"Key detected at floor {info['current_floor']}, position {current_position}")

            # Check for key collection by comparing with previous state
            has_key = info["total_keys"] > 0
            if hasattr(env, '_previous_keys') and env._previous_keys is not None:
                if has_key and env._previous_keys == 0:
                    # Additional reward for actually collecting the key
                    key_collection_bonus = 4.0
                    shaped_reward += key_collection_bonus
                    # Replace direct print with logging call
                    logger.log_event("KEY_COLLECTED", f"Key collected! +{key_collection_bonus} reward")
                    logger.log_event("KEY_COLLECTED", f"Key collected at floor {info['current_floor']}")
            # Update the key count tracker
            env._previous_keys = info["total_keys"]

            # Apply intrinsic motivation if ICM is enabled
            if args.use_icm and ppo_agent.icm is not None:
                # Convert observation to tensor for ICM
                if previous_obs is not None:
                    # Process current and next observation using frame stacking
                    state_tensor = preprocess_observation_for_icm(previous_obs, frame_stack).to(device)
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
                
                # ... existing episode completion code ...
                break
        
        # If episode was cut off, we need to compute the value of the final state
        if not done:
            with torch.no_grad():
                policy_logits, next_value = model(state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0.0  # Terminal state has value 0
            
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

        # Add current trajectory to experience replay if enabled
        if args.experience_replay:
            add_to_replay(replay_buffer, states, actions, rewards, log_probs, values, dones, max_replay_size)
            
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
                        replay_buffer, ppo_agent.batch_size, include_dones=True
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
                    logger.log_event("ERROR", f"LSTM update failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Try to fall back to non-recurrent mode
                    try:
                        logger.log_event("RECOVERY", "Attempting fallback to non-recurrent update")
                        states, actions, returns, advantages, old_log_probs = sample_from_replay(
                            replay_buffer, ppo_agent.batch_size
                        )
                        metrics = ppo_agent.update(states, actions, old_log_probs, returns, advantages)
                        optimization_steps += 1
                    except Exception as e2:
                        logger.log_event("FATAL", f"Fallback update also failed: {e2}")
                        # Skip this update to avoid terminating training
            else:
                # Standard update for non-recurrent policy
                try:
                    states, actions, returns, advantages, old_log_probs = sample_from_replay(
                        replay_buffer, ppo_agent.batch_size
                    )
                    metrics = ppo_agent.update(states, actions, old_log_probs, returns, advantages)
                    optimization_steps += 1
                except Exception as e:
                    logger.log_event("ERROR", f"Update failed: {e}")
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
            eval_reward, eval_max_floor = evaluate_policy(
                model=model,
                env_path=args.env_path,
                seed=args.seed,
                device=device,
                action_flattener=action_flattener,
                training_logger=logger,
                num_episodes=args.eval_episodes
            )
            last_eval_time = steps_done
            
            # Save a checkpoint after evaluation
            checkpoint_path = os.path.join(args.log_dir, f"eval_step_{steps_done}.pth")
            save_checkpoint(
                model, 
                checkpoint_path, 
                optimizer=ppo_agent.optimizer, 
                scheduler=ppo_agent.scheduler, 
                metrics=metrics_tracker.metrics
            )
            
            logger.log_event("CHECKPOINT", f"Saved post-evaluation checkpoint at step {steps_done}")
            
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
            checkpoint_path = os.path.join(args.log_dir, f"step_{steps_done}.pth")
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

        # Add this check after PPO updates
        if optimization_steps % 200 == 0 or ppo_agent.optimizer.param_groups[0]['lr'] < 5e-5:
            ppo_agent.reset_optimizer_state()

        # Apply floor-specific reward scaling to prioritize progression
        # Lower floors get reduced reward scale over time to encourage moving higher
        floor_scaling_factor = 1.0
        
        # Gradually reduce reward scale for easier floors as training progresses
        if steps_done > 100000:  # After initial learning period
            current_max_floor = max_floor_reached
            progress_ratio = min(1.0, steps_done / args.num_steps)
            
            # Calculate floor-specific scaling
            # Floors far below the max get diminished rewards
            if current_floor < current_max_floor - 1:
                # Scale down rewards for floors more than 1 below max
                floor_gap = current_max_floor - current_floor
                # Stronger reduction as training progresses and gap increases
                reduction = 0.3 * progress_ratio * min(floor_gap, 3)  # Cap at 3 floors difference
                floor_scaling_factor = max(0.5, 1.0 - reduction)  # Don't go below 50%
                
                # Log when we apply significant scaling
                if floor_scaling_factor < 0.8 and steps_done % 100 == 0:
                    logger.log_event("REWARD_SHAPING", f"Applying floor scaling factor {floor_scaling_factor:.2f} (floor {current_floor} vs max {current_max_floor})")
            
            # Apply the scaling factor
            shaped_reward *= floor_scaling_factor

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
                
                # Log ICM metrics
                if icm_metrics and step % 100 == 0:
                    for key, value in icm_metrics.items():
                        metrics_tracker.update(f"icm/{key}", value)
            except Exception as e:
                # Log error but don't crash training
                # Replace direct print with logging call
                logger.log_event("ERROR", f"Error updating ICM: {e}")

    # Final cleanup
    logger.close()
    env.close()

    return model, metrics_tracker, logger.metrics

def add_to_replay(replay_buffer, states, actions, rewards, log_probs, values, dones, max_replay_size):
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

def sample_from_replay(replay_buffer, batch_size, include_dones=False):
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
            debug_print(f"Error in sample_from_replay: {e}")
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
            debug_print(f"Error in sample_from_replay: {e}")
            # Fall back to lists if numpy conversion fails
            return states, actions, returns, advantages, old_log_probs

def evaluate_policy(model, env_path, seed, device, action_flattener, training_logger, num_episodes=5):
    """Evaluate the policy without exploration."""
    eval_env = create_obstacle_tower_env(
        executable_path=env_path,
        realtime_mode=False,
        timeout=300
    )
    eval_env.seed(seed + 100)  # Different seed for evaluation
    
    # Check if the model is recurrent
    is_recurrent = hasattr(model, 'use_lstm') and model.use_lstm
    
    total_reward = 0
    max_floor = 0
    
    for episode in range(num_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        
        # Initialize frame stack
        frame_stack = deque(maxlen=4)
        obs = np.transpose(obs[0], (2, 0, 1)) / 255.0
        for _ in range(4):
            frame_stack.append(obs)
        
        # Initialize LSTM state for recurrent policies
        lstm_state = None
        if is_recurrent:
            lstm_state = model.init_lstm_state(batch_size=1, device=device)
        
        while not done:
            state = np.concatenate(list(frame_stack), axis=0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Use deterministic actions for evaluation
            with torch.no_grad():
                if is_recurrent:
                    # Forward pass through recurrent model
                    policy_logits, _, next_lstm_state = model(state_tensor, lstm_state)
                    lstm_state = next_lstm_state  # Update LSTM state
                else:
                    # Standard forward pass
                    policy_logits, _ = model(state_tensor)
                
                # Select best action
                action_idx = torch.argmax(policy_logits, dim=1).item()
                action = action_flattener.lookup_action(action_idx)
                action = np.array(action)
                
            next_obs, reward, done, info = eval_env.step(action)
            episode_reward += reward
            
            # Update frame stack
            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]  # Handle tuple observation
            next_obs = np.transpose(next_obs, (2, 0, 1)) / 255.0
            frame_stack.append(next_obs)
            
            # Track max floor
            if info.get("current_floor", 0) > max_floor:
                max_floor = info["current_floor"]
        
        # Reset LSTM state between episodes
        if is_recurrent:
            lstm_state = None
            
        total_reward += episode_reward
        
    eval_env.close()
    avg_reward = total_reward / num_episodes
    
    print(f"Evaluation: Avg Reward: {avg_reward:.2f}, Max Floor: {max_floor}")
    if training_logger:
        training_logger.log_event("EVAL", f"Evaluation results: Avg Reward: {avg_reward:.2f}, Max Floor: {max_floor}")
    return avg_reward, max_floor

def preprocess_observation(observation):
    """
    Preprocess observation for neural network input.
    
    Args:
        observation: Raw observation from environment
        
    Returns:
        Preprocessed observation tensor
    """
    if observation is None:
        # Return zeros for None observations (can happen at end of episodes)
        return torch.zeros((3, 84, 84), dtype=torch.float32)
        
    # If observation is a tuple, extract visual observation
    if isinstance(observation, tuple):
        observation = observation[0]
        
    # Convert to numpy array if not already
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
        
    # Check if already in [C,H,W] format 
    if observation.shape[0] == 3 and len(observation.shape) == 3:
        # Already in correct format, just normalize
        obs_array = observation / 255.0
    else:
        # Need to transpose from [H,W,C] to [C,H,W]
        obs_array = np.transpose(observation, (2, 0, 1)) / 255.0
        
    # Convert to PyTorch tensor
    obs_tensor = torch.from_numpy(obs_array).float()
    
    # Ensure the tensor is contiguous in memory for better performance
    if not obs_tensor.is_contiguous():
        obs_tensor = obs_tensor.contiguous()
        
    # Resize the observation if needed using interpolation (faster than crop/resize chains)
    if obs_tensor.shape[1:] != (84, 84):
        obs_tensor = torch.nn.functional.interpolate(
            obs_tensor.unsqueeze(0), 
            size=(84, 84), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    return obs_tensor

def preprocess_observation_for_icm(observation, frame_stack):
    """
    Preprocess observation specifically for ICM input, ensuring correct format 
    with stacked frames for the ICM module.
    
    Args:
        observation: Raw observation from environment
        frame_stack: Deque containing previous frames
        
    Returns:
        Tensor with stacked frames suitable for ICM input [B, 12, 84, 84]
    """
    # Process the current observation
    processed_frame = preprocess_observation(observation)
    
    if frame_stack and len(frame_stack) == 4:
        # Use the frame stack to create a 12-channel input
        frames = list(frame_stack)
        # Optimize by pre-allocating tensor and copying data
        stacked_frames = torch.zeros(12, 84, 84, dtype=torch.float32)
        
        # Track how many valid frames we've processed
        valid_frames = 0
        offset = 0
        
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # If frame is numpy array, convert to tensor
                if frame.shape[0] == 3:
                    # Already in [C,H,W] format
                    frame_tensor = torch.from_numpy(frame).float()
                else:
                    # Need to transpose
                    frame_tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1)) / 255.0).float()
            elif isinstance(frame, torch.Tensor):
                frame_tensor = frame
            else:
                continue
                
            # Ensure frame has correct shape
            if frame_tensor.shape[0] == 3 and frame_tensor.shape[1:] == (84, 84):
                # Copy directly into pre-allocated tensor for better performance
                stacked_frames[offset:offset+3] = frame_tensor
                offset += 3
                valid_frames += 1
        
        # If we have enough frames, return the stacked tensor
        if valid_frames == 4:
            return stacked_frames
    
    # Fallback: Just repeat the current frame 4 times to get 12 channels
    # This is more efficient than multiple concatenation operations
    return processed_frame.repeat(4, 1, 1)

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
    parser.add_argument('--curriculum', action='store_true',
                        help='Use curriculum learning to gradually increase difficulty')
    
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
    parser.add_argument('--vf_coef', type=float, default=0.5, 
                        help='Value function loss coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--target_kl', type=float, default=0.01,
                        help='Target KL divergence for early stopping')
                        
    # ICM parameters
    parser.add_argument('--use_icm', action='store_true',
                        help='Use Intrinsic Curiosity Module for exploration')
    parser.add_argument('--icm_reward_scale', type=float, default=0.01,
                        help='Scaling factor for intrinsic rewards')
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
                        help='Console verbosity level: 0=minimal, 1=normal, 2=verbose')
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
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Sequence length for LSTM training')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()