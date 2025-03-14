import numpy as np
import torch
import gym
import itertools
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings

def to_python_type(value):
    """Convert NumPy or Torch types to Python native types"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, 'dtype'):  # NumPy scalar
        return float(value)
    elif isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    elif isinstance(value, list) or isinstance(value, tuple):
        return [to_python_type(v) for v in value]
    elif isinstance(value, dict):
        return {k: to_python_type(v) for k, v in value.items()}
    else:
        return value

def normalize(x):
    x = np.array(x)
    return (x - x.mean()) / (x.std() + 1e-8)

def save_checkpoint(model, path, optimizer=None, scheduler=None, metrics=None, update_count=None):
    """
    Save a model checkpoint including training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    if metrics is not None:
        checkpoint['metrics'] = metrics
        
    if update_count is not None:
        checkpoint['update_count'] = update_count
        
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")
    
def load_checkpoint(model, path, optimizer=None, scheduler=None):
    """
    Load a model checkpoint including training metrics
    """
    metrics = None
    update_count = None
    
    try:
        # First, try to allow NumPy scalars in the safe globals list (PyTorch 2.0+)
        try:
            # This is a PyTorch 2.0+ feature
            import torch.serialization
            # Add numpy.core.multiarray.scalar to allowed globals
            torch.serialization.add_safe_globals(['numpy.core.multiarray', 'scalar'])
            
            # Try loading with weights_only=True for better security
            checkpoint = torch.load(path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded checkpoint with weights_only=True")
            
            # For optimizer and scheduler, we need to load the full checkpoint
            if optimizer is not None or scheduler is not None or metrics is not None:
                full_checkpoint = torch.load(path)
                
                if optimizer is not None and 'optimizer_state_dict' in full_checkpoint:
                    optimizer.load_state_dict(full_checkpoint['optimizer_state_dict'])
                
                if scheduler is not None and 'scheduler_state_dict' in full_checkpoint:
                    scheduler.load_state_dict(full_checkpoint['scheduler_state_dict'])
                    
                metrics = full_checkpoint.get('metrics', None)
                update_count = full_checkpoint.get('update_count', None)
                
        except (AttributeError, ImportError, RuntimeError):
            # If add_safe_globals is not available or fails, fall back
            raise RuntimeError("Could not use add_safe_globals")
            
    except Exception as e:
        # Fall back to regular loading for older PyTorch versions or if security features fail
        warnings.warn(f"Falling back to regular checkpoint loading: {str(e)}")
        
        try:
            # Load without weights_only restriction - less secure but more compatible
            checkpoint = torch.load(path, map_location=next(model.parameters()).device)
            
            # Load model state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try direct loading in case the checkpoint is just the model state dict
                model.load_state_dict(checkpoint)
            
            # Load optimizer if provided
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler if provided
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            metrics = checkpoint.get('metrics', None)
            update_count = checkpoint.get('update_count', None)
            
            print("Successfully loaded checkpoint without weights_only")
            
        except Exception as load_err:
            print(f"Error loading checkpoint: {load_err}")
            raise
    
    return model, metrics, update_count

def plot_metrics(metrics_dir, metrics_data, show=True, save=True):
    """
    Plot training metrics
    """
    plt.figure(figsize=(20, 15))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(metrics_data.get('episode_rewards', []))
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot episode lengths
    plt.subplot(3, 2, 2)
    plt.plot(metrics_data.get('episode_lengths', []))
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    # Plot floor reached
    plt.subplot(3, 2, 3)
    plt.plot(metrics_data.get('episode_floors', []))
    plt.title('Max Floor Reached')
    plt.xlabel('Episode')
    plt.ylabel('Floor')
    
    # Plot policy loss
    plt.subplot(3, 2, 4)
    plt.plot(metrics_data.get('policy_losses', []))
    plt.title('Policy Loss')
    plt.xlabel('Update')
    plt.ylabel('Loss')
    
    # Plot value loss
    plt.subplot(3, 2, 5)
    plt.plot(metrics_data.get('value_losses', []))
    plt.title('Value Loss')
    plt.xlabel('Update')
    plt.ylabel('Loss')
    
    # Plot entropy
    plt.subplot(3, 2, 6)
    plt.plot(metrics_data.get('entropy_values', []))
    plt.title('Entropy')
    plt.xlabel('Update')
    plt.ylabel('Entropy')
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        plt.savefig(os.path.join(metrics_dir, f'metrics_{timestamp}.png'))
    
    if show:
        plt.show()
    else:
        plt.close()

def save_metrics_json(metrics_dir, metrics_data):
    """
    Save metrics data to a JSON file
    """
    # Convert numpy arrays and scalar types to JSON-serializable types
    serializable_metrics = to_python_type(metrics_data)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(metrics_dir, f'metrics_{timestamp}.json'), 'w') as f:
        json.dump(serializable_metrics, f)

class ActionFlattener:
    def __init__(self, branched_action_space):
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = gym.spaces.Discrete(len(self.action_lookup))

    def _create_lookup(self, branched_action_space):
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        return {_scalar: _action for (_scalar, _action) in enumerate(all_actions)}

    def lookup_action(self, action):
        return self.action_lookup[action]
        
class MetricsTracker:
    """
    Track and save training metrics
    """
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_floors': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_values': [],
            'clip_fractions': [],
            'approx_kl_divs': [],
            'explained_variances': [],
            'learning_rates': [],
            'steps_per_second': [],
            'fps_history': [],
            'door_openings': [],
            'key_collections': []
        }
        
        # Ensure directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
    def update(self, key, value):
        """Add a new value to a metric"""
        if key in self.metrics:
            # Convert NumPy types to Python native types before storing
            if hasattr(value, 'dtype'):  # NumPy scalar
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                else:
                    value = float(value)
            self.metrics[key].append(value)
            
    def update_from_ppo(self, ppo):
        """Update metrics from a PPO object"""
        # Use getattr with default values to avoid AttributeError if attributes don't exist
        if hasattr(ppo, 'policy_losses'):
            self.metrics['policy_losses'] = ppo.policy_losses
        elif hasattr(ppo, 'policy_loss'):
            # If there's a single value instead of a list
            if 'policy_losses' not in self.metrics:
                self.metrics['policy_losses'] = []
            self.metrics['policy_losses'].append(ppo.policy_loss)
            
        if hasattr(ppo, 'value_losses'):
            self.metrics['value_losses'] = ppo.value_losses
        elif hasattr(ppo, 'value_loss'):
            # If there's a single value instead of a list
            if 'value_losses' not in self.metrics:
                self.metrics['value_losses'] = []
            self.metrics['value_losses'].append(ppo.value_loss)
            
        if hasattr(ppo, 'entropy_values'):
            self.metrics['entropy_values'] = ppo.entropy_values
        elif hasattr(ppo, 'entropy'):
            # If there's a single value instead of a list
            if 'entropy_values' not in self.metrics:
                self.metrics['entropy_values'] = []
            self.metrics['entropy_values'].append(ppo.entropy)
            
        if hasattr(ppo, 'clip_fractions'):
            self.metrics['clip_fractions'] = ppo.clip_fractions
        
        if hasattr(ppo, 'approx_kl_divs'):
            self.metrics['approx_kl_divs'] = ppo.approx_kl_divs
        elif hasattr(ppo, 'approx_kl'):
            if 'approx_kl_divs' not in self.metrics:
                self.metrics['approx_kl_divs'] = []
            self.metrics['approx_kl_divs'].append(ppo.approx_kl)
        elif hasattr(ppo, 'kl_divergence'):
            if 'approx_kl_divs' not in self.metrics:
                self.metrics['approx_kl_divs'] = []
            self.metrics['approx_kl_divs'].append(ppo.kl_divergence)
            
        if hasattr(ppo, 'explained_variances'):
            self.metrics['explained_variances'] = ppo.explained_variances
        elif hasattr(ppo, 'explained_variance'):
            if 'explained_variances' not in self.metrics:
                self.metrics['explained_variances'] = []
            self.metrics['explained_variances'].append(ppo.explained_variance)
            
        if hasattr(ppo, 'learning_rates'):
            self.metrics['learning_rates'] = ppo.learning_rates
        elif hasattr(ppo, 'optimizer') and hasattr(ppo.optimizer, 'param_groups'):
            # Extract current learning rate from optimizer
            if 'learning_rates' not in self.metrics:
                self.metrics['learning_rates'] = []
            self.metrics['learning_rates'].append(ppo.optimizer.param_groups[0]['lr'])
            
    def save(self, plot=True):
        """Save metrics to disk"""
        # Save metrics as NumPy arrays
        for key, values in self.metrics.items():
            if values:  # Only save non-empty metrics
                np.save(os.path.join(self.log_dir, f"{key}.npy"), np.array(values))
        
        # Save metrics as JSON for easier analysis
        save_metrics_json(self.log_dir, self.metrics)
        
        # Plot metrics
        if plot:
            plot_metrics(self.log_dir, self.metrics, show=False, save=True)
            
    def get_latest(self, key, default=None):
        """Get the latest value of a metric"""
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        return default

class TrainingLogger:
    """
    Comprehensive logging system for reinforcement learning training.
    Logs detailed information about training progress, model behavior,
    and environment interactions to help with analysis and fine-tuning.
    """
    def __init__(self, log_dir, model=None, log_frequency=10):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.log_frequency = log_frequency
        self.episode_count = 0
        self.update_count = 0
        self.model = model
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write(f"=== OBSTACLE TOWER TRAINING LOG - STARTED {datetime.now()} ===\n\n")
            
        if model is not None:
            self.log_model_architecture()
    
    def update_model(self, model):
        """Update the model after logger initialization and log its architecture"""
        self.model = model
        self.log_model_architecture()
        
    def log(self, message, level="INFO"):
        """Log a message with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
            
    def log_hyperparameters(self, params):
        """Log hyperparameters at the beginning of training"""
        self.log("=== HYPERPARAMETERS ===", "CONFIG")
        for key, value in params.items():
            self.log(f"{key}: {value}", "CONFIG")
        self.log("=====================", "CONFIG")
        
    def log_model_architecture(self):
        """Log the model architecture"""
        if self.model is None:
            return
            
        self.log("=== MODEL ARCHITECTURE ===", "MODEL")
        # Get model structure as string
        model_str = str(self.model)
        for line in model_str.split('\n'):
            self.log(line, "MODEL")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"Total parameters: {total_params:,}", "MODEL")
        self.log(f"Trainable parameters: {trainable_params:,}", "MODEL")
        self.log("========================", "MODEL")
        
    def log_episode_start(self, episode_num):
        """Log the start of a new episode"""
        self.episode_count = episode_num
        if episode_num % self.log_frequency == 0:
            self.log(f"Starting episode {episode_num}", "EPISODE")
            
    def log_episode_complete(self, episode_stats):
        """Log detailed stats at the end of an episode"""
        if self.episode_count % self.log_frequency == 0:
            self.log(f"=== EPISODE {self.episode_count} COMPLETED ===", "EPISODE")
            for key, value in episode_stats.items():
                self.log(f"{key}: {value}", "EPISODE")
            self.log("==========================================", "EPISODE")
            
    def log_update(self, update_metrics):
        """Log policy update details"""
        self.update_count += 1
        if self.update_count % self.log_frequency == 0:
            self.log(f"=== POLICY UPDATE {self.update_count} ===", "UPDATE")
            for key, value in update_metrics.items():
                self.log(f"{key}: {value}", "UPDATE")
            self.log("==========================", "UPDATE")
            
    def log_environment_interaction(self, action, reward, shaped_reward, info, step_num):
        """Log detailed environment interactions (use sparingly as this generates a lot of data)"""
        # Only log occasionally or for significant events to avoid excessive output
        if step_num % 1000 == 0:
            self.log(f"Step {step_num}: Action={action}, Reward={reward}, Shaped Reward={shaped_reward}", "ENV")
            # Log any interesting information from info dict
            if "current_floor" in info:
                self.log(f"Current floor: {info['current_floor']}", "ENV")
            if "total_keys" in info:
                self.log(f"Keys: {info['total_keys']}", "ENV")
                
    def log_significant_event(self, event_type, message):
        """Log significant events like reaching a new floor, door opening, etc."""
        self.log(f"[{event_type}] {message}", "EVENT")
        
    def log_reward_breakdown(self, base_reward, shaped_components):
        """Log detailed breakdown of reward components"""
        self.log(f"Base reward: {base_reward}", "REWARD")
        for component, value in shaped_components.items():
            self.log(f"  {component}: {value}", "REWARD")
        self.log(f"  Total: {base_reward + sum(shaped_components.values())}", "REWARD")
        
    def log_training_summary(self, metrics_tracker, elapsed_time, steps_done):
        """Log a periodic summary of training progress"""
        self.log("=== TRAINING SUMMARY ===", "SUMMARY")
        self.log(f"Total steps: {steps_done}", "SUMMARY")
        self.log(f"Elapsed time: {elapsed_time:.2f} seconds", "SUMMARY")
        self.log(f"Steps per second: {steps_done / elapsed_time:.2f}", "SUMMARY")
        
        # Log latest metrics if available
        for key in ['episode_rewards', 'episode_lengths', 'episode_floors', 
                   'policy_losses', 'value_losses', 'entropy_values']:
            latest = metrics_tracker.get_latest(key)
            if latest is not None:
                self.log(f"Latest {key}: {latest}", "SUMMARY")
                
        # Log max floor reached
        floors = metrics_tracker.metrics.get('episode_floors', [])
        if floors:
            self.log(f"Max floor reached: {max(floors)}", "SUMMARY")
            
        self.log("======================", "SUMMARY")
        
    def log_to_console(self, message):
        """Log message to both file and console"""
        print(message)
        self.log(message)
        
    def close(self):
        """Close the log with a final message"""
        self.log(f"=== TRAINING COMPLETED AT {datetime.now()} ===", "INFO")