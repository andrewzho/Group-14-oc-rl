import numpy as np
import torch
import gym
import itertools
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings

# Import config
from config import NETWORK_CONFIG

class ConsoleLogger:
    """
    Enhanced console logger that also writes to a file.
    """
    def __init__(self, log_dir, filename="training_log.txt"):
        import os
        import sys
        import time
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, filename)
        self.terminal = sys.stdout
        
        # Create or clear the log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== Training Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        # Redirect stdout
        sys.stdout = self
    
    def write(self, message):
        # Write to terminal
        self.terminal.write(message)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(message)
    
    def flush(self):
        # Needed for compatibility
        self.terminal.flush()
    
    def close(self):
        import sys
        import time
        
        # Restore original stdout
        sys.stdout = self.terminal
        
        # Add final timestamp
        with open(self.log_file, 'a') as f:
            f.write(f"\n=== Log Ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

def validate_parameters(args):
    """
    Validate command line arguments and provide warnings for potential issues.
    
    Args:
        args: Command line arguments
        
    Returns:
        args: Potentially modified arguments
    """
    warnings = []
    
    # Check environment path
    if args.env_path is None:
        warnings.append("Environment path not specified. Will use default if available.")
    elif not os.path.exists(args.env_path):
        warnings.append(f"Environment executable not found at {args.env_path}")
    
    # Check network architecture
    if args.network not in ["cnn", "resnet", "lstm", "dual"]:
        warnings.append(f"Unknown network architecture: {args.network}. Will use 'cnn'.")
        args.network = "cnn"
    
    # Check demo path if demos are enabled
    if not args.no_demos and not os.path.exists(args.demo_path):
        warnings.append(f"Demonstration file not found: {args.demo_path}")
    
    # Check timestep-related parameters
    if args.eval_freq >= args.timesteps:
        warnings.append(f"Evaluation frequency ({args.eval_freq}) is >= total timesteps ({args.timesteps})")
        args.eval_freq = args.timesteps // 10
        warnings.append(f"Setting evaluation frequency to {args.eval_freq}")
    
    if args.save_freq >= args.timesteps:
        warnings.append(f"Save frequency ({args.save_freq}) is >= total timesteps ({args.timesteps})")
        args.save_freq = args.timesteps // 5
        warnings.append(f"Setting save frequency to {args.save_freq}")
    
    # Check buffer size and memory usage
    obs_size_estimate = 42 * 42 * 168  # Conservative estimate based on error message
    memory_per_transition = obs_size_estimate * 4 * 2  # float32 bytes * 2 (obs + next_obs)
    estimated_memory_gb = (args.buffer_size * memory_per_transition) / (1024**3)
    
    if estimated_memory_gb > 16:  # Assume 16GB as reasonable upper limit
        warnings.append(f"Buffer size {args.buffer_size} may require ~{estimated_memory_gb:.1f} GB of memory")
        if not args.memory_efficient:
            warnings.append("Consider using --memory-efficient flag or reducing --buffer-size")
    
    # Check compatibility of memory-efficient mode with RND
    if args.memory_efficient and not args.no_rnd and not args.force_no_rnd:
        warnings.append("Memory-efficient mode may cause issues with RND due to observation shape changes")
        warnings.append("Consider using --force-no-rnd with --memory-efficient mode")
    
    # Print warnings
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
        print()
    
    return args


def setup_network_architecture(network_type, obs_shape):
    """
    Set up the network architecture based on the specified type.
    
    Args:
        network_type: Type of network architecture ("cnn", "resnet", "lstm", "dual")
        obs_shape: Observation shape
        
    Returns:
        dict: Policy kwargs for the SAC agent
    """
    policy_kwargs = {
        "normalize_images": False,  # Images are already normalized and in channel-first format
    }
    
    if network_type == "cnn":
        # Use default CNN from stable-baselines3
        return policy_kwargs
    
    try:
        # Import feature extractor classes
        from src.models.networks import feature_extractors
        
        # Set up custom feature extractor
        policy_kwargs["features_extractor_class"] = feature_extractors[network_type]
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": NETWORK_CONFIG["hidden_dim"]}
        
        # Special handling for specific network types
        if network_type == "lstm":
            policy_kwargs["features_extractor_kwargs"]["lstm_hidden_size"] = 256
            policy_kwargs["features_extractor_kwargs"]["lstm_layers"] = 1
        elif network_type == "dual":
            policy_kwargs["features_extractor_kwargs"]["state_dim"] = 8
        elif network_type == "resnet":
            # You can add specific ResNet configurations here if needed
            pass
            
    except (ImportError, KeyError) as e:
        print(f"Error setting up network architecture: {e}")
        print("Falling back to default CNN architecture")
        # Reset to default
        policy_kwargs = {
            "normalize_images": False,
        }
    
    return policy_kwargs


class MemoryEfficientObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to reduce memory usage of observations.
    
    Reduces resolution and converts RGB to grayscale if needed.
    """
    def __init__(self, env, width=32, height=32, grayscale=True, max_frames=4, debug=False, preserve_obs_shape=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.max_frames = max_frames
        self.debug = debug
        self.preserve_obs_shape = preserve_obs_shape  # Use consistent naming throughout
        
        # Import necessary modules
        from gym import spaces
        
        # Determine new observation space
        old_shape = env.observation_space.shape
        
        if self.debug:
            print(f"Original observation space shape: {old_shape}")
        
        # If preserving shape type, maintain the same dimensions but reduce resolution
        if preserve_obs_shape:
            if len(old_shape) == 3 and old_shape[0] <= 10:  # Probably frame stacked (frames, height, width)
                frames = old_shape[0]
                new_shape = (frames, height, width)
            elif len(old_shape) == 3:  # Probably (height, width, channels)
                channels = 1 if grayscale else old_shape[2]
                new_shape = (height, width, channels)
            elif len(old_shape) == 4:  # Probably (frames, height, width, channels)
                frames = old_shape[0]
                channels = 1 if grayscale else old_shape[3] 
                new_shape = (frames, height, width, channels)
            else:
                new_shape = old_shape
                print(f"WARNING: Cannot preserve shape type for {old_shape}. Using as is.")
        else:
            # Determine observation format for standard memory-efficient mode
            if len(old_shape) == 3:
                # For (height, width, channels) OR (frames, height, width)
                if old_shape[0] <= 10:  # Likely frame stacked
                    # Convert to standard format with channels
                    frames = old_shape[0]
                    new_shape = (frames, height, width, 1 if grayscale else 3)
                else:
                    # Regular image
                    channels = 1 if grayscale else old_shape[2]
                    new_shape = (height, width, channels)
            elif len(old_shape) == 4:
                # Already (frames, height, width, channels)
                frames = old_shape[0]
                channels = 1 if grayscale else old_shape[3]
                new_shape = (frames, height, width, channels)
            else:
                print(f"WARNING: Unexpected observation shape: {old_shape}. Keeping as is.")
                new_shape = old_shape
            
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=new_shape, 
            dtype=np.uint8
        )
        
        # Store shape information for processing
        self.input_shape = old_shape
        self.output_shape = new_shape
        
        print(f"MemoryEfficientWrapper: Input shape {old_shape} -> Output shape {new_shape}")
        if preserve_obs_shape:
            print("  (Preserving shape dimensions while reducing resolution)")
        
    def observation(self, obs):
        """Process observation"""
        # Safety check
        if obs is None:
            return None
            
        try:
            import cv2
            
            if self.debug:
                print(f"Received observation with shape: {obs.shape}, dtype: {obs.dtype}")
            
            # Handle different observation formats based on preserve_obs_shape flag
            input_shape = obs.shape
            
            # Special handling for preserve_obs_shape
            if self.preserve_obs_shape:
                # For frame stacked observations (frames, height, width)
                if len(input_shape) == 3 and input_shape[0] <= 10:
                    frames = input_shape[0]
                    # Reshape but keep dimensions the same
                    processed = np.zeros((frames, self.height, self.width), dtype=np.uint8)
                    
                    for i in range(frames):
                        frame = obs[i]
                        # Resize
                        processed[i] = cv2.resize(frame, (self.width, self.height), 
                                             interpolation=cv2.INTER_AREA)
                    
                    return processed
                    
                # For regular images (height, width, channels)
                elif len(input_shape) == 3 and input_shape[0] > 10:
                    # Regular image, just resize
                    resized = cv2.resize(obs, (self.width, self.height), 
                                    interpolation=cv2.INTER_AREA)
                    
                    if self.grayscale and input_shape[2] > 1:
                        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                        
                    return resized
                    
                # For stacked frames with channels (frames, height, width, channels)
                elif len(input_shape) == 4:
                    frames = input_shape[0]
                    channels = 1 if self.grayscale else input_shape[3]
                    processed = np.zeros((frames, self.height, self.width, channels), dtype=np.uint8)
                    
                    for i in range(frames):
                        frame = obs[i]
                        # Resize
                        resized = cv2.resize(frame, (self.width, self.height), 
                                        interpolation=cv2.INTER_AREA)
                        
                        if self.grayscale and input_shape[3] > 1:
                            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                            resized = resized[..., np.newaxis]
                            
                        processed[i] = resized
                        
                    return processed
                
                # If unknown shape, return as is
                return obs
                
            # Standard processing (not preserving shape type)
            # For observations shaped like (frames, height, width)
            if len(input_shape) == 3 and input_shape[0] < 10:
                frames, height, width = input_shape
                
                # Reshape to expected format with channel dimension
                processed = np.zeros((frames, self.height, self.width, 1), dtype=np.uint8)
                
                for i in range(frames):
                    # Get frame and add channel dimension if needed
                    frame = obs[i]
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    
                    # Ensure correct shape
                    if len(frame.shape) == 2:  # If grayscale
                        frame = frame[..., np.newaxis]
                        
                    processed[i] = frame
                
                return processed
                
            # For RGB or grayscale images (height, width, channels)
            elif len(input_shape) == 3 and input_shape[2] <= 4:
                # Standard RGB/grayscale image
                resized = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
                
                if self.grayscale and input_shape[2] > 1:
                    grayscale = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                    return grayscale[..., np.newaxis]
                
                return resized
                
            # For frame-stacked observations (frames, height, width, channels)
            elif len(input_shape) == 4:
                frames, height, width, channels = input_shape
                processed = np.zeros((frames, self.height, self.width, 
                                   1 if self.grayscale else channels), 
                                   dtype=np.uint8)
                
                for i in range(frames):
                    frame = obs[i]
                    resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
                    
                    if self.grayscale and channels > 1:
                        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
                        processed[i] = gray[..., np.newaxis]
                    else:
                        processed[i] = resized
                        
                return processed
                
            # For special case (4, 42, 168) handle specially
            elif len(input_shape) == 3 and input_shape[0] < 10 and input_shape[2] > 10:
                if self.debug:
                    print(f"Special case handling for shape {input_shape}")
                    
                frames, height, features = input_shape
                
                # Create a fallback array with proper dimensions for policy
                return np.zeros((frames, self.height, self.width, 1), dtype=np.uint8)
                
            else:
                # Unknown format - create a zeros array with proper shape
                print(f"WARNING: Unhandled observation shape: {input_shape}. Creating empty array.")
                return np.zeros(self.output_shape, dtype=np.uint8)
                
        except Exception as e:
            print(f"Error processing observation: {e}")
            print(f"Observation shape: {input_shape}")
            # Return zero array as fallback
            return np.zeros(self.output_shape, dtype=np.uint8)


def create_memory_efficient_env(env, width=32, height=32, debug=False, preserve_obs_shape=False):
    """
    Apply memory-efficient wrappers to an environment.
    
    Args:
        env: Environment to wrap
        width: Width to resize observations to
        height: Height to resize observations to
        debug: Whether to print debug information
        preserve_obs_shape: Whether to preserve original shape dimensions
        
    Returns:
        Wrapped environment
    """
    # Print original observation space for debugging
    if debug:
        if hasattr(env, 'observation_space'):
            print(f"Original observation space: {env.observation_space}")
            if hasattr(env.observation_space, 'shape'):
                print(f"Original shape: {env.observation_space.shape}")
                
                # If this is a vector environment, try to access the base environment
                if hasattr(env, 'envs') and len(env.envs) > 0:
                    print(f"Base env observation space: {env.envs[0].observation_space}")
    
    # Create wrapper
    wrapped_env = MemoryEfficientObservationWrapper(
        env, 
        width=width, 
        height=height, 
        grayscale=True, 
        debug=debug,
        preserve_obs_shape=preserve_obs_shape
    )
    
    return wrapped_env


def create_experiment_name(args):
    """
    Create a descriptive experiment name based on the configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        str: Experiment name
    """
    components = ["demosac"]
    
    # Add network type
    components.append(args.network)
    
    # Add demo info
    if not args.no_demos:
        components.append("demo")
    
    # Add RND info
    if not args.no_rnd:
        components.append(f"rnd{args.rnd_coef}")
    
    # Add custom wrapper info
    if args.use_custom_wrappers:
        wrapper_name = "custom"
        if args.reward_shaping:
            wrapper_name += "_shaped"
        components.append(wrapper_name)
    
    # Add memory efficiency info
    if args.memory_efficient:
        components.append("memeff")
    
    # Join components
    name = "_".join(components)
    
    # Add timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    return f"{name}_{timestamp}"

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
        """
        Look up the actual action tuple from the flattened action index.
        
        Args:
            action: The flattened action index
            
        Returns:
            The corresponding action tuple
        """
        try:
            # Handle every possible input type
            
            # Case 1: action is a numpy array
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    # Single element array
                    action = action.item()
                elif action.size > 1:
                    # Multi-element array, take the first element
                    action = int(action.flat[0])
                else:
                    # Empty array, default to action 0
                    action = 0
                    
            # Case 2: action is a torch tensor
            elif hasattr(action, 'dim') and callable(getattr(action, 'dim')):  # Check if it's a torch tensor
                try:
                    if action.numel() == 1:
                        # Single element tensor
                        action = action.item()
                    else:
                        # Multi-element tensor
                        action = action.view(-1)[0].item()
                except Exception as e:
                    # If item() fails, convert to int directly
                    try:
                        action = int(action.detach().cpu().numpy().flat[0])
                    except:
                        # Emergency fallback
                        action = 0
                        print(f"Warning: Could not convert tensor to action index: {e}, using default action 0")
            
            # Case 3: action is a list or tuple
            elif isinstance(action, (list, tuple)):
                if len(action) > 0:
                    # Take the first element
                    action = int(action[0])
                else:
                    # Empty list, default to action 0
                    action = 0
            
            # Case 4: try to convert to int (handles float, bool, etc.)
            else:
                try:
                    action = int(action)
                except:
                    # If all else fails, default to action 0
                    action = 0
                    print(f"Warning: Could not convert {action} to int, using default action 0")
            
            # Now perform the lookup with a proper hashable key
            if action in self.action_lookup:
                return self.action_lookup[action]
            else:
                # Handle out-of-bounds index (use modulo to wrap around)
                valid_action = action % len(self.action_lookup)
                print(f"Warning: Action index {action} out of bounds, using {valid_action} instead")
                return self.action_lookup[valid_action]
                
        except Exception as e:
            # Global emergency fallback - return the first action
            print(f"Error in lookup_action: {e}, returning default action")
            return self.action_lookup[0]
        
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