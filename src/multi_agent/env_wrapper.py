import numpy as np
import gym
import os
import random
import time
import signal
import subprocess
import platform
from functools import partial
from obstacle_tower_env import ObstacleTowerEnv
from src.create_env import create_obstacle_tower_env

def find_and_kill_unity_processes():
    """Find and kill stale Unity processes to free up ports"""
    try:
        if platform.system() == "Windows":
            # Use tasklist and taskkill on Windows
            # Find Unity processes
            process = subprocess.Popen(
                "tasklist /FI \"IMAGENAME eq obstacletower*\" /FO CSV /NH", 
                shell=True, 
                stdout=subprocess.PIPE
            )
            stdout, _ = process.communicate()
            
            for line in stdout.decode().splitlines():
                if "obstacletower" in line.lower() or "unity" in line.lower():
                    try:
                        # Extract PID and kill
                        parts = line.split(',')
                        if len(parts) >= 2:
                            pid = parts[1].strip('"')
                            subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                            print(f"Killed stale Unity process with PID {pid}")
                    except:
                        pass
        else:
            # Use ps and kill on Unix-like systems
            process = subprocess.Popen(
                "ps aux | grep -i 'obstacle\\|unity' | grep -v grep", 
                shell=True, 
                stdout=subprocess.PIPE
            )
            stdout, _ = process.communicate()
            
            for line in stdout.decode().splitlines():
                try:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = parts[1]
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"Killed stale Unity process with PID {pid}")
                except:
                    pass
    except Exception as e:
        print(f"Error cleaning up processes: {e}")

class ObstacleTowerWrapper:
    """
    Wrapper for the Obstacle Tower environment that adds useful utilities
    and makes it compatible with the vectorized environment system.
    """
    
    def __init__(self, 
                executable_path, 
                timeout=300, 
                realtime_mode=False,
                config=None,
                retries=5):
        """
        Initialize the wrapper.
        
        Args:
            executable_path: Path to the Obstacle Tower executable
            timeout: Timeout for environment interactions (seconds)
            realtime_mode: Whether to run in realtime or as fast as possible
            config: Configuration dictionary for the environment
            retries: Number of times to retry creating the environment if it fails
        """
        self.executable_path = executable_path
        self.timeout = timeout
        self.realtime_mode = realtime_mode
        self.config = config if config else {}
        self.retries = retries
        
        # Initialize attributes
        self.env = None
        self._previous_keys = None
        self._previous_position = None
        self._previous_obs = None
        self._visited_positions = {}
        self._floor = 0
        
        # Create the environment
        self._create_env()
        
    def _create_env(self):
        """Create the Obstacle Tower environment with retries if needed."""
        last_exception = None
        
        for attempt in range(self.retries):
            try:
                # Try a different worker ID each time
                if 'worker_id' in self.config:
                    # Add some randomness to avoid collisions
                    self.config['worker_id'] += random.randint(1, 1000)
                
                print(f"Creating environment (attempt {attempt+1}/{self.retries}) with worker_id={self.config.get('worker_id', 'default')}")
                
                self.env = create_obstacle_tower_env(
                    executable_path=self.executable_path,
                    timeout=self.timeout,
                    realtime_mode=self.realtime_mode,
                    config=self.config
                )
                return
            except Exception as e:
                last_exception = e
                print(f"Error creating environment (attempt {attempt+1}/{self.retries}): {e}")
                
                # If this is a port collision, try to clean up processes
                if "worker number" in str(e) and "still in use" in str(e):
                    print("Detected port collision. Attempting to clean up processes...")
                    find_and_kill_unity_processes()
                    # Wait a bit for cleanup
                    time.sleep(2)
                    
                # Wait before retrying to allow resources to be freed
                time.sleep(1)
                
        # If we get here, all retries failed
        if last_exception:
            raise RuntimeError(f"Failed to create environment after {self.retries} attempts: {last_exception}")
        else:
            raise RuntimeError(f"Failed to create environment after {self.retries} attempts")
    
    @property
    def observation_space(self):
        """Get the observation space of the environment."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get the action space of the environment."""
        return self.env.action_space
    
    def reset(self):
        """Reset the environment and return the initial observation."""
        obs = self.env.reset()
        
        # Reset tracking variables
        self._previous_keys = None
        self._previous_position = None
        self._previous_obs = None
        self._visited_positions = {}
        
        return obs
    
    def step(self, action):
        """Take a step in the environment."""
        obs, reward, done, info = self.env.step(action)
        
        # Update tracking variables
        if "x_pos" in info:
            current_position = [
                info.get("x_pos", 0),
                info.get("y_pos", 0),
                info.get("z_pos", 0)
            ]
            self._previous_position = current_position
            
        if "total_keys" in info:
            self._previous_keys = info["total_keys"]
            
        if obs is not None:
            self._previous_obs = obs[0] if isinstance(obs, tuple) else obs
        
        # Track visited positions
        if "current_floor" in info and "x_pos" in info and "z_pos" in info:
            position_key = f"{info['current_floor']}_{round(info['x_pos'], 1)}_{round(info['z_pos'], 1)}"
            self._visited_positions[position_key] = self._visited_positions.get(position_key, 0) + 1
            
        return obs, reward, done, info
    
    def close(self):
        """Close the environment."""
        if self.env is not None:
            try:
                self.env.close()
            except Exception as e:
                print(f"Error closing environment: {e}")
    
    def seed(self, seed):
        """Set the random seed for the environment."""
        return self.env.seed(seed)
    
    def floor(self, floor):
        """Set the current floor (Obstacle Tower specific)."""
        if hasattr(self.env, 'floor'):
            self.env.floor(floor)
            self._floor = floor
            return True
        return False
    
    def get_floor(self):
        """Get the current floor."""
        return self._floor
    
    def __del__(self):
        """Clean up when the wrapper is deleted."""
        self.close()

# Define this function at module level (outside of any other function)
def _create_env_instance(executable_path, worker_id, timeout, realtime_mode, config):
    """Create and return a wrapped environment instance."""
    env_config = config.copy() if config else {}
    env_config['worker_id'] = worker_id
    
    return ObstacleTowerWrapper(
        executable_path=executable_path,
        timeout=timeout,
        realtime_mode=realtime_mode,
        config=env_config
    )

def create_env_fn(executable_path, worker_id=0, timeout=300, realtime_mode=False, config=None):
    """
    Create a function that returns a wrapped Obstacle Tower environment.
    
    Args:
        executable_path: Path to the Obstacle Tower executable
        worker_id: Worker ID for parallel environments
        timeout: Timeout for environment interactions (seconds)
        realtime_mode: Whether to run in realtime or as fast as possible
        config: Configuration dictionary for the environment
        
    Returns:
        Function that creates and returns a wrapped environment
    """
    # Use partial to create a function with fixed arguments
    # This is picklable, unlike a nested function
    return partial(_create_env_instance, 
                  executable_path=executable_path,
                  worker_id=worker_id,
                  timeout=timeout,
                  realtime_mode=realtime_mode,
                  config=config)

def make_vec_envs(executable_path, num_envs, seed=0, timeout=300, realtime_mode=False, worker_id_base=None):
    """
    Create multiple Obstacle Tower environments for parallel training.
    
    Args:
        executable_path: Path to the Obstacle Tower executable
        num_envs: Number of parallel environments to create
        seed: Base random seed
        timeout: Timeout for environment interactions (seconds)
        realtime_mode: Whether to run in realtime or as fast as possible
        worker_id_base: Base worker ID to use (will be randomized if None)
        
    Returns:
        ObstacleTowerVecEnv object with multiple environments
    """
    from multi_agent.vec_env import ObstacleTowerVecEnv
    
    # First try to clean up any stale processes
    find_and_kill_unity_processes()
    
    # Wait a bit for cleanup to take effect
    time.sleep(2)
    
    # Create a function for each environment with random worker IDs to avoid collisions
    env_fns = []
    
    # Use provided worker_id_base or generate a random one
    if worker_id_base is None:
        base_worker_id = random.randint(1000, 9000)  # Start with a random base to avoid collisions
    else:
        base_worker_id = worker_id_base
        
    print(f"Using worker ID base: {base_worker_id}")
    
    for i in range(num_envs):
        # Generate a unique worker ID for each environment with good spacing
        worker_id = base_worker_id + i * 100 + random.randint(1, 50)
        
        # Use a simpler way to create environment functions
        env_fn = create_env_fn(
            executable_path=executable_path, 
            worker_id=worker_id, 
            timeout=timeout, 
            realtime_mode=realtime_mode,
            config={'starting-floor': 0, 'total-floors': 10}
        )
        env_fns.append(env_fn)
    
    # Create vectorized environment
    envs = ObstacleTowerVecEnv(env_fns)
    
    # Set seeds
    seeds = [seed + i for i in range(num_envs)]
    envs.seed(seeds)
    
    return envs 