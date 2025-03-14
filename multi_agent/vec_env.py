import numpy as np
import torch
import multiprocessing as mp
from collections import deque
import time
import os
import gym
from typing import List, Tuple, Dict, Any, Optional, Union

# For type hints
from numpy.typing import NDArray
from torch import Tensor

class SubprocVecEnv:
    """
    Vectorized environment that runs multiple environments in parallel processes.
    Each environment runs in a separate process to avoid the GIL bottleneck.
    
    This implementation is specifically tailored for the Obstacle Tower environment.
    """
    
    def __init__(self, env_fns, start_method=None):
        """
        Initialize a vectorized environment with multiple subprocesses.
        
        Args:
            env_fns: List of functions, each creating one environment
            start_method: Process start method ('fork', 'spawn', or 'forkserver')
        """
        self.waiting = False
        self.closed = False
        
        if start_method is None:
            # Use 'spawn' for Windows, 'fork' is typically faster on Linux/Unix
            start_method = 'spawn' if os.name == 'nt' else 'fork'
        
        ctx = mp.get_context(start_method)
        
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, env_fn)
            # Daemon=True means the process will terminate if the parent terminates
            process = ctx.Process(target=self._worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
            
        # Get information about the first environment as a reference
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
    @staticmethod
    def _worker(remote, parent_remote, env_fn):
        """
        Worker function run in a subprocess to handle environment interactions.
        
        Args:
            remote: Pipe for communicating with the main process
            parent_remote: Pipe from the main process (to be closed by worker)
            env_fn: Function that creates and returns an environment
        """
        parent_remote.close()
        env = env_fn()
        
        try:
            while True:
                cmd, data = remote.recv()
                
                if cmd == 'step':
                    try:
                        obs, reward, done, info = env.step(data)
                        if done:
                            # Save info about the episode that just ended
                            info['terminal_observation'] = obs
                            # Automatically reset when done
                            obs = env.reset()
                        remote.send((obs, reward, done, info))
                    except Exception as e:
                        # If step fails, try to recreate the environment
                        remote.send(('error', str(e)))
                        # Try to recreate environment
                        try:
                            env.close()
                            env = env_fn()
                            obs = env.reset()
                            remote.send(('reset_success', obs))
                        except Exception as e2:
                            remote.send(('fatal_error', str(e2)))
                        
                elif cmd == 'reset':
                    try:
                        obs = env.reset()
                        remote.send((obs))
                    except Exception as e:
                        # If reset fails, try to recreate the environment
                        remote.send(('error', str(e)))
                        try:
                            env.close()
                            env = env_fn()
                            obs = env.reset()
                            remote.send(('reset_success', obs))
                        except Exception as e2:
                            remote.send(('fatal_error', str(e2)))
                
                elif cmd == 'seed':
                    env.seed(data)
                    remote.send(None)
                    
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                    
                elif cmd == 'get_spaces':
                    remote.send((env.observation_space, env.action_space))
                    
                elif cmd == 'floor':
                    # Specific to Obstacle Tower - change floor
                    if hasattr(env, 'floor'):
                        env.floor(data)
                        remote.send(True)
                    else:
                        remote.send(False)
                
                else:
                    raise NotImplementedError(f"Command {cmd} not implemented")
                    
        except KeyboardInterrupt:
            print("Worker process received KeyboardInterrupt")
        finally:
            env.close()
            remote.close()
            
    def step(self, actions):
        """
        Step all environments with the given actions.
        
        Args:
            actions: List of actions to take in each environment
            
        Returns:
            observations, rewards, dones, infos for all environments
        """
        self._assert_not_closed()
        
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
            
        self.waiting = True
        results = []
        
        # Collect results from all environments
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
                    print(f"Error in env {i}: {result[1]}")
                    # Handle environment recreation
                    result = remote.recv()
                    if result[0] == 'reset_success':
                        print(f"Environment {i} successfully reset")
                        results.append((result[1], 0.0, True, {'error': True}))
                    else:
                        print(f"Fatal error in env {i}: {result[1]}")
                        results.append((None, 0.0, True, {'fatal_error': True}))
                else:
                    results.append(result)
            except Exception as e:
                print(f"Error receiving from env {i}: {e}")
                results.append((None, 0.0, True, {'communication_error': True}))
                
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        
        return obs, np.array(rews), np.array(dones), infos
        
    def reset(self):
        """Reset all environments and return initial observations."""
        self._assert_not_closed()
        
        for remote in self.remotes:
            remote.send(('reset', None))
            
        results = []
        for i, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                if isinstance(result, tuple) and len(result) == 2 and result[0] == 'error':
                    print(f"Error resetting env {i}: {result[1]}")
                    # Handle environment recreation
                    result = remote.recv()
                    if result[0] == 'reset_success':
                        print(f"Environment {i} successfully recreated and reset")
                        results.append(result[1])
                    else:
                        print(f"Fatal error recreating env {i}: {result[1]}")
                        # Return zeros as a placeholder
                        results.append(None)
                else:
                    results.append(result)
            except Exception as e:
                print(f"Error receiving from env {i}: {e}")
                # Return zeros as a placeholder
                results.append(None)
        
        return results
        
    def seed(self, seeds):
        """Set seeds for all environments."""
        self._assert_not_closed()
        
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('seed', seed))
            
        for remote in self.remotes:
            remote.recv()
            
    def close(self):
        """Close all environments and terminate worker processes."""
        if self.closed:
            return
            
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
                
        for remote in self.remotes:
            remote.send(('close', None))
            
        for process in self.processes:
            process.join(timeout=5.0)  # Give processes 5 seconds to terminate
            
        self.closed = True
        
    def set_floor(self, floors):
        """Set the floor for each environment (Obstacle Tower specific)."""
        self._assert_not_closed()
        
        if isinstance(floors, int):
            floors = [floors] * self.num_envs
            
        for remote, floor in zip(self.remotes, floors):
            remote.send(('floor', floor))
            
        return [remote.recv() for remote in self.remotes]
        
    def _assert_not_closed(self):
        """Make sure the environments are still running."""
        if self.closed:
            raise RuntimeError("Tried to interact with closed environments")


class ObstacleTowerVecEnv(SubprocVecEnv):
    """
    Specialized vectorized environment for Obstacle Tower with additional
    utilities for frame stacking and preprocessing.
    """
    
    def __init__(self, env_fns, start_method=None, frame_stack=4):
        """
        Initialize the Obstacle Tower vectorized environment.
        
        Args:
            env_fns: List of functions, each creating one environment
            start_method: Process start method
            frame_stack: Number of frames to stack for each observation
        """
        super().__init__(env_fns, start_method)
        self.frame_stack = frame_stack
        self.stacked_obs = None
        
    def reset(self):
        """
        Reset all environments and stack frames for initial observation.
        
        Returns:
            Stacked observations from all environments
        """
        obs = super().reset()
        
        # Initialize the stacked observations
        self.stacked_obs = []
        
        for i in range(self.num_envs):
            # For each environment, create a deque to hold stacked frames
            if obs[i] is not None:
                try:
                    # Process observation - convert from [H,W,C] to [C,H,W] and normalize
                    processed = np.transpose(obs[i][0], (2, 0, 1)) / 255.0
                    frames = deque([processed] * self.frame_stack, maxlen=self.frame_stack)
                    self.stacked_obs.append(frames)
                except (IndexError, TypeError) as e:
                    # Handle malformed observations
                    print(f"Error processing observation from env {i}: {e}")
                    self.stacked_obs.append(None)
            else:
                # If observation is None (e.g., due to environment error), use None
                self.stacked_obs.append(None)
        
        # Return stacked observations
        return self._get_stacked_obs()
        
    def step(self, actions):
        """
        Step all environments with the given actions and update frame stacks.
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            stacked_obs, rewards, dones, infos
        """
        obs, rewards, dones, infos = super().step(actions)
        
        # Update frame stacks and create stacked observations
        for i, (ob, done) in enumerate(zip(obs, dones)):
            # Skip environments with None observations
            if ob is None:
                continue
                
            # Create new frame stack if it doesn't exist yet
            if self.stacked_obs[i] is None:
                try:
                    processed = np.transpose(ob[0], (2, 0, 1)) / 255.0
                    self.stacked_obs[i] = deque([processed] * self.frame_stack, maxlen=self.frame_stack)
                except (IndexError, TypeError):
                    # Still can't process, keep as None
                    continue
            else:
                # Add new frame to existing stack
                try:
                    processed = np.transpose(ob[0], (2, 0, 1)) / 255.0
                    self.stacked_obs[i].append(processed)
                except (IndexError, TypeError) as e:
                    # Handle malformed observation
                    print(f"Error updating frame stack for env {i}: {e}")
                    # Do not modify the stack if the frame is invalid
        
        # Return stacked observations along with other step results
        return self._get_stacked_obs(), rewards, dones, infos
    
    def _get_stacked_obs(self):
        """
        Internal method to safely get the current stacked observations,
        handling None values properly.
        
        Returns:
            List of stacked observations, with zeros for environments with errors
        """
        stacked = []
        for i in range(self.num_envs):
            if self.stacked_obs[i] is not None:
                try:
                    # Stack the frames along the channel dimension
                    stacked.append(np.concatenate(list(self.stacked_obs[i]), axis=0))
                except Exception as e:
                    # If stacking fails, use zeros
                    print(f"Error stacking frames for env {i}: {e}")
                    stacked.append(np.zeros((3 * self.frame_stack, 84, 84), dtype=np.float32))
            else:
                # Use zeros for environments with errors
                stacked.append(np.zeros((3 * self.frame_stack, 84, 84), dtype=np.float32))
        return stacked
        
    def get_stacked_obs(self):
        """Get the current stacked observations."""
        if self.stacked_obs is None:
            raise RuntimeError("Observations not initialized. Call reset() first.")
        return self._get_stacked_obs()
                
    def to_tensor(self, device="cpu"):
        """
        Convert stacked observations to a PyTorch tensor.
        
        Args:
            device: Device to place the tensor on
            
        Returns:
            Tensor of stacked observations
        """
        stacked_obs = self.get_stacked_obs()
        return torch.tensor(np.array(stacked_obs), dtype=torch.float32, device=device) 