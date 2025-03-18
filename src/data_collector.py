"""
Gathering trajectories from environments.
"""

import numpy as np

from src.trajectory_store import TrajectoryBatch


class TrajectoryCollector:
    """
    A TrajectoryCollector runs a policy on a batched environment and
    produces trajectory_stores containing the results.

    Args:
        parallel_envs: a BatchedEnv implementation.
        model: a Model with 'actions' in its output dicts.
        num_steps: the number of timesteps to run per
          batch of trajectory_stores that are generated.
    """

    def __init__(self, parallel_envs, model, num_steps):
        self.parallel_envs = parallel_envs
        self.model = model
        self.num_steps = num_steps
        self._prev_obs = None
        self._prev_memory_state = None
        self._prev_dones = None

    def reset(self):
        result = self.parallel_envs.reset()
        
        # Check what the reset method is actually returning
        if isinstance(result, tuple) and len(result) == 2:
            # Original behavior - returns (memory_state, obs)
            self._prev_memory_state, self._prev_obs = result
        else:
            # New behavior - might just return observations
            self._prev_obs = result
            # Create dummy memory_state if needed
            self._prev_memory_state = np.zeros([self.parallel_envs.num_envs, self.parallel_envs.observation_space.shape[0]], 
                                        dtype=np.float32)
        
        self._prev_memory_state = np.array(self._prev_memory_state)
        self._prev_obs = np.array(self._prev_obs)
        self._prev_dones = np.zeros([self.parallel_envs.num_envs], dtype=np.bool)

    def trajectory_store(self):
        if self._prev_obs is None:
            self.reset()
        batch = self.parallel_envs.num_envs
        memory_state = np.zeros((self.num_steps + 1,) + self._prev_memory_state.shape, dtype=np.float32)
        obses = np.zeros((self.num_steps + 1,) + self._prev_obs.shape, dtype=self._prev_obs.dtype)
        rews = np.zeros([self.num_steps, batch], dtype=np.float32)
        dones = np.zeros([self.num_steps + 1, batch], dtype=np.bool)
        infos = []
        model_outs = []
        for t in range(self.num_steps):
            memory_state[t] = self._prev_memory_state
            obses[t] = self._prev_obs
            dones[t] = self._prev_dones
            model_out = self.model.step(self._prev_memory_state, self._prev_obs)
            
            # Call step on the environment
            step_result = self.parallel_envs.step(model_out['actions'])
            
            # Handle different return formats from step
            if isinstance(step_result[0], tuple) and len(step_result[0]) == 2:
                # Original format: ((memory_state, obs), rewards, dones, infos)
                (step_memory_state, step_obs), step_rews, step_dones, step_infos = step_result
                self._prev_memory_state = np.array(step_memory_state)
                self._prev_obs = np.array(step_obs)
            else:
                # New format likely: (obs, rewards, dones, infos)
                step_obs, step_rews, step_dones, step_infos = step_result
                self._prev_obs = np.array(step_obs)
                # You may need to create dummy memory_state or extract them from somewhere else
                # For now, keep using the previous memory_state
            
            self._prev_dones = np.array(step_dones)
            rews[t] = np.array(step_rews)
            infos.append(step_infos)
            model_outs.append(model_out)
        
        memory_state[-1] = self._prev_memory_state
        obses[-1] = self._prev_obs
        dones[-1] = self._prev_dones
        model_outs.append(self.model.step(self._prev_memory_state, self._prev_obs))
        return TrajectoryBatch(memory_state, obses, rews, dones, infos, model_outs)
    

class TensorboardTrajectoryCollector(TrajectoryCollector):
    """A TrajectoryCollector that collects floor information for logging."""
    
    def __init__(self, parallel_envs, model, num_steps):
        super().__init__(parallel_envs, model, num_steps)
        self.floor_info = {
            'max_floor': 0,
            'new_floors': 0,
            'episodes': 0
        }
        
    def trajectory_store(self):
        result = super().trajectory_store()
        
        # Reset counters for this trajectory_store
        self.floor_info['new_floors'] = 0
        self.floor_info['episodes'] = 0
        
        # Extract floor information from the trajectory_store
        for t in range(1, result.num_steps):
            for b in range(result.batch_size):
                if result.dones[t, b]:
                    self.floor_info['episodes'] += 1
                    info = result.infos[t-1][b]
                    if 'current_floor' in info:
                        current_floor = info['current_floor']
                        start_floor = info.get('start_floor', 0) 
                        floors_completed = current_floor - start_floor
                        self.floor_info['new_floors'] += floors_completed
                        self.floor_info['max_floor'] = max(self.floor_info['max_floor'], current_floor)
        
        return result