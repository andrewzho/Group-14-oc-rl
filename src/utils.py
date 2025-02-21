import numpy as np
import torch
import gym
import itertools

def normalize(x):
    x = np.array(x)
    return (x - x.mean()) / (x.std() + 1e-8)

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"Saved checkpoint to {path}")

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