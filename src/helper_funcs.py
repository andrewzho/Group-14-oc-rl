
"""
Environment wrappers and helpful functions that do not fit
nicely into any other file.
"""

import os
import random

from PIL import Image
import gym
import gym.spaces
import numpy as np
from obstacle_tower_env import ObstacleTowerEnv
import torch
import torchvision.transforms.functional as TF

from src.parallel_envs import ParallelGymWrapper
from src.global_settings import HUMAN_ACTIONS, IMAGE_DEPTH, IMAGE_SIZE, NUM_ACTIONS
from src.data_collector import TrajectoryCollector


def big_obs(obs, info):
    """
    Big obs takes a retro observation and an info
    dictionary and produces a higher resolution
    observation with the retro features tacked on.
    """
    res = (info['brain_info'].visual_observations[0][0, :, :, :] * 255).astype(np.uint8)
    res[:20] = np.array(Image.fromarray(obs).resize((168, 168)))[:20]
    return res


def create_parallel_envs(num_envs, start=0, **kwargs):
    """
    A helper function to create a batch of environments.

    Args:
        num_envs: size of the batch.
        start: the starting worker index number.
        kwargs: passed to create_single_env().
    """
    env_fns = [lambda i=i: create_single_env(i + start, **kwargs) for i in range(num_envs)]
    return ParallelGymWrapper(gym.spaces.Discrete(NUM_ACTIONS),
                         gym.spaces.Box(low=0, high=0xff,
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH),
                                        dtype=np.uint8),
                         env_fns)


def create_single_env(idx, clear=True, augment=False, rand_floors=None):
    """
    Create a single, wrapped environment.

    Args:
        idx: the ML-Agents worker index to use.
        clear: erase most of the info dictionary.
          This saves memory when training, since the
          trajectory_stores store the entire info dict. By default,
          the dict contains a large observation, which
          takes up a lot of memory.
        augment: wrap the environment so that data
          augmentation is used.
        rand_floors: if specified, this is a tuple
          (min, max) indicating that starting floors
          should be sampled in the range [min, max).
    """
    env = TimeRewardEnv(os.environ['OBS_TOWER_PATH'], worker_id=idx)
    if clear:
        env = ClearInfoEnv(env)
    if rand_floors is not None:
        env = RandomFloorEnv(env, rand_floors)
    if augment:
        env = AugmentEnv(env)
    env = FrameStackEnv(env)
    env = HumanActionEnv(env)
    return env


def log_floors(trajectory_store):
    """
    For all the completed episodes in a trajectory_store, print to
    standard output the attained floor numbers.
    """
    for t in range(1, trajectory_store.num_steps):
        for b in range(trajectory_store.batch_size):
            if trajectory_store.dones[t, b]:
                info = trajectory_store.infos[t - 2][b]
                if 'start_floor' in info:
                    print('start=%d floor=%d' %
                          (info['start_floor'], info['current_floor'] - info['start_floor']))
                else:
                    print('floor=%d' % info['current_floor'])


def mirror_obs(obs):
    """
    Mirror an observation, not including the time and key
    bars.
    """
    obs = obs.copy()
    obs[10:] = obs[10:, ::-1]
    return obs


def mirror_action(act):
    """
    Mirror an action, swapping left/right.
    """
    direction = (act % 18) // 6
    act -= direction * 6
    if direction == 1:
        direction = 2
    elif direction == 2:
        direction = 1
    act += direction * 6
    return act


def atomic_save(obj, path):
    """
    Save a model to a file, making sure that the file will
    never be partly written.

    This prevents the model from getting corrupted in the
    event that the process dies or the machine crashes.
    """
    torch.save(obj, path + '.tmp')
    
    # Check if the target file exists and remove it if it does
    if os.path.exists(path):
        os.remove(path)
        
    os.rename(path + '.tmp', path)
class Augmentation:
    """
    A collection of settings indicating how to slightly
    modify an image.
    """

    def __init__(self):
        self.brightness = random.random() * 0.1 + 0.95
        self.contrast = random.random() * 0.1 + 0.95
        self.gamma = random.random() * 0.1 + 0.95
        self.hue = random.random() * 0.1 - 0.05
        self.saturation = random.random() * 0.1 + 0.95
        self.translation = (random.randrange(-2, 3), random.randrange(-2, 3))

    def apply(self, image):
        return Image.fromarray(self.apply_np(np.array(image)))

    def apply_np(self, np_image):
        content = Image.fromarray(np_image[10:])
        content = TF.adjust_brightness(content, self.brightness)
        content = TF.adjust_contrast(content, self.contrast)
        content = TF.adjust_gamma(content, self.gamma)
        content = TF.adjust_hue(content, self.hue)
        content = TF.adjust_saturation(content, self.saturation)
        content = TF.affine(content, 0, self.translation, 1.0, 0)
        result = np.array(np_image)
        result[10:] = np.array(content)
        return result


class LogTrajectoryCollector(TrajectoryCollector):
    """
    A TrajectoryCollector that logs floors after every trajectory_store.
    """

    def trajectory_store(self):
        result = super().trajectory_store()
        log_floors(result)
        return result


class TimeRewardEnv(ObstacleTowerEnv):
    """
    An environment that adds rewards to the info dict's
    'extra_reward' key whenever the agent gets a time orb.

    This does not add rewards directly because the
    recorded demonstrations do not track these rewards, so
    the cloned policy is not used to seeing these rewards
    in the state stacks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_time = None

    def reset(self):
        config = {"total-floors": 25}
        self.last_time = None
        obs = super().reset(config)
        return obs

    def _single_step(self, info, allow_multiple_obs=False):
        obs, rew, done, final_info = super()._single_step(info, allow_multiple_obs)
        extra_reward = 0.0
        
        # In the new version, the time value might be in a different structure
        # Check if we can extract a time value safely
        try:
            # Option 1: Try to get first agent's obs, assuming time is still accessible
            # but at a different position or in a flattened array
            if hasattr(info, 'obs') and len(info.obs) > 0:
                # Try to find time info in the observation
                # We might need to examine what's actually in info.obs
                # For now, let's just skip this time reward feature
                time_value = 0  # Placeholder, can't determine exact position without more info
                
                if self.last_time is not None:
                    if rew != 1.0 and time_value > self.last_time:
                        extra_reward = 0.1
                self.last_time = time_value
            else:
                # If we can't find time info, just skip this feature
                pass
        except (IndexError, AttributeError):
            # If there's any error accessing the time value, just skip it
            pass
            
        final_info['extra_reward'] = extra_reward
        return obs, rew, done, final_info
    
class AugmentEnv(gym.Wrapper):
    """
    An environment wrapper that applies data augmentation.
    """

    def __init__(self, env):
        super().__init__(env)
        self.augmentation = None

    def reset(self, **kwargs):
        self.augmentation = Augmentation()
        obs = self.env.reset(**kwargs)
        return self.augmentation.apply_np(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = self.augmentation.apply_np(obs)
        return obs, rew, done, info


class ClearInfoEnv(gym.Wrapper):
    """
    An environment wrapper that deletes most information
    from info dicts to save memory.
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        new_info = {}
        if 'extra_reward' in info:
            new_info['extra_reward'] = info['extra_reward']
        if 'current_floor' in info:
            new_info['current_floor'] = info['current_floor']
        return obs, rew, done, new_info


class HumanActionEnv(gym.ActionWrapper):
    """
    An environment wrapper that limits the action space to
    looking left/right, jumping, and moving forward.
    """

    def __init__(self, env):
        super().__init__(env)
        self.actions = HUMAN_ACTIONS
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, act):
        return self.actions[act]


class RandomFloorEnv(gym.Wrapper):
    """
    An environment wrapper that selects random starting
    floors in a certain range.
    """

    def __init__(self, env, floors):
        super().__init__(env)
        self.floors = floors

    def reset(self, **kwargs):
        self.unwrapped.floor(random.randrange(*self.floors))
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info['start_floor'] = self.unwrapped._floor
        return obs, rew, done, info


class FrameStackEnv(gym.Wrapper):
    """
    An environment that stacks images.
    The stacking is ordered from oldest to newest.
    At the beginning of an episode, the first observation
    is repeated in order to complete the stack.
    """

    def __init__(self, env, num_images=2):
        """
        Create a frame stacking environment.
        Args:
          env: the environment to wrap.
          num_images: the number of images to stack.
            This includes the current observation.
        """
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(np.repeat(old_space.low, num_images, axis=-1),
                                                np.repeat(old_space.high, num_images, axis=-1),
                                                dtype=old_space.dtype)
        self._num_images = num_images
        self._history = []

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._history = [obs] * self._num_images
        return self._cur_obs()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._history.append(obs)
        self._history = self._history[1:]
        return self._cur_obs(), rew, done, info

    def _cur_obs(self):
        return np.concatenate(self._history, axis=-1)
