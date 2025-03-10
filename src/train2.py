import gym
from obstacle_tower_env import ObstacleTowerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
from PIL import Image
import argparse
import torch

# Custom wrapper to preprocess observations and flatten for MlpPolicy
class PreprocessObstacleTower(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3 * 84 * 84,), dtype=np.float32
        )

    def reset(self):
        obs = self.env.reset()
        return self._preprocess(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._preprocess(obs), reward, done, info

    def _preprocess(self, obs):
        visual_obs = obs[0]
        if visual_obs.ndim == 3 and visual_obs.shape[0] == 1 and visual_obs.shape[1] == 1:
            visual_obs = visual_obs.squeeze()
            raise ValueError("Unexpected observation shape after squeeze: expected image, got vector")
        if visual_obs.dtype != np.uint8:
            visual_obs = (visual_obs * 255).astype(np.uint8) if visual_obs.max() <= 1.0 else visual_obs.astype(np.uint8)
        visual_obs = np.array(Image.fromarray(visual_obs).resize((84, 84), Image.NEAREST))
        visual_obs = visual_obs.flatten()
        return visual_obs.astype(np.float32) / 255.0

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for Obstacle Tower with Stable-Baselines3 MlpPolicy")
    parser.add_argument('--realtime', action='store_true', help="Enable real-time mode for visualization")
    args = parser.parse_args()

    def make_env():
        env = ObstacleTowerEnv('./ObstacleTower/obstacletower', retro=False, realtime_mode=args.realtime)
        env = PreprocessObstacleTower(env)
        return env

    env = make_vec_env(make_env, n_envs=1)

    # PPO setup with MlpPolicy and TensorBoard logging
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tensorboard_logs/"  # Log to this directory
    )

    # Training
    max_steps = 1000000  # Reduce to 10k for quick demo if needed
    model.learn(total_timesteps=max_steps, log_interval=10)

    model.save("logs/ppo_obstacle_tower_mlp")
    env.close()

if __name__ == "__main__":
    main()