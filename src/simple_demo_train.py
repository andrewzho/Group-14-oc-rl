import os
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from src.envs.memory_efficient_wrapper import create_memory_efficient_env
from config import *

# Create timestamp for the run
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"sac_minimal_{timestamp}"
log_path = os.path.join(LOG_DIR, run_name)
model_path = os.path.join(MODEL_DIR, run_name)
os.makedirs(log_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

print(f"Starting training run: {run_name}")
print(f"Log path: {log_path}")

# Set random seeds
np.random.seed(ENV_CONFIG["seed"])
torch.manual_seed(ENV_CONFIG["seed"])

# Create environment
def make_env(env_path, worker_id=1):
    def _init():
        env = create_memory_efficient_env(
            env_path=env_path, 
            worker_id=worker_id,
            stack_frames=ENV_CONFIG["stack_frames"],
            seed=ENV_CONFIG["seed"]
        )
        return env
    return _init

# Create and vectorize the environment
env_path = "ObstacleTower/ObstacleTower.exe"  # Update to your path
env = DummyVecEnv([make_env(env_path, worker_id=1)])

# Configure policy to work with channel-first images
policy_kwargs = {
    "features_extractor_kwargs": {"features_dim": 512},
    "normalize_images": False  # IMPORTANT: This tells SB3 we're using normalized channel-first images
}

# Create the model
model = SAC(
    policy="CnnPolicy",
    env=env,
    learning_rate=SAC_CONFIG["learning_rate"],
    buffer_size=SAC_CONFIG["buffer_size"],
    learning_starts=SAC_CONFIG["learning_starts"],
    batch_size=SAC_CONFIG["batch_size"],
    tau=SAC_CONFIG["tau"],
    gamma=SAC_CONFIG["gamma"],
    train_freq=SAC_CONFIG["train_freq"],
    tensorboard_log=log_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    policy_kwargs=policy_kwargs  # Add the policy kwargs
)

# Train the model
print(f"Starting training for {TRAIN_CONFIG['total_timesteps']} timesteps...")
model.learn(total_timesteps=TRAIN_CONFIG["total_timesteps"])

# Save checkpoints
for i in range(1, 11):
    checkpoint = 500000 * i
    if checkpoint <= TRAIN_CONFIG["total_timesteps"]:
        checkpoint_path = os.path.join(model_path, f"model_{checkpoint}")
        model.save(checkpoint_path)
        print(f"Saved checkpoint at {checkpoint} steps")

# Save the final model
final_model_path = os.path.join(model_path, "final_model")
print(f"Saving final model to {final_model_path}")
model.save(final_model_path)

print("Training completed successfully!")