"""
Configuration settings for Obstacle Tower Challenge agent.
This file centralizes all hyperparameters and settings.
"""
import os
from pathlib import Path

# Base project directories
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = BASE_DIR / "demonstrations"
LOG_DIR = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DEMO_DIR, LOG_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Environment settings
ENV_CONFIG = {
    "exec_path": None,  # Set to None to use gym environment
    "worker_id": 1,
    "retro": False,
    "realtime_mode": False,
    "timeout_wait": 60,
    "docker_training": False,
    "resolution": (84, 84),  # Resized observation shape
    "stack_frames": 4,  # Number of frames to stack
    "grayscale": True,  # Convert to grayscale
    "seed": 42,
}

# SAC hyperparameters
SAC_CONFIG = {
    "policy_type": "CnnPolicy",
    "learning_rate": 1e-3,  # Increased from 3e-4 to 1e-3 for faster learning
    "buffer_size": 1000000,
    "learning_starts": 10000,  # Reduced to start learning earlier
    "batch_size": 512,  # Increased from 256 to 512 for better stability
    "tau": 0.01,  # Increased from 0.005 for faster target network updates
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 2,  # Increased from 1 to 2 for more training per step
    "ent_coef": "auto",
    "target_update_interval": 1,
    "target_entropy": "auto",
    "use_sde": False,
    "sde_sample_freq": -1,
    "use_sde_at_warmup": False,
    "tensorboard_log": str(LOG_DIR),
    "verbose": 1,
}

# Random Network Distillation parameters
RND_CONFIG = {
    "feature_dim": 512,  # Output dimension of RND networks
    "learning_rate": 1e-4,
    "intrinsic_reward_coef": 0.1,  # Weight for intrinsic rewards
    "normalization": True,  # Normalize observations
    "update_proportion": 0.25,  # Proportion of steps to update RND
}

# Demonstration learning parameters
DEMO_CONFIG = {
    "demo_file": str(DEMO_DIR / "demonstrations.pkl"),
    "use_demonstrations": True,
    "demo_batch_size": 64,
    "bc_loss_coef": 0.5,  # Weight for behavior cloning loss
    "pretrain_steps": 10000,  # Number of pretraining steps on demonstrations
}

# Training parameters
TRAIN_CONFIG = {
    "total_timesteps": 5000000,
    "eval_freq": 10000,  # Evaluation frequency in timesteps
    "save_freq": 100000,  # Model saving frequency in timesteps
    "log_interval": 100,  # Logging interval in timesteps
    "max_episode_steps": 1000,  # Max episode length
    "n_envs": 1,  # Number of parallel environments
}

# Evaluation parameters
EVAL_CONFIG = {
    "n_eval_episodes": 10,
    "deterministic": True,
    "render": False,
}

# Neural network architecture
NETWORK_CONFIG = {
    "cnn_features": [
        {"in_channels": 4, "out_channels": 64, "kernel_size": 8, "stride": 4},  # Increased channels
        {"in_channels": 64, "out_channels": 128, "kernel_size": 4, "stride": 2},  # Increased channels
        {"in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1},  # Increased channels
    ],
    "hidden_dim": 1024,  # Increased from 512 to 1024
    "latent_dim": 512,  # Increased from 256 to 512
}