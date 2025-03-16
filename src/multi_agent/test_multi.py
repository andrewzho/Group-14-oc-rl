import os
import sys
import argparse
import time
import logging
import numpy as np
import torch
import platform
import signal
import subprocess
import random

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from multi_agent
from multi_agent.env_wrapper import make_vec_envs, find_and_kill_unity_processes
from src.model import RecurrentPPONetwork
from multi_agent.train_multi import preprocess_obs_batch

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_multi")

def ensure_clean_environment():
    """Make sure we don't have any stale Unity processes"""
    logger.info("Cleaning up any stale Unity processes...")
    find_and_kill_unity_processes()
    # Give system time to release resources
    time.sleep(3)
    logger.info("Cleanup complete")

def test_env_creation(env_path, num_envs=2, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Test environment creation"""
    logger.info(f"Testing environment creation with {num_envs} envs")
    
    # Ensure a clean environment first
    ensure_clean_environment()
    
    # Create environment
    try:
        start_time = time.time()
        envs = make_vec_envs(
            executable_path=env_path, 
            num_envs=num_envs,
            seed=random.randint(0, 1000)  # Random seed each time
        )
        logger.info(f"Environments created in {time.time() - start_time:.2f} seconds")
        
        # Reset environments
        logger.info("Resetting environments")
        obs = envs.reset()
        logger.info(f"Reset completed, observation shape: {np.array(obs).shape}")
        
        # Process observation
        obs_tensor = preprocess_obs_batch(obs, device)
        logger.info(f"Observation tensor shape: {obs_tensor.shape}")
        
        # Test a few steps
        for i in range(5):
            logger.info(f"Taking step {i+1}")
            actions = [0] * num_envs  # Use no-op action
            next_obs, rewards, dones, infos = envs.step(actions)
            logger.info(f"Step completed, rewards: {rewards}, dones: {dones}")
            obs = next_obs
        
        # Close environments
        logger.info("Closing environments...")
        envs.close()
        
        # Clean up after test
        ensure_clean_environment()
        
        logger.info("Test successful - environments work properly")
        return True
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Try to clean up if there was an error
        logger.info("Attempting cleanup after failure...")
        try:
            if 'envs' in locals():
                envs.close()
        except:
            pass
            
        ensure_clean_environment()
        
        return False

def test_model_forward(input_shape=(12, 84, 84), num_actions=8, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Test model forward pass"""
    logger.info("Testing model forward pass")
    
    # Create model
    model = RecurrentPPONetwork(input_shape=input_shape, 
                              num_actions=num_actions, 
                              use_lstm=True).to(device)
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Initialize LSTM states
    lstm_states = model.init_lstm_state(batch_size=batch_size, device=device)
    
    # Create episode_starts
    episode_starts = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # Forward pass
    try:
        logger.info("Running forward pass")
        policy_logits, values, next_lstm_states = model(dummy_input, lstm_states, episode_starts)
        
        logger.info(f"Forward pass successful")
        logger.info(f"Policy logits shape: {policy_logits.shape}")
        logger.info(f"Values shape: {values.shape}")
        logger.info(f"LSTM states shape: {next_lstm_states[0].shape}, {next_lstm_states[1].shape}")
        
        return True
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run tests"""
    parser = argparse.ArgumentParser(description="Test multi-agent Obstacle Tower")
    parser.add_argument("--env_path", type=str, default="ObstacleTower/obstacletower.exe",
                       help="Path to Obstacle Tower executable")
    parser.add_argument("--num_envs", type=int, default=2,
                       help="Number of environments to test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run tests on")
    
    args = parser.parse_args()
    
    # Initial cleanup
    ensure_clean_environment()
    
    # Test model forward pass first (doesn't require env)
    model_test_passed = test_model_forward(device=args.device)
    
    # Test environment creation with a smaller number to be safer
    env_test_passed = test_env_creation(
        env_path=args.env_path,
        num_envs=min(args.num_envs, 2),  # Test with at most 2 envs to be safe
        device=args.device
    )
    
    # Summary
    if model_test_passed and env_test_passed:
        logger.info("All tests passed!")
    else:
        logger.warning("Some tests failed. Check logs for details.")
        
    # Final cleanup
    ensure_clean_environment()

if __name__ == "__main__":
    main() 