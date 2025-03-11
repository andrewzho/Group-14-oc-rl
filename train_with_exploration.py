#!/usr/bin/env python3
"""
Script to train an enhanced PPO agent with exploration on the Obstacle Tower environment.
Focuses on better exploration to solve key-door puzzles and reach higher floors.
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start training with enhanced exploration')
    parser.add_argument('--high_entropy', action='store_true', 
                       help='Use higher entropy regularization')
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--mastery_threshold', type=int, default=5,
                       help='Number of consecutive completions required to advance to next floor')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment during training')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--steps', type=int, default=5000000,
                       help='Number of training steps')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for logs and checkpoints')
    parser.add_argument('--stable', action='store_true',
                       help='Use more conservative training settings for stability')
    args = parser.parse_args()
    
    # Build command with appropriate parameters
    cmd = ["python", "-m", "src.train_exploration"]
    
    # Training parameters
    cmd.extend(["--num_steps", str(args.steps)])
    cmd.extend(["--horizon", "2048"])  # PPO horizon
    
    # Algorithm parameters - more conservative if stable flag is used
    cmd.extend(["--use_icm"])  # Use Intrinsic Curiosity Module
    cmd.extend(["--gamma", "0.99"])
    
    if args.stable:
        # More conservative settings for stability
        cmd.extend(["--lr", "1e-4"])  # Lower learning rate
        cmd.extend(["--batch_size", "32"])  # Smaller batch size
        cmd.extend(["--ppo_epochs", "3"])  # Fewer PPO epochs
        cmd.extend(["--clip_eps", "0.1"])  # Smaller clipping epsilon
        cmd.extend(["--max_grad_norm", "0.25"])  # Stronger gradient clipping
        cmd.extend(["--intrinsic_coef", "0.1"])  # Lower intrinsic reward coefficient
    else:
        # Standard settings
        cmd.extend(["--lr", "3e-4"])
        cmd.extend(["--batch_size", "64"])
        cmd.extend(["--ppo_epochs", "4"])
    
    # Exploration parameters
    if args.high_entropy:
        entropy_val = "0.05" if not args.stable else "0.03"
        intrinsic_val = "1.0" if not args.stable else "0.5"
        cmd.extend(["--entropy_reg", entropy_val])  # Higher entropy for more exploration
        cmd.extend(["--intrinsic_coef", intrinsic_val])  # Stronger intrinsic rewards
    else:
        entropy_val = "0.03" if not args.stable else "0.02"
        intrinsic_val = "0.5" if not args.stable else "0.2"
        cmd.extend(["--entropy_reg", entropy_val])  # Default entropy
        cmd.extend(["--intrinsic_coef", intrinsic_val])  # Default intrinsic rewards
    
    # Reward shaping
    cmd.extend(["--key_reward", "1.0"])  # Bonus for finding a key
    cmd.extend(["--door_reward", "2.0"])  # Bonus for opening a door
    cmd.extend(["--floor_reward", "10.0"])  # Bonus for reaching new floor
    
    # Curriculum learning
    if args.curriculum:
        cmd.append("--curriculum")
        cmd.extend(["--starting_floor", "0"])
        cmd.extend(["--max_starting_floor", "10"])
        cmd.extend(["--mastery_threshold", str(args.mastery_threshold)])
    
    # Visualization
    if args.render:
        cmd.append("--render")
    
    # Checkpoint for resuming
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    
    # Logging directory
    if args.log_dir:
        cmd.extend(["--log_dir", args.log_dir])
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    
    # Execute the command
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main() 