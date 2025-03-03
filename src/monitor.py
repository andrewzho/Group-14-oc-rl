import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.ndimage import gaussian_filter1d

def plot_training_data(log_dir="logs", window_size=10, save_path=None, show=True):
    """Plot the training rewards, episode lengths, and floors reached."""
    rewards_path = os.path.join(log_dir, "rewards.npy")
    lengths_path = os.path.join(log_dir, "lengths.npy")
    floors_path = os.path.join(log_dir, "floors.npy")
    
    if not os.path.exists(rewards_path):
        print(f"No data found at {rewards_path}")
        return
    
    rewards = np.load(rewards_path)
    
    # Load lengths and floors if available
    lengths = np.load(lengths_path) if os.path.exists(lengths_path) else None
    floors = np.load(floors_path) if os.path.exists(floors_path) else None
    
    # Create figure with appropriate number of subplots
    num_plots = 1 + (lengths is not None) + (floors is not None)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]
    
    episodes = np.arange(len(rewards))
    
    # Smooth data for visualization
    smooth_rewards = gaussian_filter1d(rewards, sigma=window_size/3)
    
    # Plot rewards
    ax = axes[0]
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
    ax.plot(episodes, smooth_rewards, color='blue', label=f'Smoothed (window={window_size})')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Training Rewards')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot episode lengths if available
    if lengths is not None:
        smooth_lengths = gaussian_filter1d(lengths, sigma=window_size/3)
        ax = axes[1]
        ax.plot(episodes, lengths, alpha=0.3, color='green', label='Raw')
        ax.plot(episodes, smooth_lengths, color='green', label=f'Smoothed (window={window_size})')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot floors reached if available
    if floors is not None:
        smooth_floors = gaussian_filter1d(floors, sigma=window_size/3)
        ax = axes[-1]
        ax.plot(episodes, floors, alpha=0.3, color='red', label='Raw')
        ax.plot(episodes, smooth_floors, color='red', label=f'Smoothed (window={window_size})')
        ax.set_ylabel('Floor Reached')
        ax.set_title('Max Floor Reached')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add common labels and tight layout
    axes[-1].set_xlabel('Episode')
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Show if requested
    if show:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training data for Obstacle Tower")
    parser.add_argument('--log_dir', type=str, default="logs", help="Directory with logs")
    parser.add_argument('--window', type=int, default=10, help="Smoothing window size")
    parser.add_argument('--save', type=str, default=None, help="Save plot to file")
    parser.add_argument('--no_show', action='store_true', help="Don't show the plot")
    
    args = parser.parse_args()
    plot_training_data(args.log_dir, args.window, args.save, not args.no_show) 