"""
TensorBoard integration for visualizing Obstacle Tower agent performance.
"""

import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import io

class TensorboardVisualizer:
    """
    Class to visualize agent gameplay in TensorBoard.
    """
    
    def __init__(self, log_dir='runs/agent_visualization'):
        """
        Initialize the TensorBoard visualizer.
        
        Args:
            log_dir: Directory for TensorBoard logs.
        """
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.step_counter = 0
        self.episode_counter = 0
        
        # Create a figure for action distribution visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        # Action names for visualization
        self.action_names = {
            0: "No-op",
            3: "Forward",
            6: "Backward",
            9: "Left",
            12: "Right",
            15: "Jump",
            18: "Forward+Jump",
            21: "Backward+Jump",
            24: "Left+Jump",
            27: "Right+Jump",
            30: "Forward+Left",
            33: "Forward+Right"
        }
        
        # Keep track of action distribution
        self.action_counts = {action: 0 for action in self.action_names.keys()}
        
        print(f"TensorBoard visualizer initialized. Logs in {log_dir}")
        print(f"To view, run: tensorboard --logdir={os.path.dirname(log_dir)}")
    
    def log_observation(self, obs, global_step=None):
        """
        Log an observation to TensorBoard.
        
        Args:
            obs: Observation image or array
            global_step: Optional global step for TensorBoard
        """
        if global_step is None:
            global_step = self.step_counter
            self.step_counter += 1
        
        # Handle different observation formats
        if isinstance(obs, tuple) and len(obs) == 2:
            # Handle (memory_state, obs) tuple format
            _, obs_img = obs
        else:
            obs_img = obs
        
        # Convert observation to a format suitable for TensorBoard
        if isinstance(obs_img, np.ndarray):
            # Handle observations with different shapes
            if obs_img.shape[-1] == 6:  # Frame stack with 2 RGB frames
                # Just use the most recent frame (last 3 channels)
                obs_img = obs_img[..., -3:]
            
            # Make sure the image is in uint8 format
            if obs_img.dtype != np.uint8:
                obs_img = (obs_img * 255).astype(np.uint8)
                
            # Convert to PIL Image for processing
            img = Image.fromarray(obs_img)
        else:
            img = obs_img
        
        # Convert to tensor format for TensorBoard (CHW format)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        # Log to TensorBoard
        self.writer.add_image('Observations/Raw', img_tensor, global_step)
    
    def log_action(self, action, global_step=None):
        """
        Log an action to TensorBoard.
        
        Args:
            action: The action taken by the agent
            global_step: Optional global step for TensorBoard
        """
        if global_step is None:
            global_step = self.step_counter
            self.step_counter += 1
        
        # Update action count
        if action in self.action_counts:
            self.action_counts[action] += 1
        
        # Log one-hot action
        action_names = list(self.action_names.keys())
        one_hot = np.zeros(len(action_names))
        if action in action_names:
            one_hot[action_names.index(action)] = 1.0
        
        # Log the one-hot action vector
        self.writer.add_histogram('Actions/Distribution', np.array([action]), global_step)
        
        # Periodically log the action distribution
        if global_step % 100 == 0:
            self._log_action_distribution(global_step)
    
    def log_episode_video(self, frames, episode_id=None, fps=30):
        """
        Log a video of an episode to TensorBoard.
        
        Args:
            frames: List of frames (PIL Images or numpy arrays)
            episode_id: Episode identifier
            fps: Frames per second for the video
        """
        if episode_id is None:
            episode_id = self.episode_counter
            self.episode_counter += 1
        
        # Ensure frames are numpy arrays with the right format
        video_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            # Convert to CHW format for TensorBoard video
            if frame.shape[-1] == 3:  # HWC format
                frame = np.transpose(frame, (2, 0, 1))
            
            # Add batch dimension
            if frame.ndim == 3:
                frame = frame[None, ...]
                
            video_frames.append(frame)
        
        # Concatenate all frames into a single video tensor
        if video_frames:
            video_tensor = np.concatenate(video_frames, axis=0)
            
            # Convert to torch tensor (TensorBoard expects [T, C, H, W] format)
            video_tensor = torch.from_numpy(video_tensor).float() / 255.0
            
            # Add video to TensorBoard
            self.writer.add_video(f'Episodes/episode_{episode_id}', video_tensor[None, ...], fps=fps)
    
    def log_trajectory_metrics(self, trajectory_store, global_step=None):
        """
        Log metrics from a trajectory_store to TensorBoard.
        
        Args:
            trajectory_store: A TrajectoryBatch object
            global_step: Optional global step for TensorBoard
        """
        if global_step is None:
            global_step = self.step_counter
        
        # Log average rewards
        avg_reward = np.mean(trajectory_store.rews)
        self.writer.add_scalar('Trajectory/AvgReward', avg_reward, global_step)
        
        # Log value predictions
        values = trajectory_store.value_predictions()
        self.writer.add_scalar('Trajectory/AvgValuePrediction', np.mean(values), global_step)
        
        # Log action distribution
        actions = trajectory_store.actions()
        for action in self.action_names.keys():
            action_freq = np.mean(actions == action)
            self.writer.add_scalar(f'ActionFrequency/{self.action_names[action]}', action_freq, global_step)
        
        # Log example observations with value annotations
        self._log_value_heatmap(trajectory_store, global_step)
    
    def _log_action_distribution(self, global_step):
        """Log the current action distribution as a bar chart."""
        self.ax.clear()
        actions = list(self.action_names.keys())
        action_labels = [self.action_names[a] for a in actions]
        counts = [self.action_counts[a] for a in actions]
        
        # Create bar chart
        self.ax.bar(action_labels, counts)
        self.ax.set_xlabel('Actions')
        self.ax.set_ylabel('Count')
        self.ax.set_title('Agent Action Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Convert to image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log the figure as an image
        img = Image.open(buf)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        self.writer.add_image('Actions/Distribution', img_tensor, global_step)
        
        buf.close()
    
    def _log_value_heatmap(self, trajectory_store, global_step):
        """Log value predictions as a heatmap overlay on observations."""
        if trajectory_store.num_steps == 0:
            return
        
        # Sample a random timestep and batch index
        t = np.random.randint(0, trajectory_store.num_steps)
        b = np.random.randint(0, trajectory_store.batch_size)
        
        # Get observation and value
        obs = trajectory_store.obses[t, b]
        value = trajectory_store.value_predictions()[t, b]
        
        # Handle observations with different shapes
        if obs.shape[-1] == 6:  # Frame stack with 2 RGB frames
            # Just use the most recent frame (last 3 channels)
            obs = obs[..., -3:]
        
        # Create a heatmap overlay based on the value
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(obs)
        ax.set_title(f"Value Prediction: {value:.2f}")
        ax.axis('off')
        
        # Convert to image buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Log the figure as an image
        img = Image.open(buf)
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        self.writer.add_image('Values/Samples', img_tensor, global_step)
        
        plt.close(fig)
        buf.close()