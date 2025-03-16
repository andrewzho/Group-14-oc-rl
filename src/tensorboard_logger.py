import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import cv2
import os
from datetime import datetime

class TensorboardLogger:
    """
    TensorBoard logger for RL training visualization.
    Handles logging of scalar metrics, images, videos, and hyperparameters.
    """
    def __init__(self, log_dir):
        """
        Initialize the TensorBoard logger.
        
        Args:
            log_dir: Directory where TensorBoard logs will be saved
        """
        self.writer = SummaryWriter(log_dir)
        self.step_count = 0
        self.episode_count = 0
        
        # Create timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Print initialization message
        print(f"TensorBoard logger initialized - logs saved to {log_dir}")
        print(f"To view logs, run: tensorboard --logdir={log_dir}")
        
    def log_scalar(self, tag, value, step=None):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag: Name for the scalar value
            value: The scalar value to log
            step: Optional step/iteration count (uses internal counter if not provided)
        """
        if step is None:
            step = self.step_count
            
        # Convert value to native Python type if needed
        if isinstance(value, torch.Tensor):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else value.tolist()
            
        try:
            self.writer.add_scalar(tag, value, step)
        except Exception as e:
            print(f"Warning: Failed to log scalar {tag}: {e}")
    
    def log_episode(self, reward, length, floor, max_floor, step_count=None):
        """
        Log episode metrics to TensorBoard.
        
        Args:
            reward: Episode total reward
            length: Episode length in steps
            floor: Floor reached in this episode
            max_floor: Maximum floor reached across all episodes
            step_count: Total steps so far (optional)
        """
        if step_count is not None:
            self.step_count = step_count
        else:
            self.step_count += length
            
        self.episode_count += 1
        
        # Log episode metrics
        self.writer.add_scalar('episode/reward', reward, self.episode_count)
        self.writer.add_scalar('episode/length', length, self.episode_count)
        self.writer.add_scalar('episode/floor', floor, self.episode_count)
        self.writer.add_scalar('episode/max_floor', max_floor, self.episode_count)
        
        # Also log by steps for better comparison with other runs
        self.writer.add_scalar('steps/reward', reward, self.step_count)
        self.writer.add_scalar('steps/floor', floor, self.step_count)
        
    def log_update(self, metrics):
        """
        Log policy update metrics to TensorBoard.
        
        Args:
            metrics: Dictionary of metrics from policy update
        """
        for key, value in metrics.items():
            if isinstance(value, dict):
                continue  # Skip nested dictionaries
            if isinstance(value, (int, float)) or (isinstance(value, (np.ndarray, torch.Tensor)) and np.prod(np.shape(value)) == 1):
                self.log_scalar(f'update/{key}', value, self.step_count)
            
    def log_image(self, tag, image, step=None):
        """
        Log an image to TensorBoard.
        
        Args:
            tag: Name for the image
            image: The image to log (numpy array)
            step: Optional step/iteration count
        """
        if step is None:
            step = self.step_count
            
        # Convert image to proper format if needed
        if isinstance(image, np.ndarray):
            # Convert to uint8 if normalized float
            if image.dtype == np.float32 and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
                
            # Ensure HWC format (height, width, channels)
            if len(image.shape) == 3 and image.shape[0] == 3:  # CHW format
                image = np.transpose(image, (1, 2, 0))
                
            # Add batch dimension with dataformats='HWC'
            try:
                self.writer.add_image(tag, image, step, dataformats='HWC')
            except Exception as e:
                print(f"Warning: Failed to log image {tag}: {e}")
        
    def log_video(self, tag, frames, fps=30, step=None):
        """
        Log a video to TensorBoard.
        
        Args:
            tag: Name for the video
            frames: List of frames (numpy arrays)
            fps: Frames per second
            step: Optional step/iteration count
        """
        if step is None:
            step = self.step_count
            
        if not frames:
            print(f"Warning: No frames provided for video {tag}")
            return
            
        try:
            # Convert frames to proper format
            video_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    # Convert to uint8 if normalized float
                    if frame.dtype == np.float32 and frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                        
                    # Ensure HWC format (height, width, channels)
                    if len(frame.shape) == 3 and frame.shape[0] == 3:  # CHW format
                        frame = np.transpose(frame, (1, 2, 0))
                        
                    video_frames.append(frame)
                
            # Stack frames to [T, H, W, C] format
            if video_frames:
                video_tensor = np.stack(video_frames)
                
                # Convert to [T, C, H, W] for TensorBoard
                video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))
                
                # Add batch dimension [1, T, C, H, W]
                video_tensor = video_tensor[None]
                
                # Log to TensorBoard
                self.writer.add_video(tag, video_tensor, step, fps=fps)
                
                # Also save as MP4 file for easier viewing
                self._save_video_to_file(tag, video_frames, fps, step)
        except Exception as e:
            print(f"Warning: Failed to log video {tag}: {e}")
            
    def _save_video_to_file(self, tag, frames, fps=30, step=None):
        """
        Save video frames to a mp4 file.
        
        Args:
            tag: Name for the video
            frames: List of frames (numpy arrays in HWC format)
            fps: Frames per second
            step: Optional step/iteration count for filename
        """
        try:
            # Create videos directory inside the TensorBoard directory
            videos_dir = os.path.join(os.path.dirname(self.writer.log_dir), 'videos')
            os.makedirs(videos_dir, exist_ok=True)
            
            # Create output filename
            step_str = f"_step{step}" if step is not None else ""
            filename = os.path.join(videos_dir, f"{tag.replace('/', '_')}{step_str}_{self.timestamp}.mp4")
            
            # Get dimensions from first frame
            height, width, layers = frames[0].shape
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            # Add frames to video
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                video.write(frame[:, :, ::-1])
                
            # Release video writer
            video.release()
            print(f"Video saved to {filename}")
        except Exception as e:
            print(f"Warning: Failed to save video file: {e}")
        
    def log_hyperparams(self, hyperparams):
        """
        Log hyperparameters to TensorBoard.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        # Convert hyperparams to proper format for TensorBoard
        hparam_dict = {}
        metric_dict = {"validation/accuracy": 0}  # Dummy metric required by TensorBoard
        
        for key, value in hyperparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            else:
                # For non-primitive types, convert to string
                hparam_dict[key] = str(value)
        
        try:
            self.writer.add_hparams(hparam_dict, metric_dict)
        except Exception as e:
            print(f"Warning: Failed to log hyperparameters: {e}")
            
        # Also log hyperparameters as text for easier reference
        hyperparam_text = "\n".join([f"{k}: {v}" for k, v in hyperparams.items()])
        self.writer.add_text("hyperparameters", hyperparam_text)
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        print("TensorBoard logger closed")