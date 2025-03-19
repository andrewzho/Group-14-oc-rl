"""
Enhanced trajectory collector with visualization capabilities.
"""

import os
import numpy as np
import random
from collections import deque

from src.data_collector import TensorboardTrajectoryCollector
from src.trajectory_store import TrajectoryBatch
from PIL import Image, ImageDraw, ImageFont  # Add this line
import time

class VisualizedTrajectoryCollector(TensorboardTrajectoryCollector):
    """
    A TrajectoryCollector that includes visualization capabilities.
    Records observations, actions, and metrics for visualization in TensorBoard.
    """
    
    def __init__(self, parallel_envs, model, num_steps, visualizer=None, 
                 display_env_idx=0, video_record_freq=10, display=False, record_video=False,
                 display_width=800, display_height=600, video_quality='high'):
        """
        Initialize the visualized trajectory collector.
        
        Args:
            parallel_envs: A BatchedEnv implementation.
            model: A Model with 'actions' in its output dicts.
            num_steps: The number of timesteps to run per batch of trajectories.
            visualizer: TensorBoard visualizer instance.
            display_env_idx: Which environment to use for visualization (default: 0).
            video_record_freq: How often to record full episode videos (default: every 10 episodes).
            display: Whether to display gameplay in a window.
            record_video: Whether to record videos of gameplay.
            display_width: Width for the display window (default: 800)
            display_height: Height for the display window (default: 600)
            video_quality: Quality setting for videos ('low', 'medium', 'high', 'ultra')
        """
        super().__init__(parallel_envs, model, num_steps)
        self.visualizer = visualizer
        self.display_env_idx = display_env_idx
        self.video_record_freq = video_record_freq
        self.display = display
        self.record_video = record_video
        self.display_width = display_width
        self.display_height = display_height
        self.video_quality = video_quality
        
        # Setup for visualization
        if display or record_video:
            self._setup_visualization()
        
        # Track episodes for the displayed environment
        self.episode_frames = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_count = 0
        
        # Track total steps
        self.total_steps = 0
        
        print(f"Visualized Trajectory Collector initialized. Displaying environment {display_env_idx}")
    
    def _setup_visualization(self):
        """Set up visualization if needed."""
        import os
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create directory for saving videos
        if self.record_video:
            os.makedirs("agent_videos", exist_ok=True)
            self.video_dir = "agent_videos"
        
        # Setup for display window
        if self.display:
            self.window_name = "Obstacle Tower Agent"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
    
    def reset(self):
        """Reset environments and prepare for visualization."""
        result = super().reset()
        
        # Clear episode data
        self.episode_frames = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Log initial observation if visualizer is available
        if self.visualizer is not None:
            if isinstance(self._prev_obs, np.ndarray) and self.display_env_idx < len(self._prev_obs):
                self.visualizer.log_observation(self._prev_obs[self.display_env_idx], self.total_steps)
        
        return result
    
    def trajectory_store(self):
        """
        Run a trajectory_store and collect experiences with visualization.
        """
        if self._prev_obs is None:
            self.reset()
            
        # Create the trajectory_store normally
        result = super().trajectory_store()
        
        # Log visualization data for the displayed environment if visualizer is available
        if self.visualizer is not None:
            # Log observations, actions, and rewards from the trajectory_store
            self._process_visualization_data(result)
            
            # Log metrics from the trajectory_store
            self.visualizer.log_trajectory_metrics(result, self.total_steps)
        
        # Update total steps
        self.total_steps += result.num_steps * result.batch_size
        
        return result
    
    def _process_visualization_data(self, trajectory_store):
        """Process trajectory_store data for visualization."""
        # Extract data for the displayed environment
        env_idx = self.display_env_idx
        if env_idx >= trajectory_store.batch_size:
            return  # Skip if index is out of range
        
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw
        
        for t in range(trajectory_store.num_steps):
            # Get observation and convert to displayable format
            obs = trajectory_store.obses[t, env_idx]
            
            # Convert observation to displayable image
            if isinstance(obs, np.ndarray):
                # If frame stack, use most recent frame
                if obs.shape[-1] == 6:  # 2 RGB frames
                    display_obs = obs[..., -3:]
                else:
                    display_obs = obs
                
                # Create PIL image
                img = Image.fromarray(display_obs)
                
                # Add annotations
                img = self._annotate_frame(
                    img, 
                    trajectory_store.actions()[t, env_idx],
                    trajectory_store.rews[t, env_idx], 
                    trajectory_store.infos[t][env_idx] if t < len(trajectory_store.infos) else {}
                )
                
                # Convert back to numpy for display
                display_img = np.array(img)
                
                # Display in window if enabled
                if self.display:
                    cv2.imshow(self.window_name, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(50)
                    time.sleep(0.05)
                
                # Store for video recording
                if self.record_video:
                    self.episode_frames.append(display_img)
            
            # Log to TensorBoard
            if self.visualizer:
                self.visualizer.log_observation(obs, self.total_steps + t)
                self.visualizer.log_action(trajectory_store.actions()[t, env_idx], self.total_steps + t)
            
            # Store actions and rewards
            self.episode_actions.append(trajectory_store.actions()[t, env_idx])
            self.episode_rewards.append(trajectory_store.rews[t, env_idx])
            
            # Check if episode ended
            if t+1 < len(trajectory_store.dones) and trajectory_store.dones[t+1, env_idx]:
                self._handle_episode_end()
    
    def _annotate_frame(self, frame, action, reward, info):
        """Add annotations to the observation frame."""
        # Action names for visualization
        action_names = {
            0: "No-op", 3: "Forward", 6: "Backward", 9: "Left",
            12: "Right", 15: "Jump", 18: "Forward+Jump", 21: "Backward+Jump",
            24: "Left+Jump", 27: "Right+Jump", 30: "Forward+Left", 33: "Forward+Right"
        }
        
        # Create a drawing context
        draw = ImageDraw.Draw(frame)
        
        # Add action name
        action_name = action_names.get(action, str(action))
        draw.text((10, 10), f"Action: {action_name}", fill=(255, 255, 255))
        
        # Add reward
        draw.text((10, 25), f"Reward: {reward:.2f}", fill=(255, 255, 255))
        
        # Add floor info if available
        if 'current_floor' in info:
            draw.text((10, 40), f"Floor: {info['current_floor']}", fill=(255, 255, 255))
        
        # Add episode info
        draw.text((10, 55), f"Episode: {self.episode_count}", fill=(255, 255, 255))
        
        return frame
    
    def _handle_episode_end(self):
        """Handle the end of an episode for the displayed environment."""
        # Increment episode counter
        self.episode_count += 1
        
        # Record a video of the episode if it's time
        if self.record_video and self.episode_count % self.video_record_freq == 0:
            if len(self.episode_frames) > 0:
                self._save_episode_video()
                print(f"Recorded episode {self.episode_count} with {len(self.episode_frames)} frames")
        
        # Log to TensorBoard
        if self.visualizer is not None and len(self.episode_frames) > 0:
            self.visualizer.log_episode_video(self.episode_frames, self.episode_count)
        
        # Clear episode data
        self.episode_frames = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _save_episode_video(self):
        """Save the episode frames as a video file."""
        import cv2
        import os
        import datetime
        
        # Create directory
        os.makedirs(self.video_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.video_dir, f"episode_{self.episode_count}_{timestamp}.mp4")
        
        # Get video dimensions
        if not self.episode_frames:
            return
            
        height, width, layers = self.episode_frames[0].shape
        
        # Set quality parameters based on video_quality setting
        if self.video_quality == 'high':
            # Higher quality video output using H.264 codec with quality parameters
            fourcc = cv2.VideoWriter_fourcc(*'H264')  # Higher quality codec
            fps = 30
            scale_factor = min(2.0, max(640/width, 480/height))  # Upscale to at least 640x480, but no more than 2x
            bitrate = 5000000  # 5 Mbps
        elif self.video_quality == 'medium':
            # Medium quality with XVID codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = 24
            scale_factor = min(1.5, max(480/width, 360/height))  # Medium upscaling
            bitrate = 2000000  # 2 Mbps
        elif self.video_quality == 'ultra':
            # Ultra quality with 1080p output
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            fps = 30
            # Calculate scale factor to reach 1080p (1920x1080)
            scale_factor = min(3.0, max(1920/width, 1080/height))
            bitrate = 8000000  # 8 Mbps for high-quality 1080p
        else:  # 'low'
            # Low quality with basic mp4v codec, no upscaling
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 20
            scale_factor = 1.0
            bitrate = 1000000  # 1 Mbps
        
        # Calculate new dimensions if scaling
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Try H.264 first, fallback to alternatives if not available
        try:
            video = cv2.VideoWriter(filename, fourcc, fps, (new_width, new_height))
            # Test if writer is properly initialized
            if not video.isOpened():
                raise Exception(f"{fourcc} codec not available")
        except Exception as e:
            print(f"{fourcc} codec not available, falling back to mp4v: {e}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(filename, fourcc, fps, (new_width, new_height))
        
        # Apply enhanced bitrate if using OpenCV 4.1+
        try:
            video.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # Maximum quality
            if hasattr(cv2, 'VIDEOWRITER_PROP_BITRATE'):
                video.set(cv2.VIDEOWRITER_PROP_BITRATE, bitrate)
        except:
            print("Note: Enhanced bitrate settings not supported in this OpenCV version")
        
        # Write frames with upscaling if needed
        for frame in self.episode_frames:
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Release video writer
        video.release()
        print(f"Saved {self.video_quality} quality video ({new_width}x{new_height}) to {filename}")