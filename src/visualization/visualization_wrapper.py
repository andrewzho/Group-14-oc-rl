"""
Visualization wrapper for monitoring agent gameplay in Obstacle Tower.
"""

import os
import time
import datetime
import threading
import queue
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gym

class VisualizationWrapper(gym.Wrapper):
    """
    A wrapper that records observations and actions for visualization.
    Can display gameplay in real-time and log images to TensorBoard.
    """
    
    def __init__(self, env, save_dir='./agent_videos', max_queue_size=100, 
                 record_freq=1, display=False, record_episodes=True):
        """
        Initialize the visualization wrapper.
        
        Args:
            env: The environment to wrap.
            save_dir: Directory to save recorded frames.
            max_queue_size: Maximum size of the frame queue.
            record_freq: How often to record frames (1 = every frame).
            display: Whether to display frames in real-time.
            record_episodes: Whether to record full episodes.
        """
        super().__init__(env)
        self.save_dir = save_dir
        self.record_freq = record_freq
        self.display = display
        self.record_episodes = record_episodes
        
        # Create directories for saved frames
        os.makedirs(save_dir, exist_ok=True)
        self.episode_dir = None
        
        # Setup frame queue and display thread
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        
        # Start display thread if needed
        if self.display:
            self.display_thread = threading.Thread(target=self._display_loop)
            self.display_thread.daemon = True
            self.display_thread.start()
        
        # Track episode
        self.episode_count = 0
        self.step_count = 0
        self.episode_frames = []
        self.episode_rewards = []
        self.episode_actions = []
        self.current_floor = 0
        
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
        
        # Try to load a font for annotations
        try:
            self.font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                self.font = ImageFont.truetype("FreeSans.ttf", 12)
            except:
                self.font = ImageFont.load_default()

    def reset(self, **kwargs):
        """Reset the environment and start a new episode recording."""
        obs = self.env.reset(**kwargs)
        
        # Create new episode directory if recording episodes
        if self.record_episodes:
            self.episode_count += 1
            self.step_count = 0
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.episode_dir = os.path.join(self.save_dir, f"episode_{self.episode_count}_{timestamp}")
            os.makedirs(self.episode_dir, exist_ok=True)
            
            # Clear episode data
            self.episode_frames = []
            self.episode_rewards = []
            self.episode_actions = []
            self.current_floor = 0
            
            # Capture initial frame
            self._process_frame(obs, 0, 0, {})
        
        return obs

    def step(self, action):
        """Execute environment step and record frames."""
        obs, reward, done, info = self.env.step(action)
        
        self.step_count += 1
        
        # Update current floor information
        if 'current_floor' in info:
            self.current_floor = info['current_floor']
        
        # Record frame 
        if self.step_count % self.record_freq == 0:
            self._process_frame(obs, action, reward, info)
            
        # If episode is done and we're recording full episodes, save the video
        if done and self.record_episodes and len(self.episode_frames) > 0:
            self._save_episode()
        
        return obs, reward, done, info
    
    def close(self):
        """Clean up resources."""
        self.running = False
        if self.display:
            self.display_thread.join(timeout=1.0)
        super().close()
        
    def _process_frame(self, obs, action, reward, info):
        """Process an observation frame for display and recording."""
        if isinstance(obs, tuple) and len(obs) == 2:
            # Handle (memory_state, obs) tuple format
            _, obs_img = obs
        else:
            obs_img = obs
            
        # Create a copy of the observation for annotation
        frame = self._annotate_frame(obs_img, action, reward, info)
        
        # Add to episode recording
        if self.record_episodes:
            self.episode_frames.append(frame.copy())
            self.episode_rewards.append(reward)
            self.episode_actions.append(action)
            
            # Save individual frame
            if self.episode_dir:
                frame.save(os.path.join(self.episode_dir, f"frame_{self.step_count:06d}.png"))
        
        # Add to display queue if needed
        if self.display:
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
    
    def _annotate_frame(self, obs, action, reward, info):
        """Add annotations to the observation frame."""
        # Convert observation to PIL Image if needed
        if isinstance(obs, np.ndarray):
            frame = Image.fromarray(obs)
        else:
            frame = obs
        
        # Create a drawing context
        draw = ImageDraw.Draw(frame)
        
        # Add action name
        action_name = self.action_names.get(action, str(action))
        draw.text((10, 10), f"Action: {action_name}", fill=(255, 255, 255), font=self.font)
        
        # Add reward
        draw.text((10, 25), f"Reward: {reward:.2f}", fill=(255, 255, 255), font=self.font)
        
        # Add floor info if available
        if 'current_floor' in info:
            draw.text((10, 40), f"Floor: {info['current_floor']}", fill=(255, 255, 255), font=self.font)
        elif hasattr(self, 'current_floor'):
            draw.text((10, 40), f"Floor: {self.current_floor}", fill=(255, 255, 255), font=self.font)
        
        # Add step count
        draw.text((10, 55), f"Step: {self.step_count}", fill=(255, 255, 255), font=self.font)
        
        return frame
    
    def _save_episode(self):
        """Save the recorded episode as a series of images."""
        print(f"Saving episode {self.episode_count} with {len(self.episode_frames)} frames to {self.episode_dir}")
        
        # Create a summary text file with episode information
        with open(os.path.join(self.episode_dir, "episode_summary.txt"), "w") as f:
            f.write(f"Episode: {self.episode_count}\n")
            f.write(f"Steps: {self.step_count}\n")
            f.write(f"Total Reward: {sum(self.episode_rewards):.2f}\n")
            f.write(f"Max Floor: {self.current_floor}\n")
            
            # Write step-by-step information
            f.write("\nStep-by-step details:\n")
            for i, (action, reward) in enumerate(zip(self.episode_actions, self.episode_rewards)):
                action_name = self.action_names.get(action, str(action))
                f.write(f"Step {i+1}: Action={action_name}, Reward={reward:.2f}\n")
    
    def _display_loop(self):
        """Display loop for showing frames in real-time."""
        try:
            import cv2
            window_name = "Obstacle Tower Agent"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
            
            while self.running:
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except queue.Empty:
                    pass
                
            cv2.destroyAllWindows()
        except ImportError:
            print("OpenCV not available for real-time display. Install with 'pip install opencv-python'")
            self.display = False