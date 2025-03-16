"""
Observation preprocessing utilities for Obstacle Tower Challenge.

This module contains functions and classes for preprocessing observations
from the Obstacle Tower environment to make them suitable for neural networks.
"""
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Union, Optional

import torch
from gym import spaces


def grayscale(observation: np.ndarray) -> np.ndarray:
    """
    Convert RGB observation to grayscale.
    
    Args:
        observation: RGB observation array (H, W, 3)
        
    Returns:
        np.ndarray: Grayscale observation (H, W, 1)
    """
    if observation.shape[-1] == 1:
        return observation  # Already grayscale
    
    gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(gray, -1)


def resize(observation: np.ndarray, size: Tuple[int, int]=(84, 84)) -> np.ndarray:
    """
    Resize observation to target size.
    
    Args:
        observation: Input observation array
        size: Target size (height, width)
        
    Returns:
        np.ndarray: Resized observation
    """
    # Check if resizing is needed
    if observation.shape[0] == size[0] and observation.shape[1] == size[1]:
        return observation
    
    # Handle grayscale images
    if len(observation.shape) == 2 or observation.shape[-1] == 1:
        # Ensure shape is (h, w) for cv2.resize
        if len(observation.shape) == 3:
            observation = observation.squeeze(-1)
            
        resized = cv2.resize(observation, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)
    
    # Handle RGB images
    resized = cv2.resize(observation, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return resized


def normalize(observation: np.ndarray, scale: float=255.0) -> np.ndarray:
    """
    Normalize observation values to [0, 1] range.
    
    Args:
        observation: Input observation array
        scale: Value to divide by (typically 255 for images)
        
    Returns:
        np.ndarray: Normalized observation
    """
    return observation.astype(np.float32) / scale


def transpose_for_torch(observation: np.ndarray) -> np.ndarray:
    """
    Transpose observation from (H, W, C) to (C, H, W) format for PyTorch.
    
    Args:
        observation: Input observation array in (H, W, C) format
        
    Returns:
        np.ndarray: Transposed observation in (C, H, W) format
    """
    if len(observation.shape) == 3:
        return np.transpose(observation, (2, 0, 1))
    return observation


def stack_frames(frames: List[np.ndarray], axis: int=-1) -> np.ndarray:
    """
    Stack multiple frames along specified axis.
    
    Args:
        frames: List of frame arrays
        axis: Axis to stack along
        
    Returns:
        np.ndarray: Stacked frames
    """
    return np.concatenate(frames, axis=axis)


def preprocess_observation(
    observation: np.ndarray,
    grayscale: bool=True,
    resize_shape: Optional[Tuple[int, int]]=(84, 84),
    normalize_values: bool=True,
    transpose_for_pytorch: bool=True
) -> np.ndarray:
    """
    Apply standard preprocessing pipeline to observations.
    
    Args:
        observation: Raw observation from environment
        grayscale: Whether to convert to grayscale
        resize_shape: Target shape for resizing (height, width)
        normalize_values: Whether to normalize values to [0, 1]
        transpose_for_pytorch: Whether to transpose to PyTorch format
        
    Returns:
        np.ndarray: Preprocessed observation
    """
    # Convert to grayscale if requested
    if grayscale:
        observation = grayscale(observation)
    
    # Resize if shape is provided
    if resize_shape is not None:
        observation = resize(observation, resize_shape)
    
    # Normalize values if requested
    if normalize_values:
        observation = normalize(observation)
    
    # Transpose for PyTorch if requested
    if transpose_for_pytorch:
        observation = transpose_for_torch(observation)
    
    return observation


class ObservationNormalizer:
    """
    Online observation normalizer with running statistics.
    Can be used to normalize observations with adaptive mean and standard deviation.
    """
    def __init__(self, shape: Tuple[int, ...], clip_range: float=5.0, epsilon: float=1e-8):
        """
        Initialize observation normalizer.
        
        Args:
            shape: Shape of observations
            clip_range: Range to clip normalized values to
            epsilon: Small constant to avoid division by zero
        """
        self.shape = shape
        self.clip_range = clip_range
        self.epsilon = epsilon
        
        # Running statistics
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
    
    def update(self, observation: np.ndarray) -> None:
        """
        Update running statistics with new observation.
        
        Args:
            observation: New observation batch or single observation
        """
        # Ensure batch dimension
        batch_mean = np.mean(observation, axis=0)
        batch_var = np.var(observation, axis=0)
        batch_count = observation.shape[0] if len(observation.shape) > len(self.shape) else 1
        
        # Update running statistics using Welford's online algorithm
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        
        # Update mean
        self.mean = self.mean + delta * batch_count / new_count
        
        # Update variance
        self.var = (self.count * self.var + batch_count * batch_var + 
                   (delta ** 2) * self.count * batch_count / new_count) / new_count
        
        # Update count
        self.count = new_count
        
        # Recompute standard deviation
        self.std = np.sqrt(self.var) + self.epsilon
    
    def normalize(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation using current statistics.
        
        Args:
            observation: Observation to normalize
            
        Returns:
            np.ndarray: Normalized observation
        """
        normalized = (observation - self.mean) / self.std
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def unnormalize(self, normalized_observation: np.ndarray) -> np.ndarray:
        """
        Convert normalized observation back to original scale.
        
        Args:
            normalized_observation: Normalized observation
            
        Returns:
            np.ndarray: Unnormalized observation
        """
        return normalized_observation * self.std + self.mean
    
    def save(self, path: str) -> None:
        """
        Save normalizer statistics to file.
        
        Args:
            path: Path to save file
        """
        np.savez(
            path,
            mean=self.mean,
            var=self.var,
            count=self.count,
            shape=self.shape,
            clip_range=self.clip_range,
            epsilon=self.epsilon
        )
    
    def load(self, path: str) -> None:
        """
        Load normalizer statistics from file.
        
        Args:
            path: Path to save file
        """
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = int(data["count"])
        self.shape = tuple(data["shape"])
        self.clip_range = float(data["clip_range"])
        self.epsilon = float(data["epsilon"])
        self.std = np.sqrt(self.var) + self.epsilon


def extract_structured_info(info: Dict[str, Any], max_floor: int=15) -> np.ndarray:
    """
    Extract structured information from environment info dict.
    This can be used for dual encoder architectures that combine
    pixel observations with structured game state.
    
    Args:
        info: Info dictionary from environment
        max_floor: Maximum floor for normalization
        
    Returns:
        np.ndarray: Structured state vector
    """
    # Initialize structured state array
    structured_state = np.zeros(8, dtype=np.float32)
    
    # Extract information from info dict
    floor = info.get("floor", 0)
    keys = info.get("keys_collected", 0)
    doors = info.get("doors_opened", 0)
    time_remaining = info.get("time_remaining", 1.0)
    position = info.get("agent_position", (0, 0, 0))
    
    # Normalize values
    structured_state[0] = floor / max_floor  # Floor progress
    structured_state[1] = min(keys, 3) / 3.0  # Keys (max 3)
    structured_state[2] = min(doors, 10) / 10.0  # Doors (arbitrary max)
    structured_state[3] = time_remaining  # Time remaining (0-1)
    
    # Normalized 3D position coordinates
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        structured_state[4] = position[0]  # X (assume normalized)
        structured_state[5] = position[1]  # Y (assume normalized)
        structured_state[6] = position[2]  # Z (assume normalized)
    
    # Bias term
    structured_state[7] = 1.0
    
    return structured_state


def create_obstacle_tower_preprocessing_pipeline(
    channels: int,
    height: int,
    width: int,
    use_grayscale: bool=True,
    normalize: bool=True,
    frame_stack: int=4
) -> Dict[str, Any]:
    """
    Create preprocessing configuration for Obstacle Tower.
    Returns a dictionary with preprocessing functions and parameters.
    
    Args:
        channels: Number of observation channels
        height: Observation height
        width: Observation width
        use_grayscale: Whether to convert to grayscale
        normalize: Whether to normalize values
        frame_stack: Number of frames to stack
        
    Returns:
        dict: Preprocessing configuration
    """
    # Calculate output shape
    if use_grayscale:
        output_channels = 1 * frame_stack
    else:
        output_channels = 3 * frame_stack
    
    pipeline = {
        # Input shape
        "input_shape": (channels, height, width),
        
        # Preprocessing parameters
        "grayscale": use_grayscale,
        "resize": (height, width),
        "normalize": normalize,
        "frame_stack": frame_stack,
        
        # Output shape after preprocessing
        "output_shape": (output_channels, height, width),
        
        # Preprocessing function
        "preprocess_fn": lambda obs: preprocess_observation(
            obs, 
            grayscale=use_grayscale,
            resize_shape=(height, width),
            normalize_values=normalize,
            transpose_for_pytorch=True
        )
    }
    
    return pipeline