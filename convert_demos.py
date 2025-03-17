"""
Convert demonstration data from raw format to DemonstrationBuffer format.

This script processes demonstration data consisting of:
- PNG image frames
- actions.json file containing action sequence
- rewards.json file containing reward sequence

Usage:
    python convert_demos.py --input raw_demos --output demonstrations/demonstrations.pkl
"""
import os
import json
import argparse
import numpy as np
from PIL import Image
import cv2
import pickle
from tqdm import tqdm
import sys
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('demo_converter')

# Add the project root to the path so we can import the demo buffer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.sac_utils.demo_buffer import DemonstrationBuffer
except ImportError:
    logger.error("Could not import DemonstrationBuffer. Make sure you're running this script from the project root.")
    sys.exit(1)

# Apply NumPy patch for mlagents_envs if needed
if not hasattr(np, 'bool'):
    setattr(np, 'bool', np.bool_)


def load_actions_and_rewards(folder_path: str) -> Tuple[List[Any], List[float]]:
    """
    Load actions and rewards from json files.
    
    Args:
        folder_path: Path to folder containing demonstration data
        
    Returns:
        tuple: (actions, rewards)
    """
    actions_file = os.path.join(folder_path, "actions.json")
    rewards_file = os.path.join(folder_path, "rewards.json")
    
    if not os.path.exists(actions_file):
        raise FileNotFoundError(f"actions.json not found in {folder_path}")
    if not os.path.exists(rewards_file):
        raise FileNotFoundError(f"rewards.json not found in {folder_path}")
    
    # Load actions and rewards
    with open(actions_file, "r") as f:
        actions_data = json.load(f)
        
    with open(rewards_file, "r") as f:
        rewards_data = json.load(f)
    
    # Handle different data formats
    actions = _extract_data_from_json(actions_data, ["actions", "action"])
    rewards = _extract_data_from_json(rewards_data, ["rewards", "reward"])
    
    return actions, rewards


def _extract_data_from_json(data: Union[Dict, List], possible_keys: List[str]) -> List:
    """
    Extract data from potentially nested JSON structure.
    
    Args:
        data: JSON data (could be dict or list)
        possible_keys: Keys to try for extraction
        
    Returns:
        List of extracted data
    """
    if isinstance(data, list):
        return data
    
    if isinstance(data, dict):
        # Try to find the data using possible keys
        for key in possible_keys:
            if key in data:
                return data[key]
        
        # If no matching key, try values
        if len(data) > 0 and isinstance(list(data.values())[0], (list, int, float)):
            return list(data.values())
        
        # If values are not directly usable, try finding nested structures
        for value in data.values():
            if isinstance(value, (list, dict)):
                result = _extract_data_from_json(value, possible_keys)
                if result:
                    return result
    
    # Default return empty list if nothing found
    logger.warning(f"Could not extract data from: {data}")
    return []


def get_sorted_image_files(folder_path: str) -> List[str]:
    """
    Get sorted list of image files in the folder.
    
    Args:
        folder_path: Path to folder containing image files
        
    Returns:
        list: Sorted list of image file paths
    """
    image_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    
    # Try to sort by numeric part of filename
    try:
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    except:
        # Fall back to regular sort
        image_files.sort()
        
    return image_files


def process_action(action: Any, retro: bool = True) -> Any:
    """
    Process action data into the format expected by the environment.
    
    Args:
        action: Action data from demonstration
        retro: Whether to use retro format
        
    Returns:
        Processed action
    """
    # Handle different action formats
    if isinstance(action, list):
        if retro:
            # For retro mode, convert to a discrete action (0-53)
            if len(action) == 1:
                # If it's a single value in a list
                return int(action[0])
            elif len(action) > 1:
                # If it's a multi-discrete action, we need to map it
                # This is a simplified example - you may need to customize this
                # based on your specific action space
                action_mapping = {
                    # Example mapping - modify based on your environment
                    (1, 0, 0, 0): 0,  # No-op
                    (0, 1, 0, 0): 1,  # Forward
                    (0, 0, 1, 0): 2,  # Rotate left
                    (0, 0, 0, 1): 3,  # Rotate right
                    # Add other mappings as needed
                }
                action_tuple = tuple(int(a) for a in action)
                if action_tuple in action_mapping:
                    return action_mapping[action_tuple]
                else:
                    # Default to a safe action if mapping not found
                    logger.warning(f"Unknown action tuple: {action_tuple}, defaulting to 0")
                    return 0
            else:
                # Empty list, default to no-op
                return 0
        else:
            # For non-retro mode, keep as multi-discrete but ensure it's in the right format
            # Convert to numpy array if needed
            return np.array(action, dtype=np.float32)
    elif isinstance(action, dict):
        # If action is a dictionary, extract values
        # This depends on your specific format
        values = list(action.values())
        return process_action(values, retro)
    elif isinstance(action, (int, float)):
        # If action is already a scalar, use it directly
        return int(action) if retro else float(action)
    else:
        # Unknown format, default to no-op
        logger.warning(f"Unknown action format: {type(action)}, value: {action}, defaulting to 0")
        return 0 if retro else np.zeros(4, dtype=np.float32)  # Adjust size as needed


def process_demonstration_batch(image_files: List[str], actions: List[Any], rewards: List[float], 
                              resize: Optional[Tuple[int, int]] = None, retro: bool = True,
                              batch_size: int = 100) -> Tuple[List, List, List, List, List]:
    """
    Process a batch of demonstration data.
    
    Args:
        image_files: List of image file paths
        actions: List of actions
        rewards: List of rewards
        resize: Tuple (height, width) to resize images to, or None for no resizing
        retro: Whether to use retro format
        batch_size: Number of items to process in a batch
        
    Returns:
        tuple: (observations, actions, rewards, next_observations, dones)
    """
    # Ensure lengths match
    min_len = min(len(image_files), len(actions), len(rewards))
    if min_len < len(image_files) or min_len < len(actions) or min_len < len(rewards):
        logger.warning(f"Mismatched lengths - images: {len(image_files)}, actions: {len(actions)}, rewards: {len(rewards)}")
        logger.warning(f"Truncating to {min_len} steps")
        image_files = image_files[:min_len]
        actions = actions[:min_len]
        rewards = rewards[:min_len]
    
    # Process in batches to save memory
    observations = []
    processed_actions = []
    
    # Process batches
    for i in range(0, len(image_files), batch_size):
        batch_images = image_files[i:i + batch_size]
        batch_actions = actions[i:i + batch_size]
        
        # Process images
        batch_observations = []
        for img_file in tqdm(batch_images, desc=f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1}"):
            img = Image.open(img_file)
            
            # Convert to numpy array
            obs = np.array(img)
            
            # Resize if needed
            if resize is not None:
                obs = cv2.resize(obs, (resize[1], resize[0]), interpolation=cv2.INTER_AREA)
            
            # For retro mode, keep as uint8 (0-255 range)
            # Otherwise, normalize to 0-1 float32
            if not retro:
                obs = obs.astype(np.float32) / 255.0
                
            # Convert to channel-first format (HWC -> CHW)
            obs = np.transpose(obs, (2, 0, 1))
            
            # Stack frames for 4-channel format (needed for model compatibility)
            if obs.shape[0] == 3:  # If RGB
                # Duplicate first channel to create 4 channels
                obs = np.concatenate([obs[:1], obs], axis=0)[:4]
                
            batch_observations.append(obs)
        
        observations.extend(batch_observations)
        
        # Process actions
        batch_processed_actions = [process_action(action, retro) for action in batch_actions]
        processed_actions.extend(batch_processed_actions)
    
    # Create next_observations by shifting
    next_observations = observations[1:] + [observations[-1]]
    
    # Create done flags (only the last step is done)
    dones = [False] * (len(observations) - 1) + [True]
    
    return observations, processed_actions, rewards, next_observations, dones


def load_demonstration_folder(folder_path: str, resize: Optional[Tuple[int, int]] = (84, 84), 
                             retro: bool = True, batch_size: int = 100) -> Tuple[List, List, List, List, List]:
    """
    Load demonstration data from a folder.
    
    Args:
        folder_path: Path to folder containing demonstration data
        resize: Tuple (height, width) to resize images to, or None for no resizing
        retro: Whether to use retro format (smaller observations)
        batch_size: Number of items to process in a batch
        
    Returns:
        tuple: (observations, actions, rewards, next_observations, dones)
    """
    logger.info(f"Processing folder: {folder_path}")
    
    # Load actions and rewards
    actions, rewards = load_actions_and_rewards(folder_path)
    
    # Get sorted image files
    image_files = get_sorted_image_files(folder_path)
    
    # Process the demonstration data
    observations, processed_actions, rewards, next_observations, dones = process_demonstration_batch(
        image_files, actions, rewards, resize, retro, batch_size
    )
    
    # Print summary
    logger.info(f"Loaded {len(observations)} observations, {len(processed_actions)} actions, {len(rewards)} rewards")
    logger.info(f"Observation shape: {observations[0].shape}, dtype: {observations[0].dtype}")
    logger.info(f"Action sample: {processed_actions[0]}")
    logger.info(f"Reward range: {min(rewards)} to {max(rewards)}, sum: {sum(rewards)}")
    
    return observations, processed_actions, rewards, next_observations, dones


def validate_buffer(buffer: DemonstrationBuffer) -> bool:
    """
    Validate the demonstration buffer to ensure it's suitable for training.
    
    Args:
        buffer: DemonstrationBuffer to validate
        
    Returns:
        bool: True if buffer is valid, False otherwise
    """
    # Check if buffer is empty
    if buffer.num_episodes == 0 or buffer.total_transitions == 0:
        logger.error("Buffer is empty")
        return False
    
    # Check for basic integrity
    try:
        # Try to sample from the buffer (if implemented)
        sample = buffer.sample(1)
        logger.info("Successfully sampled from buffer")
        
        # Check observation format
        obs = sample["observations"]
        if isinstance(obs, np.ndarray):
            logger.info(f"Sample observation shape: {obs.shape}")
            if len(obs.shape) == 4:  # Batch, Channel, Height, Width
                logger.info(f"Observation format: {'NCHW (correct)' if obs.shape[1] in [3, 4] else 'unknown'}")
                logger.info(f"Number of channels: {obs.shape[1]}")
            elif len(obs.shape) == 3:  # Channel, Height, Width (single observation)
                logger.info(f"Observation format: {'CHW (correct)' if obs.shape[0] in [3, 4] else 'unknown'}")
                logger.info(f"Number of channels: {obs.shape[0]}")
            
        # Check action format
        actions = sample["actions"]
        logger.info(f"Sample action shape: {actions.shape if hasattr(actions, 'shape') else 'scalar'}")
        
    except Exception as e:
        logger.error(f"Error sampling from buffer: {e}")
        return False
    
    logger.info("Buffer validation passed")
    return True


def main():
    """Main function to convert demonstrations"""
    parser = argparse.ArgumentParser(description="Convert demonstration data to DemonstrationBuffer format")
    parser.add_argument("--input", type=str, default="raw_demos", help="Directory containing demonstration folders")
    parser.add_argument("--output", type=str, default="demonstrations/demonstrations1.pkl", help="Output file path")
    parser.add_argument("--resize", type=str, default="84,84", help="Resize observations to height,width")
    parser.add_argument("--retro", action="store_true", help="Use retro format (uint8, 0-255 range)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--validate", action="store_true", help="Validate buffer after creation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Parse resize dimensions
    if args.resize:
        height, width = map(int, args.resize.split(","))
        resize = (height, width)
    else:
        resize = None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Find all demonstration folders
    if os.path.isdir(args.input):
        demo_folders = []
        for item in os.listdir(args.input):
            folder_path = os.path.join(args.input, item)
            if os.path.isdir(folder_path):
                demo_folders.append(folder_path)
    else:
        logger.error(f"Input directory {args.input} not found")
        return
    
    if not demo_folders:
        logger.error(f"No demonstration folders found in {args.input}")
        return
    
    logger.info(f"Found {len(demo_folders)} demonstration folders")
    
    # Create demonstration buffer
    buffer = DemonstrationBuffer()
    
    # Process each demonstration folder
    for folder in demo_folders:
        try:
            obs, acts, rews, next_obs, dones = load_demonstration_folder(
                folder, resize=resize, retro=args.retro, batch_size=args.batch_size
            )
            buffer.add_episode(obs, acts, rews, next_obs, dones)
            logger.info(f"Added episode from {folder} to buffer")
        except Exception as e:
            logger.error(f"Error processing folder {folder}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Validate buffer if requested
    if args.validate and buffer.num_episodes > 0:
        if not validate_buffer(buffer):
            logger.warning("Buffer validation failed. The buffer may not be suitable for training.")
    
    # Save the buffer
    buffer.save(args.output)
    logger.info(f"Demonstration buffer saved to {args.output}")
    logger.info(f"Contains {buffer.total_transitions} transitions from {buffer.num_episodes} episodes")
    
    # Show buffer statistics
    stats = buffer.get_statistics()
    logger.info("\nDemonstration Statistics:")
    logger.info(f"Average Episode Length: {stats['avg_episode_length']:.2f} steps")
    logger.info(f"Average Episode Reward: {stats['avg_episode_reward']:.2f}")
    logger.info(f"Min/Max/Mean Reward: {stats['min_reward']:.2f}/{stats['max_reward']:.2f}/{stats['mean_reward']:.2f}")


if __name__ == "__main__":
    main()