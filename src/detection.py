import numpy as np
import cv2
import os

# Get global verbosity level from environment variable
VERBOSITY = int(os.environ.get('VERBOSITY', '1'))

def debug_print(*args, **kwargs):
    """Only print if verbosity level is high enough."""
    if VERBOSITY >= 2:  # Only print at verbosity level 2 or higher
        print(*args, **kwargs)

def detect_key_visually(observation, previous_observation):
    """
    Detect key appearance/disappearance in observation based on color patterns
    
    Args:
        observation: Current frame
        previous_observation: Previous frame
        
    Returns:
        bool: Whether a key-like object was detected
    """
    if previous_observation is None:
        return False
    
    # Handle tuple observations (visual_obs, vector_obs)
    if isinstance(observation, tuple):
        observation = observation[0]  # Extract visual observation
    
    if isinstance(previous_observation, tuple):
        previous_observation = previous_observation[0]  # Extract visual observation
        
    # Convert to appropriate format if needed
    if observation.shape[-1] == 3:  # Check if color channel is last dimension
        obs = observation.copy()
        prev_obs = previous_observation.copy()
    else:
        # Reshape if needed
        obs = observation.transpose(1, 2, 0) if observation.shape[0] == 3 else observation
        prev_obs = previous_observation.transpose(1, 2, 0) if previous_observation.shape[0] == 3 else previous_observation
        
    # Normalize observation if needed
    if obs.max() > 1.0:
        obs = obs / 255.0
        prev_obs = prev_obs / 255.0
        
    # Check for significant yellow pixel changes in middle region of observation
    # Enhanced yellow detection - keys in Obstacle Tower are typically yellowish
    # Convert to HSV color space for better color detection
    try:
        obs_hsv = cv2.cvtColor((obs * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        prev_hsv = cv2.cvtColor((prev_obs * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Yellow in HSV space
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        # Create masks for yellow regions
        mask_current = cv2.inRange(obs_hsv, lower_yellow, upper_yellow)
        mask_previous = cv2.inRange(prev_hsv, lower_yellow, upper_yellow)
        
        # Focus on central region where keys typically appear
        center_region_current = mask_current[20:60, 20:60]
        center_region_previous = mask_previous[20:60, 20:60]
        
        # Count yellow pixels
        current_yellow = np.sum(center_region_current > 0)
        prev_yellow = np.sum(center_region_previous > 0)
        
        # Check for significant change in yellow pixels
        return abs(current_yellow - prev_yellow) > 50
        
    except Exception as e:
        # Fallback to simpler RGB-based detection if HSV conversion fails
        current_yellow = np.sum((obs[20:60, 20:60, 0] > 0.8) & 
                              (obs[20:60, 20:60, 1] > 0.8) & 
                              (obs[20:60, 20:60, 2] < 0.4))
        
        prev_yellow = np.sum((prev_obs[20:60, 20:60, 0] > 0.8) & 
                           (prev_obs[20:60, 20:60, 1] > 0.8) & 
                           (prev_obs[20:60, 20:60, 2] < 0.4))
        
        # Significant change in yellow pixels could indicate key appearance/disappearance
        return abs(current_yellow - prev_yellow) > 50

def detect_door_visually(observation):
    """
    Detect door-like structures in observation
    
    Args:
        observation: Current frame
        
    Returns:
        bool: Whether a door-like object was detected
    """
    # Handle tuple observations (visual_obs, vector_obs)
    if isinstance(observation, tuple):
        observation = observation[0]  # Extract visual observation
    
    # Convert to appropriate format if needed
    if observation.shape[-1] == 3:  # Check if color channel is last dimension
        obs = observation.copy()
    else:
        # Reshape if needed
        obs = observation.transpose(1, 2, 0) if observation.shape[0] == 3 else observation
    
    # Normalize if needed
    if obs.max() > 1.0:
        obs = obs / 255.0
        
    try:
        # Convert to uint8 for OpenCV
        obs_uint8 = (obs * 255).astype(np.uint8)
        
        # Convert to grayscale for edge detection
        if len(obs_uint8.shape) == 3 and obs_uint8.shape[2] == 3:
            gray = cv2.cvtColor(obs_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs_uint8
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
        # Use edge detection to find rectangular structures
        edges = cv2.Canny(blurred, 30, 150)
        
        # Dilate edges to connect nearby edges
        dilated = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        # Look for rectangular contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # If the polygon has 4 vertices (rectangular) and is large enough
            if len(approx) == 4 and cv2.contourArea(contour) > 200:
                x, y, w, h = cv2.boundingRect(contour)
                # Check aspect ratio - doors are typically taller than wide
                aspect_ratio = float(h) / w
                if 1.5 < aspect_ratio < 4.0 and h > 20:  # Typical door proportions
                    return True
                    
        # Additional check for vertical lines which might indicate a door
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is not None:
            vertical_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10 and abs(y2 - y1) > 30:  # Nearly vertical line
                    vertical_lines += 1
                    
            if vertical_lines >= 2:  # Two vertical lines could be door edges
                return True
                
        return False
        
    except Exception as e:
        # Fallback to simpler detection if advanced method fails
        # Simply look for vertical edges in the image
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor((obs * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (obs * 255).astype(np.uint8)
            
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges = cv2.convertScaleAbs(edges)
        
        # Count strong vertical edges
        strong_edges = np.sum(edges > 100)
        return strong_edges > 300  # Threshold based on observation

def detect_key_collection(previous_keys, current_keys):
    """
    Detect if a key was collected by comparing previous and current key counts
    
    Args:
        previous_keys: Previous key count
        current_keys: Current key count
        
    Returns:
        bool: Whether a key was collected
    """
    return current_keys > previous_keys

def detect_door_opening(previous_obs, current_obs, has_key, previous_has_key):
    """
    Detect if a door was opened by analyzing visual changes and key state
    
    Args:
        previous_obs: Previous observation
        current_obs: Current observation
        has_key: Whether agent currently has a key
        previous_has_key: Whether agent previously had a key
        
    Returns:
        bool: Whether a door was likely opened
    """
    # If agent had a key before but not now, likely used on a door
    if previous_has_key and not has_key:
        return True
        
    # Check for significant changes in the scene that might indicate a door opening
    if previous_obs is None:
        return False
    
    # Handle tuple observations (visual_obs, vector_obs)
    if isinstance(current_obs, tuple):
        current_obs = current_obs[0]  # Extract visual observation
    
    if isinstance(previous_obs, tuple):
        previous_obs = previous_obs[0]  # Extract visual observation
        
    # Convert to appropriate format
    if current_obs.shape[-1] == 3:
        curr = current_obs.copy()
        prev = previous_obs.copy()
    else:
        curr = current_obs.transpose(1, 2, 0) if current_obs.shape[0] == 3 else current_obs
        prev = previous_obs.transpose(1, 2, 0) if previous_obs.shape[0] == 3 else previous_obs
    
    # Calculate difference between frames
    diff = np.mean(np.abs(curr - prev))
    
    # If there's a significant difference and had a key before but not now
    return diff > 0.2 and previous_has_key and not has_key