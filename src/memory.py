import numpy as np

class EnhancedKeyDoorMemory:
    def __init__(self, max_locations=50):
        self.key_locations = {}  # Floor -> list of positions
        self.door_locations = {}  # Floor -> list of positions
        self.successful_key_uses = []  # List of (key_pos, door_pos, floor)
        self.max_locations = max_locations
        self.floor_completion = set()  # Set of completed floors
        self.location_confidence = {}  # Track confidence in each location
        self.last_key_position = None
        self.last_door_position = None
        
    def add_key_location(self, floor, position):
        """Record a location where a key was found"""
        if floor not in self.key_locations:
            self.key_locations[floor] = []
            self.location_confidence[(floor, 'key')] = {}
        
        # Check if this position is already recorded
        is_new = True
        for i, existing_pos in enumerate(self.key_locations[floor]):
            if self._calc_distance(position, existing_pos) < 1.0:
                # Update confidence for this location
                pos_key = self._get_position_key(existing_pos)
                if pos_key not in self.location_confidence[(floor, 'key')]:
                    self.location_confidence[(floor, 'key')][pos_key] = 1
                else:
                    self.location_confidence[(floor, 'key')][pos_key] += 1
                
                # Update position with weighted average for better accuracy
                confidence = self.location_confidence[(floor, 'key')][pos_key]
                weight = 1.0 / confidence  # New positions have less weight as confidence increases
                updated_pos = tuple(
                    (1 - weight) * existing_pos[i] + weight * position[i] 
                    for i in range(min(len(existing_pos), len(position)))
                )
                self.key_locations[floor][i] = updated_pos
                is_new = False
                break
                
        if is_new:
            self.key_locations[floor].append(position)
            pos_key = self._get_position_key(position)
            self.location_confidence[(floor, 'key')][pos_key] = 1
            
            if len(self.key_locations[floor]) > self.max_locations:
                # Remove oldest with lowest confidence
                if len(self.key_locations[floor]) > 0:
                    # Find entry with lowest confidence
                    confidences = [
                        self.location_confidence[(floor, 'key')].get(
                            self._get_position_key(pos), 0
                        ) 
                        for pos in self.key_locations[floor]
                    ]
                    if confidences:
                        min_conf_idx = np.argmin(confidences)
                        removed_pos = self.key_locations[floor].pop(min_conf_idx)
                        removed_key = self._get_position_key(removed_pos)
                        if removed_key in self.location_confidence[(floor, 'key')]:
                            del self.location_confidence[(floor, 'key')][removed_key]
        
        # Remember the last key position for door association
        self.last_key_position = position
    
    def add_door_location(self, floor, position):
        """Record a location where a door was found/used"""
        if floor not in self.door_locations:
            self.door_locations[floor] = []
            self.location_confidence[(floor, 'door')] = {}
            
        # Check if this position is already recorded
        is_new = True
        for i, existing_pos in enumerate(self.door_locations[floor]):
            if self._calc_distance(position, existing_pos) < 1.0:
                # Update confidence for this location
                pos_key = self._get_position_key(existing_pos)
                if pos_key not in self.location_confidence[(floor, 'door')]:
                    self.location_confidence[(floor, 'door')][pos_key] = 1
                else:
                    self.location_confidence[(floor, 'door')][pos_key] += 1
                
                # Update position with weighted average for better accuracy
                confidence = self.location_confidence[(floor, 'door')][pos_key]
                weight = 1.0 / confidence  # New positions have less weight as confidence increases
                updated_pos = tuple(
                    (1 - weight) * existing_pos[i] + weight * position[i] 
                    for i in range(min(len(existing_pos), len(position)))
                )
                self.door_locations[floor][i] = updated_pos
                is_new = False
                break
                
        if is_new:
            self.door_locations[floor].append(position)
            pos_key = self._get_position_key(position)
            self.location_confidence[(floor, 'door')][pos_key] = 1
            
            if len(self.door_locations[floor]) > self.max_locations:
                # Remove oldest with lowest confidence
                if len(self.door_locations[floor]) > 0:
                    # Find entry with lowest confidence
                    confidences = [
                        self.location_confidence[(floor, 'door')].get(
                            self._get_position_key(pos), 0
                        ) 
                        for pos in self.door_locations[floor]
                    ]
                    if confidences:
                        min_conf_idx = np.argmin(confidences)
                        removed_pos = self.door_locations[floor].pop(min_conf_idx)
                        removed_key = self._get_position_key(removed_pos)
                        if removed_key in self.location_confidence[(floor, 'door')]:
                            del self.location_confidence[(floor, 'door')][removed_key]
        
        # Remember the last door position for key association
        self.last_door_position = position
            
    def store_key_door_sequence(self, key_pos, door_pos, floor):
        """Store successful key-door interaction sequence"""
        if key_pos is None or door_pos is None:
            return
            
        self.successful_key_uses.append((key_pos, door_pos, floor))
        if len(self.successful_key_uses) > self.max_locations:
            self.successful_key_uses.pop(0)
        
        # Increase confidence in these locations significantly
        key_pos_key = self._get_position_key(key_pos)
        door_pos_key = self._get_position_key(door_pos)
        
        if (floor, 'key') not in self.location_confidence:
            self.location_confidence[(floor, 'key')] = {}
        if key_pos_key not in self.location_confidence[(floor, 'key')]:
            self.location_confidence[(floor, 'key')][key_pos_key] = 0
        self.location_confidence[(floor, 'key')][key_pos_key] += 3  # Triple confidence boost
        
        if (floor, 'door') not in self.location_confidence:
            self.location_confidence[(floor, 'door')] = {}
        if door_pos_key not in self.location_confidence[(floor, 'door')]:
            self.location_confidence[(floor, 'door')][door_pos_key] = 0
        self.location_confidence[(floor, 'door')][door_pos_key] += 3  # Triple confidence boost
    
    def mark_floor_complete(self, floor):
        """Mark a floor as completed for future reference"""
        self.floor_completion.add(floor)
        
    def is_floor_complete(self, floor):
        """Check if a floor has been completed before"""
        return floor in self.floor_completion
    
    def get_proximity_bonus(self, current_pos, floor, has_key):
        """Get bonus based on proximity to relevant objects"""
        # If floor is already completed, provide small navigation bonus
        if self.is_floor_complete(floor):
            return 0.05  # Small constant bonus for known floors
        
        # If we have successful key-door sequences for this floor, prioritize those patterns
        successful_sequences = [seq for seq in self.successful_key_uses if seq[2] == floor]
        if successful_sequences and len(successful_sequences) > 0:
            # Use the most recent successful sequence as a guide
            key_pos, door_pos, _ = successful_sequences[-1]
            
            if not has_key:
                # Guide to key first
                dist_to_key = self._calc_distance(current_pos, key_pos)
                key_bonus = self._calculate_proximity_reward(dist_to_key, max_dist=10.0, max_reward=1.0)
                return key_bonus
            else:
                # Guide to door once has key
                dist_to_door = self._calc_distance(current_pos, door_pos)
                door_bonus = self._calculate_proximity_reward(dist_to_door, max_dist=10.0, max_reward=2.0)
                return door_bonus
        
        # Standard proximity logic if no successful sequences available
        if not has_key and floor in self.key_locations and self.key_locations[floor]:
            # Find distance to nearest known key, weighted by confidence
            best_bonus = 0
            for key_pos in self.key_locations[floor]:
                dist = self._calc_distance(current_pos, key_pos)
                pos_key = self._get_position_key(key_pos)
                confidence = self.location_confidence.get((floor, 'key'), {}).get(pos_key, 1)
                
                # Higher confidence locations get stronger reward gradient
                confidence_factor = min(2.0, 1.0 + (confidence / 10.0))  # Cap at 2.0x multiplier
                bonus = self._calculate_proximity_reward(dist, max_dist=8.0, max_reward=0.8) * confidence_factor
                best_bonus = max(best_bonus, bonus)
            
            return best_bonus
                
        elif has_key and floor in self.door_locations and self.door_locations[floor]:
            # Find distance to nearest known door, weighted by confidence
            best_bonus = 0
            for door_pos in self.door_locations[floor]:
                dist = self._calc_distance(current_pos, door_pos)
                pos_key = self._get_position_key(door_pos)
                confidence = self.location_confidence.get((floor, 'door'), {}).get(pos_key, 1)
                
                # Higher confidence locations get stronger reward gradient
                confidence_factor = min(2.0, 1.0 + (confidence / 10.0))  # Cap at 2.0x multiplier
                bonus = self._calculate_proximity_reward(dist, max_dist=8.0, max_reward=1.5) * confidence_factor
                best_bonus = max(best_bonus, bonus)
            
            return best_bonus
                
        return 0.0
    
    def _calculate_proximity_reward(self, distance, max_dist=8.0, max_reward=1.0):
        """Calculate proximity reward with a smoother gradient"""
        if distance >= max_dist:
            return 0.0
        
        # Quadratic falloff for smoother gradient - higher reward when closer
        normalized_dist = distance / max_dist
        reward = max_reward * (1 - normalized_dist**2)
        return max(0, reward)
    
    def get_directions_to_target(self, current_pos, floor, has_key):
        """Get directional advice for agent based on memory"""
        # If we have successful key-door sequences for this floor, prioritize those
        successful_sequences = [seq for seq in self.successful_key_uses if seq[2] == floor]
        if successful_sequences and len(successful_sequences) > 0:
            # Use the most recent successful sequence as a guide
            key_pos, door_pos, _ = successful_sequences[-1]
            
            if not has_key:
                # Direct to the key position from the successful sequence
                dist = self._calc_distance(current_pos, key_pos)
                direction = [key_pos[i] - current_pos[i] for i in range(min(len(key_pos), len(current_pos)))]
                return {
                    'target_type': 'key',
                    'distance': dist,
                    'direction': direction,
                    'confidence': 2.0  # High confidence since this was successful before
                }
            else:
                # Direct to the door position from the successful sequence
                dist = self._calc_distance(current_pos, door_pos)
                direction = [door_pos[i] - current_pos[i] for i in range(min(len(door_pos), len(current_pos)))]
                return {
                    'target_type': 'door',
                    'distance': dist,
                    'direction': direction,
                    'confidence': 2.0  # High confidence since this was successful before
                }
        
        # Standard search logic if no successful sequences
        if not has_key and floor in self.key_locations and self.key_locations[floor]:
            # Find nearest known key weighted by confidence
            min_dist = float('inf')
            nearest_key = None
            highest_confidence = 0
            
            for key_pos in self.key_locations[floor]:
                dist = self._calc_distance(current_pos, key_pos)
                pos_key = self._get_position_key(key_pos)
                confidence = self.location_confidence.get((floor, 'key'), {}).get(pos_key, 1)
                
                # Consider both distance and confidence - prefer high confidence even if slightly farther
                confidence_adjusted_dist = dist / (1 + 0.2 * min(5, confidence))
                
                if confidence_adjusted_dist < min_dist:
                    min_dist = confidence_adjusted_dist
                    nearest_key = key_pos
                    highest_confidence = confidence
                    
            if nearest_key:
                # Calculate direction vector
                direction = [nearest_key[i] - current_pos[i] for i in range(min(len(nearest_key), len(current_pos)))]
                actual_dist = self._calc_distance(current_pos, nearest_key)  # Actual distance for return
                return {
                    'target_type': 'key',
                    'distance': actual_dist,
                    'direction': direction,
                    'confidence': highest_confidence
                }
                
        elif has_key and floor in self.door_locations and self.door_locations[floor]:
            # Find nearest known door weighted by confidence
            min_dist = float('inf')
            nearest_door = None
            highest_confidence = 0
            
            for door_pos in self.door_locations[floor]:
                dist = self._calc_distance(current_pos, door_pos)
                pos_key = self._get_position_key(door_pos)
                confidence = self.location_confidence.get((floor, 'door'), {}).get(pos_key, 1)
                
                # Consider both distance and confidence
                confidence_adjusted_dist = dist / (1 + 0.2 * min(5, confidence))
                
                if confidence_adjusted_dist < min_dist:
                    min_dist = confidence_adjusted_dist
                    nearest_door = door_pos
                    highest_confidence = confidence
                    
            if nearest_door:
                # Calculate direction vector
                direction = [nearest_door[i] - current_pos[i] for i in range(min(len(nearest_door), len(current_pos)))]
                actual_dist = self._calc_distance(current_pos, nearest_door)  # Actual distance for return
                return {
                    'target_type': 'door',
                    'distance': actual_dist,
                    'direction': direction,
                    'confidence': highest_confidence
                }
                
        return None
    
    def _calc_distance(self, pos1, pos2):
        """Calculate Euclidean distance between positions"""
        return sum((pos1[i] - pos2[i])**2 for i in range(min(len(pos1), len(pos2))))**0.5
    
    def _get_position_key(self, pos):
        """Create a string key for a position by rounding coordinates"""
        if pos is None:
            return None
        # Round to 1 decimal place for better grouping of nearby positions
        rounded_pos = tuple(round(p, 1) for p in pos)
        return str(rounded_pos)
        
    def is_key_location_nearby(self, floor, position, threshold=3.0):
        """Check if current position is near a known key location"""
        if floor not in self.key_locations:
            return False
            
        for key_pos in self.key_locations[floor]:
            if self._calc_distance(position, key_pos) < threshold:
                return True
                
        return False
        
    def is_door_location_nearby(self, floor, position, threshold=3.0):
        """Check if current position is near a known door location"""
        if floor not in self.door_locations:
            return False
            
        for door_pos in self.door_locations[floor]:
            if self._calc_distance(position, door_pos) < threshold:
                return True
                
        return False