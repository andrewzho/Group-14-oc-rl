class EnhancedKeyDoorMemory:
    def __init__(self, max_locations=50):
        self.key_locations = {}  # Floor -> list of positions
        self.door_locations = {}  # Floor -> list of positions
        self.successful_key_uses = []  # List of (key_pos, door_pos, floor)
        self.max_locations = max_locations
        self.floor_completion = set()  # Set of completed floors
        
    def add_key_location(self, floor, position):
        """Record a location where a key was found"""
        if floor not in self.key_locations:
            self.key_locations[floor] = []
        
        # Check if this position is already recorded
        for existing_pos in self.key_locations[floor]:
            if self._calc_distance(position, existing_pos) < 1.0:
                return  # Skip if already have a similar position
                
        self.key_locations[floor].append(position)
        if len(self.key_locations[floor]) > self.max_locations:
            self.key_locations[floor].pop(0)  # Remove oldest
    
    def add_door_location(self, floor, position):
        """Record a location where a door was used"""
        if floor not in self.door_locations:
            self.door_locations[floor] = []
            
        # Check if this position is already recorded
        for existing_pos in self.door_locations[floor]:
            if self._calc_distance(position, existing_pos) < 1.0:
                return  # Skip if already have a similar position
                
        self.door_locations[floor].append(position)
        if len(self.door_locations[floor]) > self.max_locations:
            self.door_locations[floor].pop(0)  # Remove oldest
            
    def store_key_door_sequence(self, key_pos, door_pos, floor):
        """Store successful key-door interaction sequence"""
        self.successful_key_uses.append((key_pos, door_pos, floor))
        if len(self.successful_key_uses) > self.max_locations:
            self.successful_key_uses.pop(0)
    
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
            
        if not has_key and floor in self.key_locations and self.key_locations[floor]:
            # Find distance to nearest known key
            min_dist = float('inf')
            for key_pos in self.key_locations[floor]:
                dist = self._calc_distance(current_pos, key_pos)
                min_dist = min(min_dist, dist)
            
            # Convert to proximity score (closer = higher reward)
            if min_dist < float('inf'):
                # Sharper reward gradient near the key
                if min_dist < 2.0:
                    return max(0, 0.8 - 0.3 * min_dist)
                else:
                    return max(0, 0.5 - 0.1 * min_dist)  # Max 0.5 bonus, linearly decreasing
                
        elif has_key and floor in self.door_locations and self.door_locations[floor]:
            # Find distance to nearest known door
            min_dist = float('inf')
            for door_pos in self.door_locations[floor]:
                dist = self._calc_distance(current_pos, door_pos)
                min_dist = min(min_dist, dist)
            
            # Convert to proximity score (closer = higher reward)
            if min_dist < float('inf'):
                # Sharper gradient when we're close to the door
                if min_dist < 2.0:
                    return max(0, 1.5 - 0.5 * min_dist)  # Higher bonus when very close
                else:
                    return max(0, 1.0 - 0.2 * min_dist)  # Max 1.0 bonus, linearly decreasing
                
        return 0.0
    
    def get_directions_to_target(self, current_pos, floor, has_key):
        """Get directional advice for agent based on memory"""
        if not has_key and floor in self.key_locations and self.key_locations[floor]:
            # Find nearest known key
            min_dist = float('inf')
            nearest_key = None
            for key_pos in self.key_locations[floor]:
                dist = self._calc_distance(current_pos, key_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_key = key_pos
                    
            if nearest_key:
                # Calculate direction vector
                direction = [nearest_key[i] - current_pos[i] for i in range(min(len(nearest_key), len(current_pos)))]
                return {
                    'target_type': 'key',
                    'distance': min_dist,
                    'direction': direction
                }
                
        elif has_key and floor in self.door_locations and self.door_locations[floor]:
            # Find nearest known door
            min_dist = float('inf')
            nearest_door = None
            for door_pos in self.door_locations[floor]:
                dist = self._calc_distance(current_pos, door_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_door = door_pos
                    
            if nearest_door:
                # Calculate direction vector
                direction = [nearest_door[i] - current_pos[i] for i in range(min(len(nearest_door), len(current_pos)))]
                return {
                    'target_type': 'door',
                    'distance': min_dist,
                    'direction': direction
                }
                
        return None
    
    def _calc_distance(self, pos1, pos2):
        """Calculate Euclidean distance between positions"""
        return sum((pos1[i] - pos2[i])**2 for i in range(min(len(pos1), len(pos2))))**0.5
        
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