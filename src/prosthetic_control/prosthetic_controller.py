"""
Prosthetic Controller
Interface for controlling prosthetic arm based on SNN predictions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time


class ProstheticController:
    """
    Controls a prosthetic arm based on finger motion predictions.
    Provides a mock interface for demonstration purposes.
    """
    
    # Finger position ranges (0-100%)
    FINGER_POSITIONS = {
        'Thumb': 0,
        'Index': 0,
        'Middle': 0,
        'Ring': 0,
        'Pinky': 0
    }
    
    def __init__(self,
                 smoothing_window: int = 3,
                 confidence_threshold: float = 0.6,
                 movement_speed: float = 10.0):
        """
        Initialize the prosthetic controller.
        
        Args:
            smoothing_window: Number of predictions to smooth over
            confidence_threshold: Minimum confidence for action
            movement_speed: Speed of finger movement (degrees per second)
        """
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.movement_speed = movement_speed
        
        # Current finger positions (0-100%)
        self.positions = self.FINGER_POSITIONS.copy()
        
        # Prediction history for smoothing
        self.prediction_history = []
        
        # Target positions
        self.targets = self.FINGER_POSITIONS.copy()
        
        # Movement state
        self.is_moving = False
        self.last_update_time = time.time()
        
    def update_prediction(self, finger_name: str, confidence: float = 1.0):
        """
        Update with a new finger prediction.
        
        Args:
            finger_name: Name of the predicted finger ('Thumb', 'Index', etc.)
            confidence: Prediction confidence (0-1)
        """
        # Add to history
        self.prediction_history.append((finger_name, confidence))
        
        # Keep only recent history
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)
        
        # Compute smoothed prediction
        smoothed_prediction = self._compute_smoothed_prediction()
        
        if smoothed_prediction and smoothed_prediction[1] >= self.confidence_threshold:
            self._set_target(smoothed_prediction[0])
    
    def _compute_smoothed_prediction(self) -> Optional[Tuple[str, float]]:
        """
        Compute smoothed prediction from history.
        
        Returns:
            Tuple of (finger_name, confidence) or None
        """
        if not self.prediction_history:
            return None
        
        # Count predictions for each finger
        finger_counts = {}
        total_confidence = {}
        
        for finger, conf in self.prediction_history:
            finger_counts[finger] = finger_counts.get(finger, 0) + 1
            total_confidence[finger] = total_confidence.get(finger, 0) + conf
        
        # Find most common prediction
        max_count = max(finger_counts.values())
        most_common = [f for f, c in finger_counts.items() if c == max_count]
        
        if len(most_common) == 1:
            finger = most_common[0]
            avg_confidence = total_confidence[finger] / finger_counts[finger]
            return (finger, avg_confidence)
        
        return None
    
    def _set_target(self, finger_name: str):
        """
        Set target position for a finger.
        
        Args:
            finger_name: Name of finger to move
        """
        # Toggle between open (0%) and closed (100%)
        if self.targets[finger_name] < 50:
            self.targets[finger_name] = 100
        else:
            self.targets[finger_name] = 0
        
        self.is_moving = True
    
    def update_positions(self) -> Dict[str, float]:
        """
        Update finger positions towards targets.
        
        Returns:
            Dictionary of current finger positions
        """
        if not self.is_moving:
            return self.positions.copy()
        
        # Calculate time delta
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update each finger position
        any_moving = False
        for finger in self.positions:
            target = self.targets[finger]
            current = self.positions[finger]
            
            if abs(target - current) > 1:
                # Calculate movement
                direction = 1 if target > current else -1
                movement = self.movement_speed * dt
                
                # Update position
                new_pos = current + direction * movement
                
                # Clamp to target
                if direction > 0:
                    new_pos = min(new_pos, target)
                else:
                    new_pos = max(new_pos, target)
                
                self.positions[finger] = new_pos
                any_moving = True
            else:
                # Reached target
                self.positions[finger] = target
        
        self.is_moving = any_moving
        
        return self.positions.copy()
    
    def set_finger_position(self, finger_name: str, position: float):
        """
        Directly set a finger position.
        
        Args:
            finger_name: Name of finger
            position: Target position (0-100%)
        """
        position = np.clip(position, 0, 100)
        self.targets[finger_name] = position
        self.is_moving = True
    
    def reset_all(self):
        """Reset all fingers to open position."""
        for finger in self.targets:
            self.targets[finger] = 0
        self.is_moving = True
        self.prediction_history.clear()
    
    def get_positions(self) -> Dict[str, float]:
        """Get current finger positions."""
        return self.positions.copy()
    
    def get_status(self) -> Dict:
        """
        Get controller status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'positions': self.positions.copy(),
            'targets': self.targets.copy(),
            'is_moving': self.is_moving,
            'prediction_history': self.prediction_history.copy()
        }
    
    def send_command(self, command: str) -> bool:
        """
        Send a command to the prosthetic (mock implementation).
        
        Args:
            command: Command string
            
        Returns:
            Success status
        """
        # Mock implementation - in real system, this would send to hardware
        print(f"[Mock Prosthetic] Command: {command}")
        return True
    
    def execute_action(self, finger_name: str, action: str = 'toggle') -> bool:
        """
        Execute an action on a finger.
        
        Args:
            finger_name: Name of finger
            action: Action to perform ('toggle', 'close', 'open')
            
        Returns:
            Success status
        """
        if finger_name not in self.positions:
            return False
        
        if action == 'toggle':
            self._set_target(finger_name)
        elif action == 'close':
            self.targets[finger_name] = 100
            self.is_moving = True
        elif action == 'open':
            self.targets[finger_name] = 0
            self.is_moving = True
        else:
            return False
        
        # Send command to hardware (mock)
        self.send_command(f"{action.upper()}_{finger_name}")
        
        return True
    
    def get_control_signal(self) -> np.ndarray:
        """
        Get control signal for all fingers.
        
        Returns:
            Array of finger positions [Thumb, Index, Middle, Ring, Pinky]
        """
        fingers_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        return np.array([self.positions[f] for f in fingers_order])
