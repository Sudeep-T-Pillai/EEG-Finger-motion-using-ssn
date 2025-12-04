"""
Realtime Predictor
Handles real-time EEG data streaming and prediction
"""

import numpy as np
from typing import Optional, Callable
import time
from collections import deque


class RealtimePredictor:
    """
    Manages real-time EEG data collection and prediction.
    """
    
    def __init__(self,
                 classifier,
                 n_channels: int = 64,
                 window_size: int = 500,
                 overlap: int = 250,
                 sampling_rate: int = 250):
        """
        Initialize real-time predictor.
        
        Args:
            classifier: Trained FingerMotionClassifier instance
            n_channels: Number of EEG channels
            window_size: Size of prediction window in samples
            overlap: Overlap between windows in samples
            sampling_rate: Sampling rate in Hz
        """
        self.classifier = classifier
        self.n_channels = n_channels
        self.window_size = window_size
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        
        # Buffer for incoming data
        self.buffer = deque(maxlen=window_size)
        
        # Prediction callback
        self.prediction_callback = None
        
        # State
        self.is_running = False
        self.prediction_count = 0
        
    def set_prediction_callback(self, callback: Callable[[str, float], None]):
        """
        Set callback function for predictions.
        
        Args:
            callback: Function that takes (finger_name, confidence) as arguments
        """
        self.prediction_callback = callback
    
    def add_data(self, data: np.ndarray):
        """
        Add new EEG data to buffer.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
        """
        if data.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {data.shape[0]}")
        
        # Add samples to buffer
        for i in range(data.shape[1]):
            self.buffer.append(data[:, i])
    
    def predict_from_buffer(self) -> Optional[tuple]:
        """
        Make prediction from current buffer.
        
        Returns:
            Tuple of (finger_name, confidence) or None if buffer not full
        """
        if len(self.buffer) < self.window_size:
            return None
        
        # Convert buffer to array
        window_data = np.array(self.buffer).T  # Shape: (n_channels, window_size)
        
        # Add batch dimension
        window_data = window_data[np.newaxis, :, :]  # Shape: (1, n_channels, window_size)
        
        # Make prediction
        prediction = self.classifier.predict(window_data)[0]
        finger_name = self.classifier.FINGER_LABELS[prediction]
        
        # Confidence would require spike rates - simplified here
        confidence = 0.8  # Mock confidence
        
        self.prediction_count += 1
        
        return (finger_name, confidence)
    
    def process_window(self) -> Optional[tuple]:
        """
        Process the current window and shift buffer.
        
        Returns:
            Prediction result or None
        """
        result = self.predict_from_buffer()
        
        if result and self.prediction_callback:
            self.prediction_callback(result[0], result[1])
        
        # Shift buffer by (window_size - overlap) samples
        shift_amount = self.window_size - self.overlap
        for _ in range(shift_amount):
            if len(self.buffer) > 0:
                self.buffer.popleft()
        
        return result
    
    def simulate_realtime_stream(self,
                                 eeg_data: np.ndarray,
                                 labels: Optional[np.ndarray] = None,
                                 callback: Optional[Callable] = None) -> list:
        """
        Simulate real-time streaming with recorded data.
        
        Args:
            eeg_data: EEG data (n_samples, n_channels, n_timepoints)
            labels: Optional ground truth labels
            callback: Optional callback for each prediction
            
        Returns:
            List of predictions
        """
        predictions = []
        
        if callback:
            self.set_prediction_callback(callback)
        
        # Process each sample
        for i, sample in enumerate(eeg_data):
            # Add data to buffer (simulate real-time arrival)
            self.add_data(sample)
            
            # Try to make prediction
            result = self.predict_from_buffer()
            
            if result:
                predictions.append(result)
                
                # Small delay to simulate real-time
                time.sleep(0.01)
        
        return predictions
    
    def reset_buffer(self):
        """Clear the data buffer."""
        self.buffer.clear()
        self.prediction_count = 0
    
    def get_buffer_status(self) -> dict:
        """
        Get buffer status information.
        
        Returns:
            Dictionary with buffer info
        """
        return {
            'buffer_size': len(self.buffer),
            'window_size': self.window_size,
            'is_full': len(self.buffer) >= self.window_size,
            'prediction_count': self.prediction_count
        }
    
    def start(self):
        """Start real-time prediction."""
        self.is_running = True
        self.reset_buffer()
    
    def stop(self):
        """Stop real-time prediction."""
        self.is_running = False
    
    def is_active(self) -> bool:
        """Check if predictor is active."""
        return self.is_running


class StreamSimulator:
    """
    Simulates a real-time EEG data stream for testing.
    """
    
    def __init__(self,
                 data: np.ndarray,
                 labels: Optional[np.ndarray] = None,
                 sampling_rate: int = 250,
                 chunk_size: int = 10):
        """
        Initialize stream simulator.
        
        Args:
            data: EEG data (n_samples, n_channels, n_timepoints)
            labels: Optional labels
            sampling_rate: Sampling rate
            chunk_size: Number of samples to send at once
        """
        self.data = data
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        
        self.current_idx = 0
        self.is_streaming = False
    
    def start_stream(self):
        """Start the stream."""
        self.is_streaming = True
        self.current_idx = 0
    
    def stop_stream(self):
        """Stop the stream."""
        self.is_streaming = False
    
    def get_next_chunk(self) -> Optional[tuple]:
        """
        Get next chunk of data.
        
        Returns:
            Tuple of (data_chunk, label) or None if stream ended
        """
        if not self.is_streaming or self.current_idx >= len(self.data):
            return None
        
        end_idx = min(self.current_idx + self.chunk_size, len(self.data))
        chunk = self.data[self.current_idx:end_idx]
        
        label = None
        if self.labels is not None:
            label = self.labels[self.current_idx]
        
        self.current_idx = end_idx
        
        return (chunk, label)
    
    def has_more_data(self) -> bool:
        """Check if more data is available."""
        return self.current_idx < len(self.data)
