"""
EEG Data Loader
Handles loading EEG data from various formats (EDF, FIF, MAT)
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings


class EEGDataLoader:
    """
    Loads EEG data from multiple file formats and provides a unified interface.
    """
    
    def __init__(self, sampling_rate: int = 250):
        """
        Initialize the EEG data loader.
        
        Args:
            sampling_rate: Expected sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.channel_names = None
        self.data = None
        self.labels = None
        
    def load_from_array(self, data: np.ndarray, labels: np.ndarray, 
                        channel_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EEG data from numpy arrays.
        
        Args:
            data: EEG data array of shape (n_samples, n_channels, n_timepoints)
            labels: Labels array of shape (n_samples,)
            channel_names: Optional list of channel names
            
        Returns:
            Tuple of (data, labels)
        """
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D (samples, channels, time), got shape {data.shape}")
        
        if len(labels) != data.shape[0]:
            raise ValueError(f"Number of labels ({len(labels)}) must match number of samples ({data.shape[0]})")
        
        self.data = data
        self.labels = labels
        self.channel_names = channel_names or [f"Ch{i}" for i in range(data.shape[1])]
        
        return self.data, self.labels
    
    def load_from_mne_raw(self, raw_data, events: np.ndarray, 
                          event_id: Dict[str, int], 
                          tmin: float = -0.5, tmax: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EEG data from MNE Raw object with events.
        
        Args:
            raw_data: MNE Raw object
            events: Events array from MNE
            event_id: Dictionary mapping event names to IDs
            tmin: Start time before event (seconds)
            tmax: End time after event (seconds)
            
        Returns:
            Tuple of (data, labels)
        """
        try:
            import mne
        except ImportError:
            raise ImportError("MNE package required for loading raw data. Install with: pip install mne")
        
        # Create epochs
        epochs = mne.Epochs(raw_data, events, event_id, tmin, tmax, 
                           baseline=None, preload=True, verbose=False)
        
        # Extract data and labels
        self.data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        self.labels = epochs.events[:, -1]  # Event IDs
        self.channel_names = epochs.ch_names
        self.sampling_rate = int(epochs.info['sfreq'])
        
        return self.data, self.labels
    
    def generate_synthetic_data(self, n_samples: int = 100, 
                                n_channels: int = 64, 
                                n_timepoints: int = 500,
                                n_classes: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic EEG data for testing purposes.
        
        Args:
            n_samples: Number of samples to generate
            n_channels: Number of EEG channels
            n_timepoints: Number of time points per trial
            n_classes: Number of finger classes (default: 5 fingers)
            
        Returns:
            Tuple of (data, labels)
        """
        # Generate synthetic EEG-like signals with different frequency bands
        np.random.seed(42)
        
        data_list = []
        labels_list = []
        
        for class_id in range(n_classes):
            for _ in range(n_samples // n_classes):
                # Create base signal with noise
                signal = np.random.randn(n_channels, n_timepoints) * 0.5
                
                # Add class-specific frequency components
                t = np.linspace(0, n_timepoints / self.sampling_rate, n_timepoints)
                
                # Different frequency patterns for different fingers
                freq = 8 + class_id * 2  # Alpha to beta range (8-18 Hz)
                for ch in range(n_channels):
                    # Add sinusoidal component with some spatial variation
                    phase = np.random.rand() * 2 * np.pi
                    amplitude = 0.5 + np.random.rand() * 0.5
                    signal[ch, :] += amplitude * np.sin(2 * np.pi * freq * t + phase)
                
                data_list.append(signal)
                labels_list.append(class_id)
        
        self.data = np.array(data_list)
        self.labels = np.array(labels_list)
        self.channel_names = [f"Ch{i}" for i in range(n_channels)]
        
        return self.data, self.labels
    
    def get_data_info(self) -> Dict:
        """
        Get information about loaded data.
        
        Returns:
            Dictionary with data information
        """
        if self.data is None:
            return {"status": "No data loaded"}
        
        return {
            "n_samples": self.data.shape[0],
            "n_channels": self.data.shape[1],
            "n_timepoints": self.data.shape[2],
            "sampling_rate": self.sampling_rate,
            "n_classes": len(np.unique(self.labels)) if self.labels is not None else 0,
            "channel_names": self.channel_names
        }
