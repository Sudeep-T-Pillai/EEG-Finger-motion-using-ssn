"""
Spike Encoder
Converts continuous EEG features into spike trains for SNN input
"""

import numpy as np
import torch
from typing import Optional


class SpikeEncoder:
    """
    Encodes continuous-valued features into spike trains using rate coding.
    """
    
    def __init__(self, encoding_method: str = 'rate', time_steps: int = 100):
        """
        Initialize the spike encoder.
        
        Args:
            encoding_method: Method for spike encoding ('rate', 'latency', 'temporal')
            time_steps: Number of time steps for spike train
        """
        self.encoding_method = encoding_method
        self.time_steps = time_steps
        
    def rate_encoding(self, features: np.ndarray, max_rate: float = 1.0) -> torch.Tensor:
        """
        Rate encoding: Higher feature values produce higher spike rates.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            max_rate: Maximum spike rate (probability of spike per time step)
            
        Returns:
            Spike trains of shape (n_samples, time_steps, n_features)
        """
        n_samples, n_features = features.shape
        
        # Ensure features are in [0, 1] range
        features_norm = np.clip(features, 0, 1)
        
        # Generate spikes based on feature values
        spike_trains = []
        for sample in features_norm:
            # For each time step, generate spikes with probability proportional to feature value
            spikes = np.random.rand(self.time_steps, n_features) < (sample * max_rate)
            spike_trains.append(spikes.astype(np.float32))
        
        # Return tensor with gradient tracking enabled
        return torch.tensor(np.array(spike_trains), dtype=torch.float32, requires_grad=False)
    
    def latency_encoding(self, features: np.ndarray, max_latency: Optional[int] = None) -> torch.Tensor:
        """
        Latency encoding: Higher feature values produce earlier spikes.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            max_latency: Maximum latency in time steps
            
        Returns:
            Spike trains of shape (n_samples, time_steps, n_features)
        """
        if max_latency is None:
            max_latency = self.time_steps - 1
        
        n_samples, n_features = features.shape
        spike_trains = np.zeros((n_samples, self.time_steps, n_features), dtype=np.float32)
        
        # Ensure features are in [0, 1] range
        features_norm = np.clip(features, 0.01, 1)  # Avoid zero
        
        for i, sample in enumerate(features_norm):
            # Calculate spike time for each feature
            # Higher values -> earlier spikes (lower latency)
            latencies = (max_latency * (1 - sample)).astype(int)
            latencies = np.clip(latencies, 0, self.time_steps - 1)
            
            for j, latency in enumerate(latencies):
                spike_trains[i, latency, j] = 1.0
        
        return torch.tensor(spike_trains, dtype=torch.float32)
    
    def temporal_encoding(self, features: np.ndarray, n_spikes: int = 5) -> torch.Tensor:
        """
        Temporal encoding: Distribute spikes temporally based on feature values.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            n_spikes: Number of spikes per feature
            
        Returns:
            Spike trains of shape (n_samples, time_steps, n_features)
        """
        n_samples, n_features = features.shape
        spike_trains = np.zeros((n_samples, self.time_steps, n_features), dtype=np.float32)
        
        # Ensure features are in [0, 1] range
        features_norm = np.clip(features, 0, 1)
        
        for i, sample in enumerate(features_norm):
            for j, value in enumerate(sample):
                if value > 0:
                    # Generate spike times based on feature value
                    spike_interval = self.time_steps / (n_spikes + 1)
                    base_times = np.arange(1, n_spikes + 1) * spike_interval
                    
                    # Add jitter proportional to (1 - value)
                    jitter = (1 - value) * spike_interval * 0.3
                    spike_times = (base_times + np.random.randn(n_spikes) * jitter).astype(int)
                    spike_times = np.clip(spike_times, 0, self.time_steps - 1)
                    
                    spike_trains[i, spike_times, j] = 1.0
        
        return torch.tensor(spike_trains, dtype=torch.float32)
    
    def encode(self, features: np.ndarray) -> torch.Tensor:
        """
        Encode features into spikes using the selected method.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Spike trains
        """
        if self.encoding_method == 'rate':
            return self.rate_encoding(features)
        elif self.encoding_method == 'latency':
            return self.latency_encoding(features)
        elif self.encoding_method == 'temporal':
            return self.temporal_encoding(features)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
    
    def decode_spikes(self, spike_trains: torch.Tensor, method: str = 'rate') -> np.ndarray:
        """
        Decode spike trains back to continuous values.
        
        Args:
            spike_trains: Spike trains of shape (n_samples, time_steps, n_features)
            method: Decoding method ('rate' or 'time_to_first_spike')
            
        Returns:
            Decoded values of shape (n_samples, n_features)
        """
        if method == 'rate':
            # Average spike rate over time
            decoded = torch.mean(spike_trains, dim=1).numpy()
        elif method == 'time_to_first_spike':
            # Time to first spike (inverse of latency)
            spike_trains_np = spike_trains.numpy()
            n_samples, time_steps, n_features = spike_trains_np.shape
            decoded = np.zeros((n_samples, n_features))
            
            for i in range(n_samples):
                for j in range(n_features):
                    spike_times = np.where(spike_trains_np[i, :, j] > 0)[0]
                    if len(spike_times) > 0:
                        # Normalize to [0, 1]
                        decoded[i, j] = 1 - (spike_times[0] / time_steps)
                    else:
                        decoded[i, j] = 0
        else:
            raise ValueError(f"Unknown decoding method: {method}")
        
        return decoded
