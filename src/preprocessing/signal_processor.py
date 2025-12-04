"""
Signal Processor
Applies filtering, normalization, and preprocessing to EEG signals
"""

import numpy as np
from scipy import signal
from typing import Tuple, Optional


class SignalProcessor:
    """
    Processes EEG signals with filtering and normalization.
    """
    
    def __init__(self, sampling_rate: int = 250):
        """
        Initialize the signal processor.
        
        Args:
            sampling_rate: Sampling rate of EEG data in Hz
        """
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, 
                       low_freq: float = 0.5, 
                       high_freq: float = 50.0,
                       order: int = 5) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            order: Filter order
            
        Returns:
            Filtered data with same shape as input
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each sample and channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
        
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, 
                     notch_freq: float = 50.0,
                     quality_factor: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove powerline interference.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            notch_freq: Frequency to notch out (typically 50 or 60 Hz)
            quality_factor: Quality factor of the notch filter
            
        Returns:
            Filtered data with same shape as input
        """
        # Design notch filter
        b, a = signal.iirnotch(notch_freq, quality_factor, self.sampling_rate)
        
        # Apply filter to each sample and channel
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
        
        return filtered_data
    
    def normalize(self, data: np.ndarray, 
                  method: str = 'zscore') -> np.ndarray:
        """
        Normalize EEG data.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized data with same shape as input
        """
        normalized_data = np.zeros_like(data)
        
        if method == 'zscore':
            # Z-score normalization (mean=0, std=1)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    mean = np.mean(data[i, j, :])
                    std = np.std(data[i, j, :])
                    if std > 0:
                        normalized_data[i, j, :] = (data[i, j, :] - mean) / std
                    else:
                        normalized_data[i, j, :] = data[i, j, :] - mean
                        
        elif method == 'minmax':
            # Min-max normalization (0 to 1)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    min_val = np.min(data[i, j, :])
                    max_val = np.max(data[i, j, :])
                    if max_val > min_val:
                        normalized_data[i, j, :] = (data[i, j, :] - min_val) / (max_val - min_val)
                    else:
                        normalized_data[i, j, :] = data[i, j, :]
                        
        elif method == 'robust':
            # Robust normalization using median and IQR
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    median = np.median(data[i, j, :])
                    q75, q25 = np.percentile(data[i, j, :], [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        normalized_data[i, j, :] = (data[i, j, :] - median) / iqr
                    else:
                        normalized_data[i, j, :] = data[i, j, :] - median
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized_data
    
    def apply_baseline_correction(self, data: np.ndarray, 
                                  baseline_window: Tuple[int, int]) -> np.ndarray:
        """
        Apply baseline correction by subtracting baseline mean.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            baseline_window: Tuple of (start_idx, end_idx) for baseline period
            
        Returns:
            Baseline-corrected data
        """
        start_idx, end_idx = baseline_window
        corrected_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                baseline_mean = np.mean(data[i, j, start_idx:end_idx])
                corrected_data[i, j, :] = data[i, j, :] - baseline_mean
        
        return corrected_data
    
    def downsample(self, data: np.ndarray, 
                   target_rate: int) -> Tuple[np.ndarray, int]:
        """
        Downsample EEG data to target sampling rate.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            target_rate: Target sampling rate in Hz
            
        Returns:
            Tuple of (downsampled data, new sampling rate)
        """
        if target_rate >= self.sampling_rate:
            return data, self.sampling_rate
        
        downsample_factor = self.sampling_rate // target_rate
        
        downsampled_data = data[:, :, ::downsample_factor]
        new_rate = self.sampling_rate // downsample_factor
        
        return downsampled_data, new_rate
    
    def preprocess_pipeline(self, data: np.ndarray,
                           apply_bandpass: bool = True,
                           apply_notch: bool = True,
                           apply_normalize: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            data: Raw EEG data
            apply_bandpass: Whether to apply bandpass filter
            apply_notch: Whether to apply notch filter
            apply_normalize: Whether to normalize data
            
        Returns:
            Preprocessed data
        """
        processed = data.copy()
        
        if apply_bandpass:
            processed = self.bandpass_filter(processed)
        
        if apply_notch:
            processed = self.notch_filter(processed)
        
        if apply_normalize:
            processed = self.normalize(processed)
        
        return processed
