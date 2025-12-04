"""
Feature Extractor
Extracts relevant features from preprocessed EEG signals for SNN input
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional


class FeatureExtractor:
    """
    Extracts time-domain, frequency-domain, and statistical features from EEG signals.
    """
    
    def __init__(self, sampling_rate: int = 250):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_rate: Sampling rate of EEG data in Hz
        """
        self.sampling_rate = sampling_rate
        
    def extract_time_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract time-domain features from EEG data.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            
        Returns:
            Feature array of shape (n_samples, n_channels * n_features)
        """
        n_samples, n_channels, n_timepoints = data.shape
        features_list = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_channels):
                channel_data = data[i, j, :]
                
                # Statistical features
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                var = np.var(channel_data)
                rms = np.sqrt(np.mean(channel_data**2))
                skewness = skew(channel_data)
                kurt = kurtosis(channel_data)
                
                # Peak-to-peak amplitude
                ptp = np.ptp(channel_data)
                
                # Zero crossing rate
                zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
                
                sample_features.extend([mean, std, var, rms, skewness, kurt, ptp, zero_crossings])
            
            features_list.append(sample_features)
        
        return np.array(features_list)
    
    def extract_frequency_features(self, data: np.ndarray,
                                   bands: Optional[Dict[str, tuple]] = None) -> np.ndarray:
        """
        Extract frequency-domain features using power spectral density.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            bands: Dictionary of frequency bands {name: (low_freq, high_freq)}
            
        Returns:
            Feature array of shape (n_samples, n_channels * n_bands)
        """
        if bands is None:
            # Default EEG frequency bands
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            }
        
        n_samples, n_channels, n_timepoints = data.shape
        features_list = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_channels):
                channel_data = data[i, j, :]
                
                # Compute power spectral density
                freqs, psd = sp_signal.welch(channel_data, self.sampling_rate, 
                                            nperseg=min(256, n_timepoints))
                
                # Extract power in each frequency band
                for band_name, (low, high) in bands.items():
                    idx_band = np.logical_and(freqs >= low, freqs <= high)
                    band_power = np.trapz(psd[idx_band], freqs[idx_band])
                    sample_features.append(band_power)
            
            features_list.append(sample_features)
        
        return np.array(features_list)
    
    def extract_wavelet_features(self, data: np.ndarray, 
                                wavelet: str = 'db4',
                                level: int = 4) -> np.ndarray:
        """
        Extract wavelet-based features.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            wavelet: Wavelet type (default: Daubechies 4)
            level: Decomposition level
            
        Returns:
            Feature array with wavelet coefficients statistics
        """
        try:
            import pywt
        except ImportError:
            # If pywt not available, return frequency features instead
            return self.extract_frequency_features(data)
        
        n_samples, n_channels, n_timepoints = data.shape
        features_list = []
        
        for i in range(n_samples):
            sample_features = []
            for j in range(n_channels):
                channel_data = data[i, j, :]
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(channel_data, wavelet, level=level)
                
                # Extract statistics from each decomposition level
                for coeff in coeffs:
                    mean_coeff = np.mean(np.abs(coeff))
                    std_coeff = np.std(coeff)
                    energy = np.sum(coeff**2)
                    sample_features.extend([mean_coeff, std_coeff, energy])
            
            features_list.append(sample_features)
        
        return np.array(features_list)
    
    def extract_all_features(self, data: np.ndarray,
                            include_time: bool = True,
                            include_freq: bool = True,
                            include_wavelet: bool = False) -> np.ndarray:
        """
        Extract all requested features and concatenate them.
        
        Args:
            data: EEG data of shape (n_samples, n_channels, n_timepoints)
            include_time: Whether to include time-domain features
            include_freq: Whether to include frequency-domain features
            include_wavelet: Whether to include wavelet features
            
        Returns:
            Combined feature array
        """
        features = []
        
        if include_time:
            time_features = self.extract_time_features(data)
            features.append(time_features)
        
        if include_freq:
            freq_features = self.extract_frequency_features(data)
            features.append(freq_features)
        
        if include_wavelet:
            wavelet_features = self.extract_wavelet_features(data)
            features.append(wavelet_features)
        
        if not features:
            raise ValueError("At least one feature type must be selected")
        
        return np.hstack(features)
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features to [0, 1] range for better SNN input encoding.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Normalized features
        """
        # Min-max normalization
        min_vals = np.min(features, axis=0, keepdims=True)
        max_vals = np.max(features, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (features - min_vals) / range_vals
        
        return normalized
