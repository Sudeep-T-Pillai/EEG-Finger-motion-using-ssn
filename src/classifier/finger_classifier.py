"""
Finger Motion Classifier
High-level interface for classifying individual finger movements from EEG
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from ..preprocessing import EEGDataLoader, SignalProcessor, FeatureExtractor
from ..snn import SpikeEncoder, SNNModel
from ..snn.snn_model import SNNClassifier


class FingerMotionClassifier:
    """
    Complete pipeline for EEG-based finger motion classification using SNN.
    """
    
    # Finger labels mapping
    FINGER_LABELS = {
        0: 'Thumb',
        1: 'Index',
        2: 'Middle',
        3: 'Ring',
        4: 'Pinky'
    }
    
    def __init__(self,
                 sampling_rate: int = 250,
                 n_channels: int = 64,
                 time_steps: int = 100,
                 hidden_sizes: List[int] = [256, 128],
                 learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize the finger motion classifier.
        
        Args:
            sampling_rate: EEG sampling rate in Hz
            n_channels: Number of EEG channels
            time_steps: Number of SNN time steps
            hidden_sizes: Hidden layer sizes for SNN
            learning_rate: Learning rate for training
            device: Device to use ('cuda' or 'cpu')
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.time_steps = time_steps
        
        # Initialize preprocessing components
        self.data_loader = EEGDataLoader(sampling_rate)
        self.signal_processor = SignalProcessor(sampling_rate)
        self.feature_extractor = FeatureExtractor(sampling_rate)
        self.spike_encoder = SpikeEncoder(encoding_method='rate', time_steps=time_steps)
        
        # Initialize SNN classifier (will be set up after feature extraction)
        self.snn_classifier = None
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.device = device
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
    def preprocess_data(self, data: np.ndarray, labels: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full preprocessing pipeline from raw EEG to spike trains.
        
        Args:
            data: Raw EEG data (n_samples, n_channels, n_timepoints)
            labels: Ground truth labels (n_samples,)
            
        Returns:
            Tuple of (spike_trains, labels_tensor)
        """
        # Apply signal processing
        processed_data = self.signal_processor.preprocess_pipeline(
            data,
            apply_bandpass=True,
            apply_notch=True,
            apply_normalize=True
        )
        
        # Extract features
        features = self.feature_extractor.extract_all_features(
            processed_data,
            include_time=True,
            include_freq=True,
            include_wavelet=False
        )
        
        # Normalize features for spike encoding
        features = self.feature_extractor.normalize_features(features)
        
        # Encode to spike trains
        spike_trains = self.spike_encoder.encode(features)
        
        # Convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return spike_trains, labels_tensor
    
    def train(self,
              train_data: np.ndarray,
              train_labels: np.ndarray,
              val_data: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None,
              n_epochs: int = 50,
              batch_size: int = 32,
              verbose: bool = True) -> Dict:
        """
        Train the SNN classifier.
        
        Args:
            train_data: Training EEG data (n_samples, n_channels, n_timepoints)
            train_labels: Training labels (n_samples,)
            val_data: Validation EEG data (optional)
            val_labels: Validation labels (optional)
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Preprocess training data
        if verbose:
            print("Preprocessing training data...")
        train_spikes, train_labels_tensor = self.preprocess_data(train_data, train_labels)
        
        # Initialize SNN classifier if not already done
        if self.snn_classifier is None:
            input_size = train_spikes.shape[2]  # Feature dimension
            self.snn_classifier = SNNClassifier(
                input_size=input_size,
                hidden_sizes=self.hidden_sizes,
                output_size=5,  # 5 fingers
                learning_rate=self.learning_rate,
                time_steps=self.time_steps,
                device=self.device
            )
            if verbose:
                print(f"Initialized SNN with input size: {input_size}")
        
        # Preprocess validation data if provided
        if val_data is not None and val_labels is not None:
            if verbose:
                print("Preprocessing validation data...")
            val_spikes, val_labels_tensor = self.preprocess_data(val_data, val_labels)
        
        # Training loop
        n_samples = train_spikes.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Shuffle training data
            indices = torch.randperm(n_samples)
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_spikes = train_spikes[batch_indices]
                batch_labels = train_labels_tensor[batch_indices]
                
                loss = self.snn_classifier.train_epoch(batch_spikes, batch_labels)
                epoch_losses.append(loss)
            
            # Calculate epoch metrics
            avg_loss = np.mean(epoch_losses)
            train_acc, _ = self.snn_classifier.evaluate(train_spikes, train_labels_tensor)
            
            self.train_history['loss'].append(avg_loss)
            self.train_history['accuracy'].append(train_acc)
            
            # Validation
            if val_data is not None:
                val_acc, val_loss = self.snn_classifier.evaluate(val_spikes, val_labels_tensor)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_accuracy'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, "
                          f"Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}, "
                          f"Acc: {train_acc:.4f}")
        
        return self.train_history
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict finger movements from EEG data.
        
        Args:
            data: EEG data (n_samples, n_channels, n_timepoints)
            
        Returns:
            Predicted finger labels (n_samples,)
        """
        if self.snn_classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess data
        dummy_labels = np.zeros(data.shape[0])
        spike_trains, _ = self.preprocess_data(data, dummy_labels)
        
        # Make predictions
        predictions = self.snn_classifier.predict(spike_trains)
        
        return predictions.numpy()
    
    def predict_with_names(self, data: np.ndarray) -> List[str]:
        """
        Predict finger movements and return finger names.
        
        Args:
            data: EEG data (n_samples, n_channels, n_timepoints)
            
        Returns:
            List of finger names
        """
        predictions = self.predict(data)
        return [self.FINGER_LABELS[pred] for pred in predictions]
    
    def evaluate(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_data: Test EEG data
            test_labels: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.snn_classifier is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess test data
        test_spikes, test_labels_tensor = self.preprocess_data(test_data, test_labels)
        
        # Evaluate
        accuracy, loss = self.snn_classifier.evaluate(test_spikes, test_labels_tensor)
        
        # Get predictions for confusion matrix
        predictions = self.snn_classifier.predict(test_spikes).numpy()
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_id in range(5):
            mask = test_labels == class_id
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == test_labels[mask])
                per_class_accuracy[self.FINGER_LABELS[class_id]] = class_acc
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'per_class_accuracy': per_class_accuracy,
            'predictions': predictions,
            'true_labels': test_labels
        }
    
    def save(self, path: str):
        """Save the trained model."""
        if self.snn_classifier is None:
            raise ValueError("No model to save. Train the model first.")
        self.snn_classifier.save_model(path)
    
    def load(self, path: str):
        """Load a trained model."""
        if self.snn_classifier is None:
            # Need to initialize with proper size - this requires knowing the input size
            raise ValueError("Cannot load model without knowing input size. Train first or provide architecture.")
        self.snn_classifier.load_model(path)
