"""
Training Example
Demonstrates how to train the SNN model for finger motion classification
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.preprocessing import EEGDataLoader
from src.classifier import FingerMotionClassifier


def main():
    """Main training example."""
    print("=" * 60)
    print("EEG Finger Motion Classification Training Example")
    print("=" * 60)
    
    # Configuration
    n_samples = 200
    n_channels = 64
    n_timepoints = 500
    sampling_rate = 250
    n_epochs = 50
    
    print(f"\nConfiguration:")
    print(f"  Samples: {n_samples}")
    print(f"  Channels: {n_channels}")
    print(f"  Timepoints: {n_timepoints}")
    print(f"  Sampling rate: {sampling_rate} Hz")
    print(f"  Epochs: {n_epochs}")
    
    # Generate synthetic data
    print("\n" + "-" * 60)
    print("Generating synthetic EEG data...")
    data_loader = EEGDataLoader(sampling_rate=sampling_rate)
    data, labels = data_loader.generate_synthetic_data(
        n_samples=n_samples,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=5
    )
    
    print(f"Generated data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Split into train and validation sets
    split_idx = int(0.8 * n_samples)
    train_data, val_data = data[:split_idx], data[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"\nTrain set: {train_data.shape[0]} samples")
    print(f"Validation set: {val_data.shape[0]} samples")
    
    # Initialize classifier
    print("\n" + "-" * 60)
    print("Initializing SNN Classifier...")
    classifier = FingerMotionClassifier(
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        time_steps=100,
        hidden_sizes=[256, 128],
        learning_rate=0.001
    )
    
    # Train the model
    print("\n" + "-" * 60)
    print("Training the model...")
    print("-" * 60)
    
    history = classifier.train(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        n_epochs=n_epochs,
        batch_size=32,
        verbose=True
    )
    
    # Evaluate on validation set
    print("\n" + "-" * 60)
    print("Final Evaluation on Validation Set:")
    print("-" * 60)
    
    results = classifier.evaluate(val_data, val_labels)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    
    print("\nPer-class Accuracy:")
    for finger, acc in results['per_class_accuracy'].items():
        print(f"  {finger}: {acc:.4f}")
    
    # Make some predictions
    print("\n" + "-" * 60)
    print("Sample Predictions:")
    print("-" * 60)
    
    sample_indices = [0, 5, 10, 15, 20]
    sample_data = val_data[sample_indices]
    sample_labels = val_labels[sample_indices]
    
    predictions = classifier.predict_with_names(sample_data)
    
    print("\nPrediction | True Label")
    print("-" * 30)
    for i, (pred, true_label) in enumerate(zip(predictions, sample_labels)):
        true_name = classifier.FINGER_LABELS[true_label]
        status = "✓" if pred == true_name else "✗"
        print(f"{status} {pred:8s} | {true_name:8s}")
    
    # Save the model
    model_path = 'checkpoints/finger_classifier.pth'
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n" + "-" * 60)
    print(f"Saving model to {model_path}...")
    classifier.save(model_path)
    print("Model saved successfully!")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
