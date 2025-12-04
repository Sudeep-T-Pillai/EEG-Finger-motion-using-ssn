"""
Prosthetic Control Demo
Demonstrates real-time prosthetic arm control using EEG predictions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from src.preprocessing import EEGDataLoader
from src.classifier import FingerMotionClassifier
from src.prosthetic_control import ProstheticController, RealtimePredictor


def print_hand_visual(positions: dict):
    """Print a simple text-based visualization of hand positions."""
    print("\n" + "=" * 50)
    print("Prosthetic Hand Status:")
    print("=" * 50)
    
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    for finger in fingers:
        pos = positions[finger]
        bar_length = int(pos / 5)  # Scale to 20 chars max
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{finger:8s} [{bar}] {pos:5.1f}%")
    
    print("=" * 50)


def main():
    """Main demo for prosthetic control."""
    print("=" * 60)
    print("EEG-Controlled Prosthetic Arm Demo")
    print("=" * 60)
    
    # Configuration
    n_samples = 50
    n_channels = 64
    n_timepoints = 500
    sampling_rate = 250
    
    print("\nInitializing system...")
    
    # Generate test data
    print("Generating synthetic EEG data...")
    data_loader = EEGDataLoader(sampling_rate=sampling_rate)
    data, labels = data_loader.generate_synthetic_data(
        n_samples=n_samples,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=5
    )
    
    # Quick training (reduced epochs for demo)
    print("Training SNN classifier (quick demo training)...")
    classifier = FingerMotionClassifier(
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        time_steps=50,  # Reduced for faster demo
        hidden_sizes=[128, 64],  # Smaller network for demo
        learning_rate=0.01
    )
    
    classifier.train(
        train_data=data,
        train_labels=labels,
        n_epochs=20,
        batch_size=16,
        verbose=False
    )
    
    print("Training complete!")
    
    # Initialize prosthetic controller
    print("\nInitializing prosthetic controller...")
    controller = ProstheticController(
        smoothing_window=3,
        confidence_threshold=0.5,
        movement_speed=50.0  # Faster for demo
    )
    
    # Initialize real-time predictor
    predictor = RealtimePredictor(
        classifier=classifier,
        n_channels=n_channels,
        window_size=n_timepoints,
        sampling_rate=sampling_rate
    )
    
    # Set up prediction callback
    def on_prediction(finger_name: str, confidence: float):
        """Callback for each prediction."""
        print(f"\n>>> Detected: {finger_name} (confidence: {confidence:.2f})")
        controller.update_prediction(finger_name, confidence)
    
    predictor.set_prediction_callback(on_prediction)
    
    # Run demo
    print("\n" + "-" * 60)
    print("Starting Real-Time Demo")
    print("-" * 60)
    print("\nProcessing EEG signals and controlling prosthetic hand...")
    
    controller.reset_all()
    print_hand_visual(controller.get_positions())
    
    # Simulate real-time stream with test data
    n_demo_samples = 10
    demo_data = data[:n_demo_samples]
    demo_labels = labels[:n_demo_samples]
    
    for i in range(n_demo_samples):
        # Process one sample
        sample = demo_data[i:i+1]
        true_label = demo_labels[i]
        true_finger = classifier.FINGER_LABELS[true_label]
        
        print(f"\n--- Sample {i+1}/{n_demo_samples} ---")
        print(f"True finger: {true_finger}")
        
        # Make prediction
        prediction = classifier.predict_with_names(sample)[0]
        confidence = 0.75 + np.random.rand() * 0.2  # Mock confidence
        
        # Update controller
        on_prediction(prediction, confidence)
        
        # Update positions (simulate movement)
        for _ in range(3):
            time.sleep(0.1)
            controller.update_positions()
        
        # Show hand status
        print_hand_visual(controller.get_positions())
        
        # Small delay between samples
        time.sleep(0.5)
    
    # Final status
    print("\n" + "-" * 60)
    print("Demo Complete!")
    print("-" * 60)
    
    status = controller.get_status()
    print("\nFinal Controller Status:")
    print(f"  Positions: {status['positions']}")
    print(f"  Targets: {status['targets']}")
    print(f"  Is Moving: {status['is_moving']}")
    
    # Get control signal
    control_signal = controller.get_control_signal()
    print(f"\nControl Signal (Thumb to Pinky): {control_signal}")
    
    print("\n" + "=" * 60)
    print("Prosthetic control demo completed!")
    print("=" * 60)
    print("\nNote: This is a demonstration using synthetic data.")
    print("In a real system, this would control actual prosthetic hardware.")


if __name__ == '__main__':
    main()
