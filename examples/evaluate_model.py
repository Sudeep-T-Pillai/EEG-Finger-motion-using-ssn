"""
Model Evaluation Example
Demonstrates how to evaluate the trained SNN model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.preprocessing import EEGDataLoader
from src.classifier import FingerMotionClassifier


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - Finger Motion Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path='results/training_history.png'):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and len(history['val_loss']) > 0:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
        ax2.plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def main():
    """Main evaluation example."""
    print("=" * 60)
    print("EEG Finger Motion Classification - Model Evaluation")
    print("=" * 60)
    
    # Configuration
    n_samples = 200
    n_channels = 64
    n_timepoints = 500
    sampling_rate = 250
    
    # Generate data
    print("\nGenerating synthetic EEG data...")
    data_loader = EEGDataLoader(sampling_rate=sampling_rate)
    data, labels = data_loader.generate_synthetic_data(
        n_samples=n_samples,
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        n_classes=5
    )
    
    # Split data
    split_idx = int(0.7 * n_samples)
    val_split_idx = int(0.85 * n_samples)
    
    train_data = data[:split_idx]
    train_labels = labels[:split_idx]
    val_data = data[split_idx:val_split_idx]
    val_labels = labels[split_idx:val_split_idx]
    test_data = data[val_split_idx:]
    test_labels = labels[val_split_idx:]
    
    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")
    
    # Train model
    print("\n" + "-" * 60)
    print("Training model...")
    classifier = FingerMotionClassifier(
        sampling_rate=sampling_rate,
        n_channels=n_channels,
        time_steps=100,
        hidden_sizes=[256, 128],
        learning_rate=0.001
    )
    
    history = classifier.train(
        train_data=train_data,
        train_labels=train_labels,
        val_data=val_data,
        val_labels=val_labels,
        n_epochs=50,
        batch_size=32,
        verbose=True
    )
    
    # Plot training history
    print("\n" + "-" * 60)
    print("Generating training plots...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Evaluating on Test Set:")
    print("-" * 60)
    
    results = classifier.evaluate(test_data, test_labels)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    print(f"Loss: {results['loss']:.4f}")
    
    print("\nPer-class Accuracy:")
    for finger, acc in results['per_class_accuracy'].items():
        print(f"  {finger}: {acc:.4f}")
    
    # Generate confusion matrix
    print("\n" + "-" * 60)
    print("Generating confusion matrix...")
    
    class_names = [classifier.FINGER_LABELS[i] for i in range(5)]
    plot_confusion_matrix(results['true_labels'], results['predictions'], class_names)
    
    # Detailed classification report
    print("\n" + "-" * 60)
    print("Detailed Classification Report:")
    print("-" * 60)
    report = classification_report(
        results['true_labels'],
        results['predictions'],
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save report to file
    os.makedirs('results', exist_ok=True)
    with open('results/classification_report.txt', 'w') as f:
        f.write("EEG Finger Motion Classification - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Loss: {results['loss']:.4f}\n\n")
        f.write("Per-class Accuracy:\n")
        for finger, acc in results['per_class_accuracy'].items():
            f.write(f"  {finger}: {acc:.4f}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("Detailed Classification Report:\n")
        f.write("-" * 60 + "\n")
        f.write(report)
    
    print("\nEvaluation report saved to results/classification_report.txt")
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
