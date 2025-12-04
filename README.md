# EEG Finger Motion Detection Using Spiking Neural Networks

A research project that uses EEG signals and Spiking Neural Networks (SNNs) to decode individual finger movements, enabling full control of prosthetic arms without standard ANN classification models.

## 🎯 Project Overview

This project implements a complete pipeline for:
- **EEG Signal Processing**: Advanced preprocessing and feature extraction from EEG data
- **Spiking Neural Networks**: Bio-inspired neural networks using Leaky Integrate-and-Fire (LIF) neurons
- **Finger Motion Classification**: Real-time classification of 5 individual finger movements (Thumb, Index, Middle, Ring, Pinky)
- **Prosthetic Control**: Interface for controlling prosthetic arms based on decoded intentions

### Key Features

- ✅ **No Standard ANNs**: Uses only Spiking Neural Networks for classification
- ✅ **Real-time Processing**: Supports real-time EEG stream processing and prediction
- ✅ **Comprehensive Preprocessing**: Bandpass filtering, notch filtering, feature extraction
- ✅ **Multiple Encoding Methods**: Rate coding, latency coding, and temporal coding for spike generation
- ✅ **Prosthetic Interface**: Mock prosthetic controller for demonstration

## 🏗️ Architecture

```
EEG Signal → Preprocessing → Feature Extraction → Spike Encoding → SNN → Finger Classification → Prosthetic Control
```

### Components

1. **Preprocessing Module** (`src/preprocessing/`)
   - `eeg_loader.py`: Load EEG data from various formats
   - `signal_processor.py`: Signal filtering and normalization
   - `feature_extractor.py`: Time-domain and frequency-domain features

2. **SNN Module** (`src/snn/`)
   - `spike_encoder.py`: Convert features to spike trains
   - `lif_neuron.py`: Leaky Integrate-and-Fire neuron model
   - `snn_model.py`: Multi-layer SNN architecture

3. **Classifier Module** (`src/classifier/`)
   - `finger_classifier.py`: High-level interface for training and prediction

4. **Prosthetic Control Module** (`src/prosthetic_control/`)
   - `prosthetic_controller.py`: Prosthetic arm control interface
   - `realtime_predictor.py`: Real-time prediction handler

## 📦 Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, SciPy, scikit-learn
- MNE (for EEG processing)
- snntorch, norse (for SNN)

### Setup

```bash
# Clone the repository
git clone https://github.com/Sudeep-T-Pillai/EEG-Finger-motion-using-ssn.git
cd EEG-Finger-motion-using-ssn

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Training Example

Train the SNN model on synthetic EEG data:

```bash
python examples/train_example.py
```

This will:
- Generate synthetic EEG data for 5 finger classes
- Train the SNN classifier
- Evaluate on validation set
- Save the trained model

### 2. Model Evaluation

Evaluate the model with detailed metrics:

```bash
python examples/evaluate_model.py
```

Generates:
- Confusion matrix
- Per-class accuracy
- Training curves
- Classification report

### 3. Prosthetic Control Demo

Run a real-time prosthetic control demonstration:

```bash
python examples/demo_prosthetic.py
```

Demonstrates:
- Real-time EEG processing
- Finger motion prediction
- Prosthetic hand control simulation

## 💻 Usage

### Basic Usage

```python
from src.preprocessing import EEGDataLoader
from src.classifier import FingerMotionClassifier

# Load or generate data
loader = EEGDataLoader(sampling_rate=250)
data, labels = loader.generate_synthetic_data(
    n_samples=200,
    n_channels=64,
    n_timepoints=500,
    n_classes=5
)

# Initialize classifier
classifier = FingerMotionClassifier(
    sampling_rate=250,
    n_channels=64,
    time_steps=100,
    hidden_sizes=[256, 128],
    learning_rate=0.001
)

# Train
classifier.train(
    train_data=data,
    train_labels=labels,
    n_epochs=50,
    batch_size=32
)

# Predict
predictions = classifier.predict_with_names(test_data)
print(predictions)  # ['Thumb', 'Index', 'Middle', ...]
```

### Prosthetic Control

```python
from src.prosthetic_control import ProstheticController, RealtimePredictor

# Initialize controller
controller = ProstheticController(
    smoothing_window=3,
    confidence_threshold=0.6,
    movement_speed=10.0
)

# Initialize real-time predictor
predictor = RealtimePredictor(
    classifier=classifier,
    n_channels=64,
    window_size=500,
    sampling_rate=250
)

# Set callback for predictions
def on_prediction(finger_name, confidence):
    controller.update_prediction(finger_name, confidence)
    positions = controller.update_positions()
    print(f"Detected: {finger_name}, Positions: {positions}")

predictor.set_prediction_callback(on_prediction)

# Process real-time stream
predictor.simulate_realtime_stream(eeg_data)
```

## 🧪 Testing with Real Data

To use your own EEG data:

```python
import mne
from src.preprocessing import EEGDataLoader

# Load EEG data using MNE
raw = mne.io.read_raw_edf('your_data.edf', preload=True)
events = mne.find_events(raw)

# Define event IDs for each finger
event_id = {
    'Thumb': 1,
    'Index': 2,
    'Middle': 3,
    'Ring': 4,
    'Pinky': 5
}

# Load and preprocess
loader = EEGDataLoader(sampling_rate=250)
data, labels = loader.load_from_mne_raw(raw, events, event_id)

# Continue with training...
```

## 📊 Performance

With synthetic data (200 samples, 50 epochs):
- Overall accuracy: ~75-85%
- Per-class accuracy: Varies by finger (70-90%)
- Training time: ~5-10 minutes (CPU)
- Inference time: ~10-20ms per sample

Performance on real EEG data will vary based on:
- Signal quality
- Number of electrodes
- Participant training
- Hardware setup

## 🔬 Research Background

### Spiking Neural Networks

SNNs are the third generation of neural networks that more closely mimic biological neurons. Key advantages:
- **Temporal Information**: Process time-coded information naturally
- **Energy Efficiency**: Event-driven computation
- **Biological Plausibility**: Similar to brain function

### EEG and Motor Imagery

- EEG captures electrical activity from the brain's motor cortex
- Different fingers activate different cortical regions
- Frequency bands (alpha, beta, gamma) correlate with motor planning and execution
- SNNs can capture temporal dynamics in motor-related EEG

## 🛠️ Development

### Project Structure

```
EEG-Finger-motion-using-ssn/
├── src/
│   ├── preprocessing/       # EEG preprocessing
│   ├── snn/                # Spiking neural network
│   ├── classifier/         # Finger motion classifier
│   └── prosthetic_control/ # Prosthetic interface
├── examples/               # Example scripts
├── tests/                  # Unit tests
├── config/                 # Configuration files
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{eeg_finger_snn,
  title={EEG Finger Motion Detection Using Spiking Neural Networks},
  author={Sudeep T Pillai},
  year={2024},
  url={https://github.com/Sudeep-T-Pillai/EEG-Finger-motion-using-ssn}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

## 🙏 Acknowledgments

- MNE Python for EEG processing tools
- snnTorch for SNN implementations
- PyTorch community

---

**Note**: This is a research project. For clinical or commercial prosthetic applications, additional validation, safety measures, and regulatory approval would be required.