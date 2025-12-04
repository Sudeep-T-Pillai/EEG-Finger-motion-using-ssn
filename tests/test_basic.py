"""
Basic tests to verify core functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.preprocessing import EEGDataLoader, SignalProcessor, FeatureExtractor
        from src.snn import SpikeEncoder, LIFNeuron, SNNModel
        from src.classifier import FingerMotionClassifier
        from src.prosthetic_control import ProstheticController, RealtimePredictor
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation."""
    print("\nTesting data generation...")
    
    try:
        from src.preprocessing import EEGDataLoader
        
        loader = EEGDataLoader(sampling_rate=250)
        data, labels = loader.generate_synthetic_data(
            n_samples=10,
            n_channels=8,
            n_timepoints=100,
            n_classes=5
        )
        
        assert data.shape == (10, 8, 100), f"Wrong data shape: {data.shape}"
        assert labels.shape == (10,), f"Wrong labels shape: {labels.shape}"
        assert len(np.unique(labels)) == 5, f"Wrong number of classes: {len(np.unique(labels))}"
        
        print("✓ Data generation successful")
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False


def test_signal_processing():
    """Test signal processing."""
    print("\nTesting signal processing...")
    
    try:
        from src.preprocessing import SignalProcessor
        
        processor = SignalProcessor(sampling_rate=250)
        
        # Create test data
        data = np.random.randn(5, 8, 100)
        
        # Test bandpass filter
        filtered = processor.bandpass_filter(data)
        assert filtered.shape == data.shape, "Bandpass filter changed shape"
        
        # Test normalization
        normalized = processor.normalize(data, method='zscore')
        assert normalized.shape == data.shape, "Normalization changed shape"
        
        print("✓ Signal processing successful")
        return True
    except Exception as e:
        print(f"✗ Signal processing failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    
    try:
        from src.preprocessing import FeatureExtractor
        
        extractor = FeatureExtractor(sampling_rate=250)
        
        # Create test data
        data = np.random.randn(5, 8, 100)
        
        # Test time features
        time_features = extractor.extract_time_features(data)
        assert time_features.shape[0] == 5, "Wrong number of samples"
        assert time_features.ndim == 2, "Wrong feature dimensionality"
        
        # Test frequency features
        freq_features = extractor.extract_frequency_features(data)
        assert freq_features.shape[0] == 5, "Wrong number of samples"
        
        print("✓ Feature extraction successful")
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False


def test_spike_encoding():
    """Test spike encoding."""
    print("\nTesting spike encoding...")
    
    try:
        from src.snn import SpikeEncoder
        
        encoder = SpikeEncoder(encoding_method='rate', time_steps=50)
        
        # Create test features
        features = np.random.rand(5, 10)  # 5 samples, 10 features
        
        # Test encoding
        spikes = encoder.encode(features)
        assert spikes.shape == (5, 50, 10), f"Wrong spike shape: {spikes.shape}"
        assert torch.is_tensor(spikes), "Output should be a tensor"
        
        print("✓ Spike encoding successful")
        return True
    except Exception as e:
        print(f"✗ Spike encoding failed: {e}")
        return False


def test_lif_neuron():
    """Test LIF neuron."""
    print("\nTesting LIF neuron...")
    
    try:
        from src.snn import LIFNeuron
        
        lif = LIFNeuron(threshold=1.0, decay=0.9)
        
        # Test forward pass
        batch_size = 4
        n_neurons = 8
        input_current = torch.randn(batch_size, n_neurons)
        membrane_potential = torch.zeros(batch_size, n_neurons)
        refractory_counter = torch.zeros(batch_size, n_neurons)
        
        spikes, new_mem, new_refrac = lif(input_current, membrane_potential, refractory_counter)
        
        assert spikes.shape == (batch_size, n_neurons), "Wrong spike shape"
        assert new_mem.shape == (batch_size, n_neurons), "Wrong membrane shape"
        
        print("✓ LIF neuron successful")
        return True
    except Exception as e:
        print(f"✗ LIF neuron failed: {e}")
        return False


def test_snn_model():
    """Test SNN model."""
    print("\nTesting SNN model...")
    
    try:
        from src.snn import SNNModel
        
        model = SNNModel(
            input_size=10,
            hidden_sizes=[16, 8],
            output_size=5,
            time_steps=20
        )
        
        # Test forward pass
        spike_trains = torch.randn(2, 20, 10)  # batch=2, time=20, features=10
        output_spikes, spike_counts = model(spike_trains)
        
        assert output_spikes.shape == (2, 20, 5), f"Wrong output shape: {output_spikes.shape}"
        assert spike_counts.shape == (2, 5), f"Wrong spike counts shape: {spike_counts.shape}"
        
        print("✓ SNN model successful")
        return True
    except Exception as e:
        print(f"✗ SNN model failed: {e}")
        return False


def test_prosthetic_controller():
    """Test prosthetic controller."""
    print("\nTesting prosthetic controller...")
    
    try:
        from src.prosthetic_control import ProstheticController
        
        controller = ProstheticController()
        
        # Test prediction update
        controller.update_prediction('Thumb', 0.8)
        
        # Test position update
        positions = controller.update_positions()
        assert len(positions) == 5, "Wrong number of fingers"
        assert all(0 <= v <= 100 for v in positions.values()), "Invalid positions"
        
        # Test reset
        controller.reset_all()
        
        print("✓ Prosthetic controller successful")
        return True
    except Exception as e:
        print(f"✗ Prosthetic controller failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_data_generation,
        test_signal_processing,
        test_feature_extraction,
        test_spike_encoding,
        test_lif_neuron,
        test_snn_model,
        test_prosthetic_controller,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    exit(main())
