"""
Spiking Neural Network Model
Complete SNN architecture for EEG-based finger motion classification
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from .lif_neuron import LIFLayer


class SNNModel(nn.Module):
    """
    Multi-layer Spiking Neural Network for finger motion classification.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int = 5,
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 time_steps: int = 100):
        """
        Initialize SNN model.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output classes (5 fingers by default)
            threshold: Spike threshold for neurons
            decay: Membrane potential decay factor
            time_steps: Number of time steps for simulation
        """
        super(SNNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.time_steps = time_steps
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(LIFLayer(prev_size, hidden_size, threshold, decay))
            prev_size = hidden_size
        
        # Output layer
        self.layers.append(LIFLayer(prev_size, output_size, threshold, decay))
        
    def forward(self, spike_trains: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            spike_trains: Input spike trains of shape (batch_size, time_steps, input_size)
            
        Returns:
            Tuple of (output_spikes, spike_counts)
            - output_spikes: Spikes over time (batch_size, time_steps, output_size)
            - spike_counts: Total spike count per output neuron (batch_size, output_size)
        """
        batch_size = spike_trains.shape[0]
        device = spike_trains.device
        
        # Initialize states for all layers
        layer_states = []
        for layer in self.layers:
            mem, refrac = layer.init_states(batch_size, device)
            layer_states.append((mem, refrac))
        
        # Storage for output spikes over time
        output_spikes_over_time = []
        
        # Process each time step
        for t in range(self.time_steps):
            # Get input spikes at this time step
            spikes = spike_trains[:, t, :]
            
            # Propagate through layers
            for i, layer in enumerate(self.layers):
                mem, refrac = layer_states[i]
                spikes, mem, refrac = layer(spikes, mem, refrac)
                layer_states[i] = (mem, refrac)
            
            # Store output layer spikes
            output_spikes_over_time.append(spikes.unsqueeze(1))
        
        # Concatenate output spikes over time
        output_spikes = torch.cat(output_spikes_over_time, dim=1)
        
        # Count total spikes for each output neuron
        spike_counts = torch.sum(output_spikes, dim=1)
        
        return output_spikes, spike_counts
    
    def predict(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Make predictions from spike trains.
        
        Args:
            spike_trains: Input spike trains
            
        Returns:
            Predicted class labels (batch_size,)
        """
        with torch.no_grad():
            _, spike_counts = self.forward(spike_trains)
            predictions = torch.argmax(spike_counts, dim=1)
        
        return predictions
    
    def get_spike_rates(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Get output spike rates for each class.
        
        Args:
            spike_trains: Input spike trains
            
        Returns:
            Spike rates (batch_size, output_size)
        """
        with torch.no_grad():
            _, spike_counts = self.forward(spike_trains)
            spike_rates = spike_counts / self.time_steps
        
        return spike_rates


class SNNClassifier:
    """
    Wrapper class for training and evaluating SNN models.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_sizes: List[int] = [256, 128],
                 output_size: int = 5,
                 learning_rate: float = 0.001,
                 time_steps: int = 100,
                 device: Optional[str] = None):
        """
        Initialize SNN classifier.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of classes
            learning_rate: Learning rate for optimizer
            time_steps: Number of simulation time steps
            device: Device to use ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = SNNModel(input_size, hidden_sizes, output_size, time_steps=time_steps)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.time_steps = time_steps
        
    def train_epoch(self, spike_trains: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Train for one epoch.
        
        Args:
            spike_trains: Input spike trains (batch_size, time_steps, input_size)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        
        spike_trains = spike_trains.to(self.device)
        labels = labels.to(self.device)
        
        # Detach spike trains as they are inputs without gradients
        spike_trains = spike_trains.detach()
        
        self.optimizer.zero_grad()
        
        # Forward pass
        _, spike_counts = self.model(spike_trains)
        
        # Compute loss using spike counts as logits
        loss = self.criterion(spike_counts, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, spike_trains: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        """
        Evaluate the model.
        
        Args:
            spike_trains: Input spike trains
            labels: Ground truth labels
            
        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()
        
        spike_trains = spike_trains.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            _, spike_counts = self.model(spike_trains)
            loss = self.criterion(spike_counts, labels)
            
            predictions = torch.argmax(spike_counts, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        
        return accuracy, loss.item()
    
    def predict(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            spike_trains: Input spike trains
            
        Returns:
            Predicted labels
        """
        self.model.eval()
        spike_trains = spike_trains.to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict(spike_trains)
        
        return predictions.cpu()
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
