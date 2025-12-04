"""
Leaky Integrate-and-Fire (LIF) Neuron
Implementation of LIF neuron dynamics for SNN
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpikeFunction(torch.autograd.Function):
    """
    Surrogate gradient for spike function.
    Forward: Heaviside step function
    Backward: Sigmoid derivative (surrogate)
    """
    
    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Use sigmoid derivative as surrogate gradient
        # Scale for better gradient flow
        scale = 10.0
        sig = torch.sigmoid(scale * (input - ctx.threshold))
        grad_input = grad_output * scale * sig * (1 - sig)
        return grad_input, None


spike_fn = SpikeFunction.apply


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron model.
    """
    
    def __init__(self, 
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 reset: float = 0.0,
                 refractory_period: int = 0):
        """
        Initialize LIF neuron.
        
        Args:
            threshold: Membrane potential threshold for spike generation
            decay: Membrane potential decay factor (0 < decay < 1)
            reset: Reset potential after spike
            refractory_period: Number of time steps in refractory period
        """
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.decay = decay
        self.reset = reset
        self.refractory_period = refractory_period
        
    def forward(self, input_current: torch.Tensor, 
                membrane_potential: torch.Tensor,
                refractory_counter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of LIF neuron for one time step.
        
        Args:
            input_current: Input current at this time step (batch_size, n_neurons)
            membrane_potential: Current membrane potential (batch_size, n_neurons)
            refractory_counter: Refractory counter (batch_size, n_neurons)
            
        Returns:
            Tuple of (spikes, new_membrane_potential, new_refractory_counter)
        """
        # Update membrane potential with leak
        membrane_potential = self.decay * membrane_potential
        
        # Add input current only to neurons not in refractory period
        not_refractory = (refractory_counter == 0).float()
        membrane_potential = membrane_potential + input_current * not_refractory
        
        # Generate spikes using surrogate gradient function
        spikes = spike_fn(membrane_potential, self.threshold)
        
        # Reset membrane potential for neurons that spiked
        membrane_potential = membrane_potential * (1 - spikes) + self.reset * spikes
        
        # Update refractory counter
        refractory_counter = torch.clamp(refractory_counter - 1, min=0)
        refractory_counter = refractory_counter + self.refractory_period * spikes
        
        return spikes, membrane_potential, refractory_counter
    
    def init_states(self, batch_size: int, n_neurons: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize neuron states.
        
        Args:
            batch_size: Batch size
            n_neurons: Number of neurons
            device: Device to create tensors on
            
        Returns:
            Tuple of (membrane_potential, refractory_counter)
        """
        membrane_potential = torch.zeros(batch_size, n_neurons, device=device)
        refractory_counter = torch.zeros(batch_size, n_neurons, device=device)
        return membrane_potential, refractory_counter


class LIFLayer(nn.Module):
    """
    Layer of LIF neurons with learnable weights.
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 threshold: float = 1.0,
                 decay: float = 0.9,
                 use_bias: bool = True):
        """
        Initialize LIF layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output neurons
            threshold: Spike threshold
            decay: Membrane potential decay
            use_bias: Whether to use bias term
        """
        super(LIFLayer, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        # Learnable weights
        self.fc = nn.Linear(input_size, output_size, bias=use_bias)
        
        # LIF neuron dynamics
        self.lif = LIFNeuron(threshold=threshold, decay=decay)
        
    def forward(self, spikes_in: torch.Tensor,
                membrane_potential: torch.Tensor,
                refractory_counter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for one time step.
        
        Args:
            spikes_in: Input spikes (batch_size, input_size)
            membrane_potential: Current membrane potential
            refractory_counter: Current refractory counter
            
        Returns:
            Tuple of (spikes_out, new_membrane_potential, new_refractory_counter)
        """
        # Apply linear transformation
        current = self.fc(spikes_in)
        
        # Update LIF neuron states
        spikes_out, membrane_potential, refractory_counter = self.lif(
            current, membrane_potential, refractory_counter
        )
        
        return spikes_out, membrane_potential, refractory_counter
    
    def init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize layer states.
        
        Args:
            batch_size: Batch size
            device: Device for tensors
            
        Returns:
            Tuple of (membrane_potential, refractory_counter)
        """
        return self.lif.init_states(batch_size, self.output_size, device)
