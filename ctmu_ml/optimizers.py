"""
Optimizers that incorporate CTMU's telic recursion
"""
import numpy as np
from typing import List
from ctmu_core.state import TelicState
from .layers import TelosLayer

class TelicOptimizer:
    """Optimizer that uses telic feedback for updates"""
    def __init__(self, layers: List[TelosLayer], learning_rate: float = 0.01):
        self.layers = layers
        self.lr = learning_rate
        
    def step(self, loss: float):
        """Update parameters using telic feedback and loss"""
        # Reference TelicState evolution:
        """ctmu_core/state.py
        startLine: 34
        endLine: 55
        """
        
        for layer in self.layers:
            # Get telic state
            state = layer.telic_state
            
            # Scale updates by telic coherence and loss
            effective_lr = self.lr * (1 + state.coherence) * np.exp(-loss)
            
            # Calculate gradient scaling based on loss and utility
            grad_scale = loss * state.utility
            
            # Update weights with telic influence and loss
            layer.weights -= effective_lr * layer.weights * grad_scale
            layer.bias -= effective_lr * layer.bias * grad_scale
            
            # Update layer's telic state based on loss
            layer.telic_state = layer.telic_state.evolve(loss)