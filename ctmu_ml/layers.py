"""
Neural network layers implementing CTMU concepts
"""
import numpy as np
from typing import Optional
from ctmu_core.tellers import Teller
from ctmu_core.state import TelicState

class TelosLayer(Teller):
    """Layer that implements purposeful processing"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(f"telos_layer_{in_features}_{out_features}")
        self.weights = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)
        self.telic_state = TelicState.create_initial()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with telic influence"""
        # Reference Reality process method:
        """ctmu_core/domains/reality.py
        startLine: 79
        endLine: 97
        """
        
        output = np.dot(x, self.weights) + self.bias
        # Apply telic influence
        output *= (1 + self.telic_state.coherence)
        return np.tanh(output)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass with telic feedback"""
        # Update telic state based on gradient
        self.telic_state = self.telic_state.evolve(np.mean(np.abs(grad)))
        return grad @ self.weights.T 