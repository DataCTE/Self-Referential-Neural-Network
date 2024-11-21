"""
Implementation of Syntors (active signs) in CTMU.
Syntors are fundamental processors that combine syntactic and semantic operations.
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class SyntacticIdentification:
    """Represents identification between syntactic elements"""
    source: str
    target: str
    strength: float
    telic_direction: np.ndarray

    def compute_coherence(self) -> float:
        """Compute coherence of identification"""
        return np.tanh(self.strength)

@dataclass 
class Syntor:
    """
    Active sign that processes both syntactic and semantic content.
    Implements CTMU's concept of signs as active processors.
    """
    input_type: str
    output_type: str
    internal_state: np.ndarray
    telic_vector: np.ndarray
    absorption_field: np.ndarray  # For inner expansion

    def process(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process input through internal state transformation.
        Returns processed data and coherence measure.
        """
        # Inner expansion through absorption
        absorbed = np.outer(input_data, self.absorption_field)
        
        # Process through internal state
        processed = absorbed @ self.internal_state
        
        # Update internal state through telic feedback
        telic_alignment = np.dot(processed, self.telic_vector)
        self.internal_state = np.tanh(self.internal_state + telic_alignment * processed)
        
        coherence = np.mean(np.abs(telic_alignment))
        return processed, coherence