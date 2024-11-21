"""
Non-Terminal Domain Implementation (LS)
Represents the pre-physical domain of potentials and telic recursion.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ctmu_core.state import TelicState, SCSPLState

@dataclass
class NonTerminalState:
    """
    Represents a state in the non-terminal domain (LS).
    Contains potentials and telic properties.
    """
    potential: np.ndarray  # State potentials
    telic_field: np.ndarray  # Field of purposeful direction
    coherence: float  # Internal consistency
    
    @classmethod
    def create_initial(cls) -> 'NonTerminalState':
        """Create initial non-terminal state."""
        return cls(
            potential=np.ones(3),
            telic_field=np.ones(3),
            coherence=1.0
        )

class NonTerminalDomain:
    """
    Implementation of the Non-Terminal Domain (LS).
    Handles potential states and telic recursion.
    """
    
    def __init__(self):
        self.state = NonTerminalState.create_initial()
        self.history: List[NonTerminalState] = []
        
    def evolve_potential(self, delta_metactime: float) -> np.ndarray:
        """Evolve potential states through telic recursion."""
        # Apply telic influence
        telic_factor = np.sum(self.state.telic_field)
        evolved_potential = self.state.potential * np.exp(delta_metactime * telic_factor)
        
        # Update coherence
        self.state.coherence *= (1 + delta_metactime * telic_factor)
        
        # Update state
        self.state.potential = evolved_potential
        
        return evolved_potential
    
    def generate_possibilities(self) -> List[np.ndarray]:
        """Generate possible states from current potential."""
        # Create variations of current potential
        possibilities = []
        base_potential = self.state.potential
        
        # Generate variations through telic field influence
        for i in range(3):
            variation = base_potential * (1 + 0.1 * self.state.telic_field[i])
            possibilities.append(variation)
            
        return possibilities
    
    def update_telic_field(self, utility: float):
        """Update telic field based on utility feedback."""
        self.state.telic_field *= (1 + 0.1 * utility)
        self.state.telic_field /= np.linalg.norm(self.state.telic_field)