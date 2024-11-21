"""
Terminal Domain Implementation (LO)
Represents the observable, physical universe with actualized states and events.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from ctmu_core.state import TelicState, SCSPLState

@dataclass
class TerminalState:
    """
    Represents a state in the terminal domain (LO).
    Contains actualized events and observable quantities.
    """
    actuality: np.ndarray  # Actualized state vector
    observables: Dict[str, float]  # Observable quantities
    physical_laws: Dict[str, callable]  # Active physical laws
    
    @classmethod
    def create_initial(cls) -> 'TerminalState':
        """Create initial terminal state."""
        return cls(
            actuality=np.zeros(3),
            observables={
                'energy': 1.0,
                'entropy': 0.0,
                'information': 0.0
            },
            physical_laws={}
        )

class TerminalDomain:
    """
    Implementation of the Terminal Domain (LO).
    Handles actualization of events and physical law enforcement.
    """
    
    def __init__(self):
        self.state = TerminalState.create_initial()
        self.history: List[TerminalState] = []
        
    def actualize_event(self, potential_state: np.ndarray) -> np.ndarray:
        """Convert potential state to actuality through measurement."""
        # Apply physical constraints
        constrained_state = self._apply_physical_laws(potential_state)
        
        # Update actuality through measurement
        self.state.actuality = constrained_state
        
        # Update observables
        self._update_observables()
        
        return constrained_state
    
    def _apply_physical_laws(self, state: np.ndarray) -> np.ndarray:
        """Apply physical laws to constrain state evolution."""
        constrained_state = state.copy()
        
        for law in self.state.physical_laws.values():
            constrained_state = law(constrained_state)
            
        return constrained_state
    
    def _update_observables(self):
        """Update observable quantities based on current state."""
        # Update energy
        self.state.observables['energy'] = np.sum(self.state.actuality ** 2)
        
        # Update entropy
        prob_dist = np.abs(self.state.actuality) ** 2
        prob_dist /= np.sum(prob_dist)
        self.state.observables['entropy'] = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
        
        # Update information
        self.state.observables['information'] = np.log2(len(self.state.actuality))