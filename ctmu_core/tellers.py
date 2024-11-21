"""
Implementation of Tellers (syntactic operators) in CTMU.
Tellers are SCSPL operators that process and transform reality through
self-configuration and self-processing.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Set, Dict, List
import numpy as np
from ctmu_core.state import TelicState, MetaState

class Teller(ABC):
    """
    Base class for all tellers (SCSPL operators) in CTMU.
    Tellers are fundamental syntactic operators that process reality through
    self-configuration and self-processing language (SCSPL).
    """
    
    def __init__(self, identity: str, dimension: int = 3):
        self.identity = identity
        self.dimension = dimension
        self.meta_state = MetaState.create_initial()
        self.connections: Dict[str, 'Teller'] = {}
        self.scspl_state = {
            'syntax': np.zeros(dimension),  # Syntactic structure
            'semantics': np.zeros(dimension),  # Semantic content
            'telesis': 1.0,  # Telic capacity
            'conspansion': 0.0  # Conspansive evolution rate
        }
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data according to teller's SCSPL logic.
        Implements basic SCSPL operations:
        1. Syntactic processing - structural transformation
        2. Semantic interpretation - meaning assignment
        3. Telic recursion - purposeful evolution
        
        Args:
            input_data: Can be either:
                - float: Represents delta meta-time for evolution
                - TelicState: State to be processed
                - np.ndarray: Raw data to be interpreted
                
        Returns:
            Processed state or transformed data
        """
        # Handle meta-time evolution
        if isinstance(input_data, float):
            delta_t = input_data
            
            # Update SCSPL state through telic recursion
            self.scspl_state['syntax'] *= (1 + delta_t * self.scspl_state['telesis'])
            self.scspl_state['semantics'] += delta_t * self.scspl_state['syntax']
            self.scspl_state['conspansion'] += delta_t * self.scspl_state['telesis']
            
            # Generate new telic state
            new_state = TelicState(
                potential=self.scspl_state['syntax'],
                actuality=self.scspl_state['semantics'],
                coherence=self.scspl_state['telesis'],
                utility=self.scspl_state['conspansion']
            )
            
            return new_state
            
        # Handle telic state processing
        elif isinstance(input_data, TelicState):
            # Apply SCSPL transformations
            processed_potential = input_data.potential * self.scspl_state['syntax']
            processed_actuality = input_data.actuality * self.scspl_state['semantics']
            
            return TelicState(
                potential=processed_potential,
                actuality=processed_actuality,
                coherence=input_data.coherence * self.scspl_state['telesis'],
                utility=input_data.utility * (1 + self.scspl_state['conspansion'])
            )
            
        # Handle raw data processing
        elif isinstance(input_data, np.ndarray):
            # Apply syntactic transformation
            processed = input_data * self.scspl_state['syntax']
            # Apply semantic interpretation
            processed *= self.scspl_state['semantics']
            return processed
            
        return input_data
    
    def connect(self, other: 'Teller', relation_type: str) -> None:
        """
        Establish a typed connection with another teller.
        Implements CTMU's principle of syndiffeonesis (unity through difference).
        """
        # Verify dimensional compatibility
        if self.dimension != other.dimension:
            raise ValueError("Tellers must have compatible dimensions")
            
        self.connections[relation_type] = other
        other.connections[f"inverse_{relation_type}"] = self
        
        # Update SCSPL states to reflect new connection
        self._update_scspl_state(other, relation_type)
        
    def _update_scspl_state(self, other: 'Teller', relation_type: str) -> None:
        """
        Update SCSPL state based on new connection.
        Implements telic feedback through state updates.
        """
        # Syntactic update - structural coupling
        coupling_strength = 0.1 * np.exp(-len(self.connections))
        self.scspl_state['syntax'] += (other.scspl_state['syntax'] - 
                                      self.scspl_state['syntax']) * coupling_strength
        
        # Semantic update - meaning transfer
        semantic_vector = self._compute_semantic_vector(relation_type, other)
        self.scspl_state['semantics'] = (self.scspl_state['semantics'] + 
                                        semantic_vector) / 2
        
        # Update telesis and conspansion
        self.scspl_state['telesis'] *= (1 + len(self.connections) * 0.1)
        self.scspl_state['conspansion'] += 0.1 * self.scspl_state['telesis']
        
    def _compute_semantic_vector(self, relation_type: str, other: 'Teller') -> np.ndarray:
        """Compute semantic vector for connection type."""
        # Hash-based encoding of relation semantics
        base_vector = np.array([
            hash(relation_type) % 100,
            hash(other.identity) % 100,
            hash(self.identity) % 100
        ]) / 100.0
        
        # Extend to full dimension if needed
        if self.dimension > 3:
            extended = np.zeros(self.dimension)
            extended[:3] = base_vector
            return extended
        return base_vector[:self.dimension]

class SecondaryTeller(Teller):
    """
    Secondary tellers represent conscious entities with higher-order
    SCSPL processing capabilities and memory.
    """
    
    def __init__(self, identity: str, dimension: int = 3):
        super().__init__(identity, dimension)
        self.consciousness_level = 1.0
        self.memory: List[TelicState] = []
        
    def _update_states(self, new_state: TelicState, delta_t: float) -> None:
        """
        Update internal states after evolution.
        Implements conscious state tracking and memory updates.
        
        Args:
            new_state: New telic state after processing
            delta_t: Time increment in meta-time
        """
        # Update meta state
        self.meta_state.telic_state = new_state
        self.meta_state.meta_time += delta_t
        self.meta_state.remember_state(self.meta_state.meta_time, new_state)
        
        # Update SCSPL state with consciousness factor
        consciousness_factor = np.tanh(self.consciousness_level)
        self.scspl_state['syntax'] *= (1 + delta_t * new_state.coherence * consciousness_factor)
        self.scspl_state['semantics'] += delta_t * new_state.potential * consciousness_factor
        self.scspl_state['telesis'] *= (1 + delta_t * new_state.utility)
        
        # Update memory
        if len(self.memory) > 100:  # Maintain finite memory
            self.memory.pop(0)
        self.memory.append(new_state)

class TertiaryTeller(Teller):
    """
    Tertiary tellers represent fundamental particles exhibiting
    basic SCSPL operations at the quantum level.
    """
    
    def __init__(self, identity: str, dimension: int = 3):
        super().__init__(identity, dimension)
        self.quantum_state = np.ones(dimension) / np.sqrt(dimension)
        
    def process(self, input_data: Any) -> Any:
        """
        Process input through quantum SCSPL operations.
        Implements quantum-level telic recursion.
        """
        if isinstance(input_data, float):
            evolved_state = self._evolve_quantum_state(input_data)
            new_telic = self._quantum_to_telic_state(evolved_state)
            self._update_states(new_telic, input_data)
            return new_telic
        return input_data
        
    def _evolve_quantum_state(self, delta_t: float) -> np.ndarray:
        """Evolve quantum state through SCSPL operations."""
        # Apply quantum evolution with conspansive factor
        phase = delta_t * (1 + self.scspl_state['conspansion'])
        evolved_state = self.quantum_state * np.exp(1j * phase)
        return evolved_state / np.linalg.norm(evolved_state)

    def _update_states(self, new_state: TelicState, delta_t: float) -> None:
        """Update internal states after evolution."""
        self.meta_state.telic_state = new_state
        self.meta_state.meta_time += delta_t
        self.meta_state.remember_state(self.meta_state.meta_time, new_state)
        
        # Update SCSPL state based on new state
        self.scspl_state['syntax'] *= (1 + delta_t * new_state.coherence)
        self.scspl_state['semantics'] += delta_t * new_state.potential

    def _quantum_to_telic_state(self, quantum_state: np.ndarray) -> TelicState:
        """Convert quantum state to telic state."""
        potential = np.abs(quantum_state)
        actuality = np.real(quantum_state)
        
        return TelicState(
            potential=potential,
            actuality=actuality,
            coherence=np.abs(np.vdot(quantum_state, quantum_state)),
            utility=self.scspl_state['telesis']
        )
