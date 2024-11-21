"""
CTMU State Implementation.

Implements state representation for CTMU reality, including:
1. Telic State - purposeful direction
2. Meta State - state about states
3. SCSPL State - reality's self-processing state
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

@dataclass
class TelicState:
    """
    Represents a state with purpose (telos) in CTMU.
    Combines potential (what could be) with actuality (what is).
    """
    potential: np.ndarray  # Vector of potentialities
    actuality: np.ndarray  # Vector of actualities
    coherence: float  # Measure of internal consistency
    utility: float  # Measure of telic effectiveness
    
    @classmethod
    def create_initial(cls) -> 'TelicState':
        """Create initial balanced state."""
        return cls(
            potential=np.ones(3),
            actuality=np.zeros(3),
            coherence=1.0,
            utility=0.0
        )
    
    def evolve(self, delta_metactime: float) -> 'TelicState':
        """
        Evolve state through metactime.
        Implements telic evolution where potential becomes actual.
        """
        # Calculate actualization factor
        actualization = delta_metactime * self.coherence
        
        # Move potential towards actuality
        new_potential = self.potential * np.exp(-actualization)
        new_actuality = self.actuality + (self.potential - self.actuality) * actualization
        
        # Update coherence and utility
        new_coherence = self.coherence * (1 + delta_metactime * self.utility)
        new_utility = self.utility + delta_metactime * self.coherence
        
        return TelicState(
            potential=new_potential,
            actuality=new_actuality,
            coherence=new_coherence,
            utility=new_utility
        )

@dataclass
class SCSPLState:
    """
    Represents the state of reality's self-processing language.
    Combines syntax (form) with semantics (meaning).
    """
    syntax_vector: np.ndarray  # Structural relationships
    semantic_vector: np.ndarray  # Meaningful content
    processing_state: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def create_initial(cls) -> 'SCSPLState':
        """Create initial SCSPL state."""
        return cls(
            syntax_vector=np.ones(3),
            semantic_vector=np.ones(3),
            processing_state={
                'conspansion_rate': 1.0,
                'telic_efficiency': 1.0,
                'quantum_coherence': 1.0
            }
        )
    
    def evolve(self, delta_metactime: float) -> 'SCSPLState':
        """Evolve SCSPL state through self-processing."""
        # Update vectors through self-modification
        new_syntax = self.syntax_vector * np.exp(delta_metactime * 
                    self.processing_state['conspansion_rate'])
        new_semantics = self.semantic_vector * np.exp(delta_metactime * 
                       self.processing_state['telic_efficiency'])
        
        # Update processing state
        new_processing = {
            'conspansion_rate': self.processing_state['conspansion_rate'] * 
                               (1 + delta_metactime),
            'telic_efficiency': self.processing_state['telic_efficiency'] * 
                              self.processing_state['quantum_coherence'],
            'quantum_coherence': self.processing_state['quantum_coherence'] * 
                               np.exp(-delta_metactime)
        }
        
        return SCSPLState(
            syntax_vector=new_syntax,
            semantic_vector=new_semantics,
            processing_state=new_processing
        )
    
@dataclass
class MetaState:
    """
    Represents meta-state in CTMU - state about states.
    Tracks history and evaluates utility of state transitions.
    """
    meta_time: float = 0.0
    telic_state: TelicState = field(default_factory=TelicState.create_initial)
    state_history: Dict[float, TelicState] = field(default_factory=dict)
    
    @classmethod
    def create_initial(cls) -> 'MetaState':
        """Create initial meta state."""
        initial = cls()
        initial.remember_state(0.0, initial.telic_state)
        return initial
        
    def remember_state(self, time: float, state: TelicState):
        """Store state in history at given time."""
        self.state_history[time] = state
        
    def get_state_at(self, time: float) -> Optional[TelicState]:
        """Retrieve state at given time if it exists."""
        return self.state_history.get(time)
        
    def evaluate_utility(self, state: TelicState) -> float:
        """
        Evaluate utility of a state transition.
        Considers:
        1. Coherence changes
        2. Alignment between potential and actuality
        3. Information gain through actualization
        """
        # Get previous state
        prev_state = self.state_history.get(self.meta_time - 1.0, self.telic_state)
        
        # Evaluate coherence change
        coherence_change = state.coherence - prev_state.coherence
        
        # Evaluate alignment between potential and actuality
        alignment = np.dot(state.potential, state.actuality)
        prev_alignment = np.dot(prev_state.potential, prev_state.actuality)
        alignment_change = alignment - prev_alignment
        
        # Calculate information gain through actualization
        info_gain = np.log2(np.linalg.norm(state.actuality) + 1) - \
                   np.log2(np.linalg.norm(prev_state.actuality) + 1)
                   
        # Combine factors with weights
        utility = (coherence_change + 1) * \
                 (alignment_change + 1) * \
                 (info_gain + 1)
                 
        return float(utility)
