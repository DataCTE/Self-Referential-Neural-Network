"""
Implementation of Telesis (self-configuring causation) in CTMU.
Telesis represents the self-configuring causation aspect where reality
processes itself through telic recursion.
"""
from typing import Any, List, Optional, Tuple, Dict
import numpy as np
from ctmu_core.tellers import Teller
from ctmu_core.state import TelicState, MetaState
from ctmu_core.metaformal import MetaformalSystem

class Telesis:
    """
    Represents the self-configuring causation aspect of reality.
    Telesis binds tellers and enables coherent causation through:
    1. Telic recursion (purposeful feedback)
    2. Coherent state combination
    3. Utility evaluation
    """
    
    def __init__(self):
        self._bound_tellers: List[Teller] = []
        self._meta_state = MetaState.create_initial()
        self._metaformal = MetaformalSystem()
        self._telic_history: Dict[float, float] = {}
        
    def bind_teller(self, teller: Teller) -> None:
        """
        Bind a teller to this telesis instance.
        Tellers are syntactic operators that process reality.
        """
        self._bound_tellers.append(teller)
        
    def process_telic_recursion(self, delta_metactime: float) -> None:
        """
        Process telic recursion through metaformal system.
        
        Args:
            delta_metactime: Time increment in meta-time dimension
        """
        coherence_sum = 0.0
        utility_sum = 0.0
        
        for teller in self._bound_tellers:
            syntor = teller.to_syntor()
            self._metaformal.add_syntor(teller.identity, syntor)
            
            teller_state = teller.process(delta_metactime)
            if isinstance(teller_state, TelicState):
                coherence_sum += teller_state.coherence
                utility_sum += teller_state.utility
        
        self._metaformal.telic_field *= np.exp(delta_metactime * coherence_sum)
        self._metaformal.telic_field /= np.linalg.norm(self._metaformal.telic_field)
                
        self._metaformal.process_meta_relation(
            "telic_binding",
            [t.identity for t in self._bound_tellers],
        )
        
        self._meta_state.meta_time += delta_metactime
        self._meta_state.telic_state = TelicState(
            potential=self._metaformal.telic_field,
            actuality=self._meta_state.telic_state.actuality,
            coherence=coherence_sum / len(self._bound_tellers),
            utility=utility_sum / len(self._bound_tellers)
        )
    
    def propagate_causation(self, delta_meta_time: float) -> Tuple[TelicState, float]:
        """
        Propagate causal effects through bound tellers.
        Implements telic recursion through the network of tellers.
        
        Args:
            delta_meta_time: Time step in meta-time dimension
            
        Returns:
            Tuple of (new telic state, total utility gained)
        """
        # Start with current telic state
        current_state = self._meta_state.telic_state
        total_utility = current_state.utility
        
        # First pass: Collect all new states
        new_states = []
        for teller in self._bound_tellers:
            new_state = teller.process(delta_meta_time)
            if isinstance(new_state, TelicState):
                new_states.append(new_state)
        
        # Second pass: Coherently combine all states
        if new_states:
            # Start with first new state
            combined_state = new_states[0]
            # Combine with remaining states
            for state in new_states[1:]:
                combined_state = self._combine_states(combined_state, state)
                utility = self._meta_state.evaluate_utility(state)
                total_utility += utility
                self._telic_history[self._meta_state.meta_time] = utility
            
            # Update current state with combined state
            current_state = self._combine_states(current_state, combined_state)
        
        # Update meta-state
        self._meta_state.telic_state = current_state
        self._meta_state.meta_time += delta_meta_time
        self._meta_state.remember_state(self._meta_state.meta_time, current_state)
        
        return current_state, total_utility
    
    def _combine_states(self, state1: TelicState, state2: TelicState) -> TelicState:
        """
        Combine two telic states coherently.
        Implements coherent aspect of telesis through:
        1. Weighted combination of potentials/actualities
        2. Coherence evaluation based on similarity
        3. Conservation of utility
        """
        # Normalize vectors before combining
        s1_norm = np.linalg.norm(state1.potential)
        s2_norm = np.linalg.norm(state2.potential)
        if s1_norm > 0 and s2_norm > 0:
            w1 = s1_norm / (s1_norm + s2_norm)
            w2 = s2_norm / (s1_norm + s2_norm)
        else:
            w1 = w2 = 0.5
            
        # Weighted combination
        combined_potential = w1 * state1.potential + w2 * state2.potential
        combined_actuality = w1 * state1.actuality + w2 * state2.actuality
        
        # Evaluate coherence through similarity
        potential_similarity = np.dot(state1.potential, state2.potential) / \
                             (np.linalg.norm(state1.potential) * np.linalg.norm(state2.potential) + 1e-8)
        actuality_similarity = np.dot(state1.actuality, state2.actuality) / \
                             (np.linalg.norm(state1.actuality) * np.linalg.norm(state2.actuality) + 1e-8)
        coherence_factor = (potential_similarity + actuality_similarity) / 2
        
        # Create combined state
        return TelicState(
            potential=combined_potential,
            actuality=combined_actuality,
            utility=w1 * state1.utility + w2 * state2.utility,
            coherence=min(state1.coherence, state2.coherence) * (coherence_factor + 1) / 2
        )
    
    @property
    def state(self) -> MetaState:
        """Get current meta-state."""
        return self._meta_state
        
    @property
    def telic_history(self) -> Dict[float, float]:
        """Get history of telic utility values."""
        return self._telic_history.copy()
