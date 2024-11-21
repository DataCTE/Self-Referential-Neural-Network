"""
CTMU Core - A Python implementation of core concepts from the 
Cognitive-Theoretic Model of the Universe (CTMU) by Christopher Michael Langan
"""

from .tellers import Teller, SecondaryTeller, TertiaryTeller
from .telesis import Telesis
from .manifold import ConspansiveManifold
from .domains.reality import Reality
from .metaformal import MetaformalSystem

__all__ = ["Teller", "SecondaryTeller", "TertiaryTeller", "Telesis", "ConspansiveManifold", "Reality", "MetaformalSystem"]
