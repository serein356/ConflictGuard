"""
conflictguard
=============
Core activation-steering implementation for multimodal GUI agents.
"""

from .adapters import get_adapter, ADAPTER_REGISTRY, ModelAdapter
from .steering import MalleableModel, SteeringVectorPT, LeashLayer

__all__ = [
    "get_adapter",
    "ADAPTER_REGISTRY",
    "ModelAdapter",
    "MalleableModel",
    "SteeringVectorPT",
    "LeashLayer",
]
