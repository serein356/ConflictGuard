"""conflictguard.adapters – ModelAdapter"""

from .base import ModelAdapter
from .qwen3vl import Qwen3VLAdapter
from .uivenus import UIVenusAdapter
from .uitars import UITARSAdapter

#: Registry mapping adapter name → class.  Extend this dict when adding new adapters.
ADAPTER_REGISTRY: dict[str, type[ModelAdapter]] = {
    "qwen3vl": Qwen3VLAdapter,
    "uivenus": UIVenusAdapter,
    "uitars": UITARSAdapter,
}


def get_adapter(name: str) -> ModelAdapter:
    """
    Instantiate an adapter by name.

    Parameters
    ----------
    name
        One of the keys in ADAPTER_REGISTRY.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    cls = ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown adapter '{name}'. "
            f"Available: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    return cls()


__all__ = [
    "ModelAdapter",
    "Qwen3VLAdapter",
    "UIVenusAdapter",
    "UITARSAdapter",
    "ADAPTER_REGISTRY",
    "get_adapter",
]
