from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyaugmecon")
except PackageNotFoundError:
    __version__ = "unknown"

from .config import (
    PyAugmeconConfig as PyAugmeconConfig,
)
from .results import PyAugmeconResult as PyAugmeconResult, Solution as Solution
from .solver.core import PyAugmecon as PyAugmecon

__all__ = [
    "PyAugmecon",
    "PyAugmeconConfig",
    "PyAugmeconResult",
    "Solution",
    "__version__",
]
