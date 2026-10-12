"""grspin: relaxation of classical spins with the general-relativistic spin-spin interaction.

See ``docs/METHODS.md`` for the model and the definitions of the observables.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - depends on how the package is installed
    __version__ = version("gr-spin-frustration")
except PackageNotFoundError:  # pragma: no cover - running from a plain source checkout
    __version__ = "1.0.0"

from .config import COUPLINGS, RNN_UNIT_DENSITY, AnnealConfig, SimulationConfig

__all__ = [
    "COUPLINGS",
    "RNN_UNIT_DENSITY",
    "AnnealConfig",
    "SimulationConfig",
    "__version__",
]
