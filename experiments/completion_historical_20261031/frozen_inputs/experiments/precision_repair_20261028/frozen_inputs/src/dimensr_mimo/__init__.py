"""Joint static multi-output symbolic regression."""
from .config import SearchConfig
from .regressor import DimenSRMIMORegressor, PACKAGE_VERSION as __version__
from .graph import System, Vertex
from .physical import AffineCoordinates, PhysicalModel
from .physical_selection import select_physical

__all__ = ['SearchConfig', 'DimenSRMIMORegressor', 'System', 'Vertex', 'AffineCoordinates', 'PhysicalModel', 'select_physical']
