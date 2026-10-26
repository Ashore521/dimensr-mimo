"""Joint static multi-output symbolic regression."""
from .config import SearchConfig
from .regressor import DimenSRMIMORegressor
from .graph import System, Vertex

__version__ = '1.0.0'
__all__ = ['SearchConfig', 'DimenSRMIMORegressor', 'System', 'Vertex']
