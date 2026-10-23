"""Shared-expression, dimension-aware multi-output symbolic regression."""
from .config import SearchConfig
from .regressor import MIMOSymbolicRegressor

__version__ = "0.1.0"
__all__ = ["MIMOSymbolicRegressor", "SearchConfig"]
