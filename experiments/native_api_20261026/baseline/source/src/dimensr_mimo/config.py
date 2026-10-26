"""MIMO controls extend DimenSR's bounded search configuration."""
from dataclasses import dataclass
import math
from ._single.config import SearchConfig as SingleConfig


@dataclass
class SearchConfig(SingleConfig):
    population_size: int = 32
    epochs: int = 4
    gp_generations: int = 2
    selection_generations: int = 4
    max_evaluations: int = 2000
    max_references: int = 12
    use_sharing: bool = True
    use_constraints: bool = True
    degradation_relative: float = 0.10
    degradation_absolute: float = 0.02
    wall_seconds: float | None = None
    initialization_fraction: float = 0.15

    def validate(self):
        super().validate()
        for key in ('use_sharing', 'use_constraints'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key + ' must be boolean.')
        if type(self.max_references) is not int or self.max_references < 1:
            raise ValueError('max_references must be positive.')
        for k in ('degradation_relative', 'degradation_absolute'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError(k + ' must be finite and nonnegative.')
        if self.wall_seconds is not None and (not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0):
            raise ValueError('wall_seconds must be positive or None.')
        if not 0 < self.initialization_fraction < 0.3:
            raise ValueError('initialization_fraction must lie in (0, 0.3).')
