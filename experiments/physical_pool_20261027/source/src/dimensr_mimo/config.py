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
    reference_budget: str = 'token'
    use_sharing: bool = True
    use_constraints: bool = True
    degradation_relative: float = 0.10
    degradation_absolute: float = 0.02
    wall_seconds: float | None = None
    time_allocation: str = 'global'
    initialization_fraction: float = 0.15
    initialization: str = 'legacy'
    final_selection: str = 'weighted'
    acceptance: dict | None = None

    def validate(self):
        super().validate()
        for key in ('use_sharing', 'use_constraints'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key + ' must be boolean.')
        if type(self.max_references) is not int or self.max_references < 1:
            raise ValueError('max_references must be positive.')
        if self.reference_budget not in ('token','expanded'):
            raise ValueError('reference_budget must be token or expanded.')
        for k in ('degradation_relative', 'degradation_absolute'):
            if not math.isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError(k + ' must be finite and nonnegative.')
        if self.wall_seconds is not None and (not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0):
            raise ValueError('wall_seconds must be positive or None.')
        if self.time_allocation not in ('global','phased'):
            raise ValueError('time_allocation must be global or phased.')
        if self.time_allocation=='phased' and self.wall_seconds is None:
            raise ValueError('Phased time allocation requires wall_seconds.')
        if not 0 < self.initialization_fraction < 0.3:
            raise ValueError('initialization_fraction must lie in (0, 0.3).')
        if self.initialization not in ('legacy','dimensional','compound'):
            raise ValueError('initialization must be legacy, dimensional or compound.')
        if self.final_selection not in ('weighted','guarded','precision'):
            raise ValueError('final_selection must be weighted, guarded or precision.')
        if self.final_selection=='precision':
            from .acceptance import validate_limits
            validate_limits(self.acceptance)
            if not self.use_units or not self.use_constraints:
                raise ValueError('Precision acceptance requires units and per-output constraints.')
        elif self.acceptance is not None:
            raise ValueError('Acceptance limits require precision final selection.')
