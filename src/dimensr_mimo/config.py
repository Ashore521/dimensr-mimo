from dataclasses import dataclass
import math
from ._vendor.expression import ARITY


@dataclass
class SearchConfig:
    max_features: int = 256
    generations: int = 4
    proposals_per_generation: int = 160
    max_terms: int = 5
    max_nodes: int = 13
    max_depth: int = 6
    operators: tuple = ("add", "sub", "mul", "div", "square", "sin", "cos")
    complexity_weight: float = 1e-4
    validation_fraction: float = .2
    polish_max_nfev: int = 30
    random_state: int = 0
    interval_max_boxes: int = 32
    head_ridge: float = 0.0

    def validate(self):
        for name in ("max_features", "proposals_per_generation", "max_terms", "max_nodes", "max_depth", "interval_max_boxes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("generations", "polish_max_nfev", "random_state"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if not math.isfinite(self.complexity_weight) or self.complexity_weight < 0:
            raise ValueError("complexity_weight must be finite and nonnegative.")
        if not isinstance(self.head_ridge, (int, float)) or isinstance(self.head_ridge, bool) or not math.isfinite(self.head_ridge) or self.head_ridge < 0:
            raise ValueError("head_ridge must be finite and nonnegative.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if not self.operators or len(set(self.operators)) != len(self.operators) or set(self.operators) - set(ARITY):
            raise ValueError("operators must contain distinct supported operators.")
