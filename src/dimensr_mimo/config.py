"""MIMO controls extend DimenSR's bounded search configuration."""
from dataclasses import dataclass
import math
import os
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
    sharing_strategy: str = 'random'
    use_constraints: bool = True
    degradation_relative: float = 0.10
    degradation_absolute: float = 0.02
    wall_seconds: float | None = None
    time_allocation: str = 'global'
    initialization_fraction: float = 0.15
    initialization: str = 'legacy'
    precision_repair: bool = False
    compositional_completion: bool = False
    final_selection: str = 'weighted'
    acceptance: dict | None = None
    accuracy_search: bool = False
    search_objective: str = 'mean'
    parameter_diversity: bool = False
    budget_mode: str = 'legacy'
    stage_evaluations: dict | None = None
    residual_rounds: int = 64
    sharing_rounds: int = 2
    sharing_attempts: int = 512
    sharing_evaluations_per_round: int = 24
    stop_metric: str | None = None
    stop_target: float | tuple | None = None
    stop_after_n_epochs: int = 0
    parallel_mode: str = 'serial'
    n_cpus: int = 1
    reward_violation_weight: float = 1.0
    max_working_memory_mb: float = 256.0
    search_schedule: str = 'fixed'
    stagnation_settings: dict | None = None
    structure_limits: dict | None = None
    stage_windows: dict | None = None
    constant_fit_mode: str = 'joint'
    constant_prediction_limit: int | None = None
    imitation_scan_limit: int = 0
    compact_elites: bool = False
    fixed_training_reference: tuple | None = None
    constant_jacobian: str = 'dense'
    constant_local_parameters: int = 32
    stage_budgets: list | None = None
    residual_guidance: bool = False
    compact_seed_terms: int = 0
    parameter_bank_counts: tuple | None = None
    constant_focus_worst: bool = False
    portfolio_settings: dict | None = None
    enhanced_prefix: dict | None = None
    compression_r2_drop: float | tuple | None = None

    @classmethod
    def accuracy_portfolio(cls, *, max_evaluations, portfolio_settings=None, **options):
        """Opt-in persistent producer portfolio with one inclusive fit budget."""
        values = dict(budget_mode='evaluations', accuracy_search=True,
                      initialization='compound', sharing_strategy='closure',
                      final_selection='pareto', acceptance=None, wall_seconds=None,
                      time_allocation='global', search_schedule='portfolio',
                      stage_evaluations=None, use_constraints=True,
                      use_units=True, use_rademacher=True)
        values.update(options)
        values.update(max_evaluations=max_evaluations, portfolio_settings=portfolio_settings)
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def accuracy_pareto_progress(cls, *, max_evaluations, stage_budgets, **options):
        """Explicit-budget, worst-output mode; no implicit increase in total work."""
        values = dict(search_objective='worst', residual_guidance=True,
                      constant_fit_mode='separable', constant_focus_worst=True,
                      compact_seed_terms=4)
        values.update(options)
        return cls.accuracy_pareto_stagnation(max_evaluations=max_evaluations,
                                             stage_budgets=stage_budgets, **values)

    @classmethod
    def accuracy_pareto_stagnation(cls, **options):
        """Opt-in historical stagnation checks with one shared evaluation budget."""
        from .stagnation import settings
        values = dict(budget_mode='evaluations', accuracy_search=True, initialization='compound',
            sharing_strategy='closure', final_selection='pareto', acceptance=None,
            wall_seconds=None, time_allocation='global', max_evaluations=1_000_000,
            stage_evaluations=None, search_schedule='stagnation')
        values.update(options)
        values['stagnation_settings'] = settings(values.get('stagnation_settings'))
        config = cls(**values)
        config.validate()
        return config

    @classmethod
    def accuracy_pareto(cls, **options):
        """Finite evaluation search with training targets and validation Pareto selection."""
        values = dict(budget_mode='evaluations', accuracy_search=True, initialization='compound',
                      sharing_strategy='closure', final_selection='pareto', acceptance=None,
                      wall_seconds=None, time_allocation='global')
        values.update(options)
        total = values.get('max_evaluations', cls.max_evaluations)
        if type(total) is not int or total < 1:
            raise ValueError('max_evaluations must be a positive integer.')
        if values.get('stage_evaluations') is None:
            counts = [max(1, total * 120 // 3000), total * 2100 // 3000,
                      total * 600 // 3000, total * 132 // 3000]
            values['stage_evaluations'] = dict(zip(
                ('initialization', 'exploration', 'nsga2', 'residual', 'sharing'),
                counts + [total-sum(counts)]))
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if self.compression_r2_drop is not None:
            values = self.compression_r2_drop
            values = values if isinstance(values, (list, tuple)) else (values,)
            if (self.final_selection != 'pareto' or not self.interval_checks or not values or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                        not math.isfinite(v) or v < 0 for v in values)):
                raise ValueError('compression_r2_drop requires Pareto mode, interval checks and finite nonnegative tolerances.')
        super().validate()
        for name in ('residual_guidance', 'constant_focus_worst'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(name + ' must be boolean.')
        if type(self.compact_seed_terms) is not int or self.compact_seed_terms < 0:
            raise ValueError('compact_seed_terms must be nonnegative.')
        if self.parameter_bank_counts is not None:
            if (not isinstance(self.parameter_bank_counts, (list, tuple)) or
                    any(type(v) is not int or v < 0 for v in self.parameter_bank_counts) or
                    not 0 < sum(self.parameter_bank_counts) <= self.max_references or not self.use_sharing):
                raise ValueError('Parameter bank counts require sharing and available reference slots.')
        if self.constant_focus_worst and self.search_objective != 'worst':
            raise ValueError('Worst-output fitting focus requires worst-output scoring.')
        if (self.residual_guidance or self.compact_seed_terms or self.stage_budgets is not None) and self.search_schedule != 'stagnation':
            raise ValueError('Progress controls require stagnation scheduling.')
        if self.constant_fit_mode not in ('joint', 'separable', 'local'):
            raise ValueError('Unknown constant fit mode.')
        if type(self.constant_local_parameters) is not int or self.constant_local_parameters < 1:
            raise ValueError('constant_local_parameters must be positive.')
        if self.constant_jacobian not in ('dense', 'output_sparse'):
            raise ValueError('Unknown constant Jacobian mode.')
        if self.constant_jacobian != 'dense' and self.constant_fit_mode != 'joint':
            raise ValueError('Output sparsity currently supports joint fitting only.')
        if self.fixed_training_reference is not None:
            if (not isinstance(self.fixed_training_reference, (tuple, list)) or not self.fixed_training_reference or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                        for v in self.fixed_training_reference) or self.final_selection != 'pareto' or not self.use_constraints):
                raise ValueError('Invalid fixed training reference.')
        if self.constant_prediction_limit is not None and (
                type(self.constant_prediction_limit) is not int or self.constant_prediction_limit < 1):
            raise ValueError('constant_prediction_limit must be positive or None.')
        if type(self.imitation_scan_limit) is not int or self.imitation_scan_limit < 0:
            raise ValueError('imitation_scan_limit must be nonnegative.')
        if type(self.compact_elites) is not bool:
            raise ValueError('compact_elites must be boolean.')
        for name, keys in (('structure_limits', {'parameters', 'nonlinear_parameters', 'nodes', 'depth'}),
                           ('stage_windows', {'exploration', 'nsga2', 'residual', 'sharing'})):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, dict) or not value or set(value) - keys or
                    any(type(v) is not int or v < 1 for v in value.values())):
                raise ValueError('Invalid ' + name)
        if self.stage_windows is not None and (self.search_schedule != 'stagnation' or self.structure_limits is None):
            raise ValueError('Stage windows require stagnation search with fixed structure limits.')
        if (self.structure_limits is not None or self.imitation_scan_limit or self.compact_elites) and self.final_selection != 'pareto':
            raise ValueError('Capacity and feedback controls require Pareto mode.')
        if self.search_schedule not in ('fixed', 'stagnation', 'portfolio'):
            raise ValueError('Unknown search schedule.')
        if self.search_schedule == 'portfolio':
            from .portfolio_scheduler import settings
            self.portfolio_settings = settings(self.portfolio_settings)
            if (self.budget_mode != 'evaluations' or self.final_selection != 'pareto' or
                    not self.use_constraints or not self.use_units or not self.use_rademacher or
                    self.parallel_mode != 'serial' or self.n_cpus != 1 or
                    self.stage_evaluations is not None or self.stagnation_settings is not None or
                    self.stop_after_n_epochs != 0):
                raise ValueError('Portfolio requires serial Pareto evaluations and a shared budget.')
        elif self.portfolio_settings is not None:
            raise ValueError('Portfolio settings require portfolio scheduling.')
        if self.enhanced_prefix is not None:
            if self.search_schedule != 'portfolio':
                raise ValueError('Enhanced prefix requires portfolio scheduling.')
            from .enhanced_portfolio import prefix_config
            prefix_config(self)
        if self.search_schedule == 'stagnation':
            from .stagnation import settings
            controls = settings(self.stagnation_settings)
            if controls['max_restarts'] > 2:
                raise ValueError('At most two internal restarts are supported.')
            if (self.final_selection != 'pareto' or self.budget_mode != 'evaluations' or
                    self.stage_evaluations is not None or not self.use_constraints or
                    ((self.stop_metric is not None or self.stop_target is not None) and
                     (self.search_objective != 'worst' or self.stop_after_n_epochs != 0))):
                raise ValueError('Stagnation requires Pareto evaluation mode and fixed protection; immediate targets require worst scoring.')
            if self.stage_budgets is not None:
                phases = {'exploration', 'nsga2', 'residual', 'sharing', 'restart'}
                if (not isinstance(self.stage_budgets, list) or not self.stage_budgets or
                        len(self.stage_budgets) > controls['max_restarts'] + 1 or
                        any(not isinstance(row, dict) or set(row) != phases or
                            any(type(v) is not int or v < 0 for v in row.values()) for row in self.stage_budgets)):
                    raise ValueError('Each cycle needs five nonnegative first-evaluation stage caps.')
                if (self.stage_budgets[-1]['restart'] != 0 or
                        any(row['restart'] < 1 for row in self.stage_budgets[:-1]) or
                        controls['initialization_evaluations'] + sum(sum(row.values()) for row in self.stage_budgets) > self.max_evaluations):
                    raise ValueError('Stage plan exceeds the global budget or has an invalid restart allowance.')
        elif self.stagnation_settings is not None:
            raise ValueError('Stagnation settings require the opt-in stagnation schedule.')
        if type(self.stop_after_n_epochs) is not int or self.stop_after_n_epochs < 0:
            raise ValueError('stop_after_n_epochs must be a nonnegative integer.')
        from .stopping import target_vector
        target_vector(self.stop_metric, self.stop_target)
        if self.parallel_mode not in ('serial', 'process'):
            raise ValueError('parallel_mode must be serial or process.')
        if type(self.n_cpus) is not int or self.n_cpus < 1:
            raise ValueError('n_cpus must be a positive integer.')
        if self.parallel_mode == 'process' and self.n_cpus > (os.cpu_count() or 1):
            raise ValueError('n_cpus exceeds the available logical CPU count.')
        for name in ('reward_violation_weight', 'max_working_memory_mb'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(name + ' must be finite and nonnegative.')
        if self.max_working_memory_mb == 0:
            raise ValueError('max_working_memory_mb must be positive.')
        if type(self.sharing_evaluations_per_round) is not int or self.sharing_evaluations_per_round < 1:
            raise ValueError('sharing_evaluations_per_round must be a positive integer.')
        if self.final_selection == 'pareto':
            if (self.budget_mode != 'evaluations' or self.search_objective not in ('mean', 'worst') or
                    self.parameter_diversity or not self.use_rademacher or not self.use_units):
                raise ValueError('Pareto mode requires evaluation budgets, mean scoring, units and Rademacher complexity.')
        elif self.stop_metric is not None or self.parallel_mode != 'serial' or self.reward_violation_weight != 1:
            raise ValueError('Training targets, parallel evaluation and reward weighting require Pareto mode.')
        if self.budget_mode not in ('legacy', 'evaluations'):
            raise ValueError('Unknown budget mode.')
        if self.budget_mode == 'evaluations':
            stages = ('initialization', 'exploration', 'nsga2', 'residual', 'sharing')
            if not self.accuracy_search or self.wall_seconds is not None or self.time_allocation != 'global':
                raise ValueError('Evaluation budgets require accuracy_search, wall_seconds=None and global allocation.')
            if self.precision_repair or self.compositional_completion:
                raise ValueError('The first evaluation-budget mode excludes legacy repair/completion.')
            if self.search_schedule in ('stagnation', 'portfolio'):
                pass
            elif not isinstance(self.stage_evaluations, dict) or set(self.stage_evaluations) != set(stages):
                raise ValueError('Explicit evaluation allowances for all five stages are required.')
            elif any(type(v) is not int or v < 0 for v in self.stage_evaluations.values()) or self.stage_evaluations['initialization'] < 1:
                raise ValueError('Stage allowances must be nonnegative integers with positive initialization.')
            elif sum(self.stage_evaluations.values()) > self.max_evaluations:
                raise ValueError('Stage allowances exceed the global evaluation budget.')
            for name in ('residual_rounds', 'sharing_rounds', 'sharing_attempts'):
                if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                    raise ValueError(name + ' must be a nonnegative integer.')
        elif self.stage_evaluations is not None:
            raise ValueError('Stage evaluation allowances require evaluation-budget mode.')
        if self.search_objective not in ('mean', 'worst', 'tolerance_mean_max'):
            raise ValueError('Unknown experimental search objective.')
        if type(self.parameter_diversity) is not bool:
            raise ValueError('parameter_diversity must be boolean.')
        if self.search_objective != 'mean' or self.parameter_diversity:
            if not self.accuracy_search or self.precision_repair or self.compositional_completion:
                raise ValueError('Alignment experiments require accuracy_search without legacy repair/completion.')
            from .acceptance import validate_limits
            if self.search_objective == 'tolerance_mean_max' or self.parameter_diversity:
                validate_limits(self.acceptance)
            if self.search_objective == 'tolerance_mean_max' and any(t <= 0 for t in self.acceptance['nrmse']):
                raise ValueError('Tolerance-normalized scoring requires positive accuracy limits.')
        for key in ('use_sharing', 'use_constraints', 'precision_repair', 'compositional_completion', 'accuracy_search'):
            if type(getattr(self, key)) is not bool:
                raise ValueError(key + ' must be boolean.')
        if type(self.max_references) is not int or self.max_references < 1:
            raise ValueError('max_references must be positive.')
        if self.reference_budget not in ('token','expanded'):
            raise ValueError('reference_budget must be token or expanded.')
        if self.sharing_strategy not in ('random','matched','closure'):
            raise ValueError('sharing_strategy must be random, matched or closure.')
        if self.sharing_strategy=='closure' and (self.final_selection not in ('precision', 'pareto') or not self.use_sharing):
            raise ValueError('Closure sharing requires precision/Pareto selection and use_sharing.')
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
        if self.initialization not in ('legacy','dimensional','compound','implicit'):
            raise ValueError('initialization must be legacy, dimensional, compound or implicit.')
        if self.precision_repair and (self.initialization!='compound' or self.sharing_strategy!='closure' or self.final_selection!='precision'):
            raise ValueError('Precision repair requires compound initialization, closure sharing and precision selection.')
        if self.precision_repair and self.compositional_completion:
            raise ValueError('Precision repair and compositional completion are mutually exclusive.')
        if self.compositional_completion and (self.initialization!='compound' or self.sharing_strategy!='closure' or self.final_selection!='precision'):
            raise ValueError('Compositional completion requires compound initialization, closure sharing and precision selection.')
        if self.accuracy_search and ((self.wall_seconds is None and self.budget_mode != 'evaluations') or self.initialization!='compound' or
                self.sharing_strategy!='closure' or self.final_selection not in ('precision', 'pareto')):
            raise ValueError('Accuracy search requires a time budget, compound initialization, closure sharing and precision selection.')
        if self.final_selection not in ('weighted','guarded','precision','pareto'):
            raise ValueError('final_selection must be weighted, guarded, precision or pareto.')
        if self.final_selection=='precision':
            from .acceptance import validate_limits
            validate_limits(self.acceptance)
            if not self.use_units or not self.use_constraints:
                raise ValueError('Precision acceptance requires units and per-output constraints.')
        elif self.acceptance is not None:
            raise ValueError('Acceptance limits require precision final selection.')
