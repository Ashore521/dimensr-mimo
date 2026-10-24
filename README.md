# DimenSR-MIMO

Dimension-aware symbolic regression for multiple inputs and outputs, with a shared dictionary of nonlinear expressions. Each output has its own linear coefficients; the search selects basis functions using the residuals of all outputs together.

[中文说明](docs/README.zh-CN.md) · [Algorithm](docs/ALGORITHM.md) · [Validation](docs/VALIDATION.md) · [Original DimenSR](https://github.com/Ashore521/dimensr)

For example, the system

```text
y1 =  2 a b + 0.5 c
y2 = -1 a b + 3.0 c
```

can use the same two basis functions, `a*b` and `c`. Shared expression nodes are evaluated once per prediction batch. This is a coupled search rather than a collection of independently fitted single-output estimators.

## Install

Python 3.10 or newer is required. From a clone of this repository:

```bash
git clone https://github.com/Ashore521/dimensr-mimo.git
cd dimensr-mimo
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/shared_system.py
python -m pytest
```

Runtime dependencies are NumPy, SciPy, SymPy and scikit-learn. A GPU is not required. Installation from this repository is supported; this project has not been published to PyPI.

## Python API

```python
import numpy as np
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig

rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(160, 3))
Y = np.column_stack((2 * X[:, 0] * X[:, 1] + 0.5 * X[:, 2],
                     -X[:, 0] * X[:, 1] + 3 * X[:, 2]))

model = MIMOSymbolicRegressor(
    config=SearchConfig(random_state=42),
    feature_names=["a", "b", "c"],
    target_names=["y1", "y2"],
)
model.fit(X[:120], Y[:120], X_val=X[120:140], Y_val=Y[120:140],
          input_bounds=[[-2, 2]] * 3)
print(model.expressions_)                     # list of SymPy expressions
print(model.predict(X[140:]).shape)            # (20, 2)
print(model.audit_domain())                   # audit the declared input box
model.save("model.json")
restored = MIMOSymbolicRegressor.load("model.json")
np.testing.assert_allclose(restored.predict(X[140:]), model.predict(X[140:]))
```

`X` has shape `(samples, inputs)` and `Y` has shape `(samples, outputs)`. Predictions always have two dimensions. Supply finite numeric arrays and one distinct name per column. When validation arrays are omitted, a reproducible internal split uses `validation_fraction`. Keep a separate test set outside `fit`; validation chooses the final candidate and is not an unbiased test of it.

### Dimensions and coefficient units

Unit vectors use the seven SI base dimensions in order **L, M, T, I, Theta, N, J**. For example, length is `[1,0,0,0,0,0,0]` and time is `[0,0,1,0,0,0,0]`. Provide one vector per input and per output. Omitted units are dimensionless.

```python
length = [1, 0, 0, 0, 0, 0, 0]
time = [0, 0, 1, 0, 0, 0, 0]
speed = [1, 0, -1, 0, 0, 0, 0]
area = [2, 0, 0, 0, 0, 0, 0]
model = MIMOSymbolicRegressor(
    feature_names=["distance", "duration"],
    target_names=["speed", "area"],
    feature_units=[length, time],
    target_units=[speed, area],
)
# Fit with strictly positive duration bounds to admit distance/duration.
```

Coefficients are dimensionless by default, so an output can use only basis functions with matching dimensions. `coefficient_units` explicitly lists the permitted coefficient dimensions: a connection is legal when `target_units - basis_units` is in that list. An intercept is legal only when the output's unit vector is in the list. Include the zero vector if dimensionless coefficients should remain allowed. Internal nonlinear feature constants remain dimensionless. Values must already use a consistent unit system; dimensional checking does not convert centimetres to metres.

### Search and selection

1. Build typed terminals and simple compositions, rejecting incompatible dimensions and unsupported domains.
2. Rank dictionary candidates using aggregate, output-normalized residual projections. Greedily fit a small shared basis with a separate least-squares head for each output.
3. Generate new compositions and constant mutations from retained features; repeat within the configured budget.
4. Refine internal nonlinear constants jointly across outputs using variable projection: recompute the linear heads at each nonlinear step.
5. Select from the training loss/complexity Pareto candidates using validation loss plus `complexity_weight * complexity`.

Complexity counts unique shared expression nodes, nonzero output connections and nonzero intercepts. `max_terms` limits the number of selected shared basis functions, not the number of inputs or outputs. This release is a practical extension of DimenSR, with no claim of algorithmic novelty or demonstrated superiority to other MIMO methods.

| `SearchConfig` field | Default | Purpose |
| --- | ---: | --- |
| `max_features` | 256 | Retained dictionary budget; must accommodate inputs plus two constants |
| `generations` | 4 | Composition/mutation rounds after initial search |
| `proposals_per_generation` | 160 | Proposed features per round |
| `max_terms` | 5 | Maximum shared basis size |
| `max_nodes`, `max_depth` | 13, 6 | Limits on each feature tree |
| `operators` | `add, sub, mul, div, square, sin, cos` | Enabled expression operations |
| `complexity_weight` | 0.0001 | Complexity penalty in validation selection |
| `validation_fraction` | 0.2 | Internal validation fraction |
| `polish_max_nfev` | 30 | Nonlinear least-squares evaluation budget; 0 disables refinement |
| `random_state` | 0 | Reproducible random seed |
| `interval_max_boxes` | 32 | Interval subdivision budget per feature audit |
| `head_ridge` | 0.0 | Ridge strength on column-scaled output heads; intercepts are unpenalized |
| `search_seconds` | `None` | Optional process CPU time limit for search, including domain checks and pruning |
| `candidate_quota` | 0 | Positive-relevance candidates reserved per output within the 12-candidate shortlist |
| `prune_tolerance` | 0.0 | Optional per-output normalized training MSE allowance for connection pruning/refitting |

Supported operators also include `neg`, `sqrt`, `exp` and `log`. Transcendental inputs must be dimensionless; addition and subtraction require matching dimensions. The search is stochastic and budgeted, so it may miss valid complex expressions.

For noisy data, `SearchConfig(head_ridge=0.001)` enables regularized output heads.
The default zero retains the original least-squares behavior and exact-formula
recovery. Select regularization on validation data. In our [real-data sealed
holdout study](docs/REAL_DATA_BENCHMARK.md), this option reduced aggregate test
NRMSE by 0.88% across three UCI datasets; all paired 95% intervals crossed zero.
Parkinson subject generalization remained worse than the mean predictor, and
Extra Trees performed better on Air Quality. The result does not establish a
general improvement. Raw datasets and fitted tree binaries are not distributed.

For a bounded run, use `SearchConfig(search_seconds=20, generations=10000)` and
limit BLAS/OpenMP threads to one when comparing computation budgets. The search
returns its completed archive at the time limit; diagnostics report CPU usage
and overshoot from uninterruptible numerical calls. Validation, display and
serialization are outside this search limit. Time-limited searches can vary
across machines even with the same random seed; save the JSON model to preserve
the fitted result.

`candidate_quota=1` adds per-output shortlist coverage. `prune_tolerance=0.005`
tries removing weak output connections and refits the remaining head, allowing
at most 0.005 additional normalized training MSE per output relative to the
unpruned candidate. With a time limit, pruning uses a reserved final 20% of the
budget. Both options are disabled by default. Their separate evaluation follows
the [v3 protocol](experiments/real_blind_v3/PROTOCOL.md); neither option by itself
guarantees improved predictions.

The [equal-budget study and new real-data holdouts](docs/REAL_DATA_EQUAL_BUDGET.md)
completed 180 development runs and 34 frozen holdout evaluations. Neither
candidate coverage nor pruning passed its adoption gates, so defaults remain
unchanged. Independent symbolic search had 3.29% lower development NRMSE with
33.48% higher shared-DAG complexity. On new holdouts, Extra Trees was best on
Gas Turbine and Ridge had the lowest point estimate on SML2010; joint symbolic
search showed no general accuracy advantage. All per-output results and failed
adoption gates are retained in the report.

The follow-up [budget and capacity diagnosis](docs/BUDGET_CAPACITY_DIAGNOSIS.md)
completed 108 paired development runs at 20/60 CPU seconds. Tripling the budget
did not improve aggregate joint-model error, and widening the basis bound alone
worsened default operating points. Some gains appeared under tighter complexity
caps. Full inner-validation-selected curves, candidate archives and paired
results are available; this diagnosis changes no algorithm defaults and makes
no new blind-evaluation claim.

The [shared/private prototype](docs/SHARED_PRIVATE_EXPERIMENT.md) then completed
135 training runs and 225 development evaluations. Compared with independent
search using the same validation selector, its primary variant had 2.20% higher
NRMSE and 3.31% higher DAG complexity, so it did not advance. The experimental
trainer, a runnable synthetic demonstration and portable JSON models are
available in `experiments/shared_private_v1`; released defaults remain unchanged.

The subsequent [optional semantic-transfer study](docs/SEMANTIC_TRANSFER_EXPERIMENT.md)
completed 45 paired cells and 180 development evaluations. After full independent
search, transferring subexpressions by residual similarity reduced DAG complexity
by 7.31% with 0.065% higher NRMSE than independent search at the same 25-second
allowance. It outperformed random transfer on aggregate error and complexity,
but missed the prespecified advancement gate. The prototype and runnable demo
are in `experiments/semantic_transfer_v1`; it remains experimental and changes
no released defaults.

The [external evaluation on three new measured datasets](docs/EXTERNAL_TRANSFER_EVALUATION.md)
then froze all 120 models before opening Tetouan power, water-treatment and
video-transcoding holdouts. Semantic transfer reduced mean NRMSE by 0.71% and
DAG complexity by 8.01% against equal-budget independent search, missing the
adoption gate. Released joint search was the strongest symbolic reference on
aggregate, and Extra Trees had the lowest overall error. Per-output failures,
paired block intervals and domain audits are included; the prototype remains
experimental.

The [contextual subtree-transfer experiment](docs/CONTEXTUAL_TRANSFER_EXPERIMENT.md)
adds optional donor substitution inside nonlinear basis functions. Its controlled
whole-basis ablation separates this move from other transfer changes. The report
contains development results, advancement gates and a runnable seeded example;
the released estimator and its defaults remain unchanged.

The [retrospective candidate-library diagnosis](docs/CANDIDATE_LIBRARY_DIAGNOSIS.md)
then scores all 27,420 retained candidates and enumerates 12.95 million systems
without refitting. At the original selected-system complexity, contextual
transfer's hindsight NRMSE is 5.99% below its selected score, almost entirely
outside the current inner-validation eligibility band. Minimizing the same
single validation score does not recover this gap. The report motivates more
stable validation-based selection; hindsight minima are not achieved performance
or new blind-test evidence, and no released algorithm changes are made.

### Domain behavior

`input_bounds` is one finite `[lower, upper]` pair per input. Declare the intended operating range, including endpoints. Candidates must receive `enclosed` from the numerical interval audit on their original expression trees; both `invalid` and `unresolved` are rejected. Algebraic cancellation does not erase an illegal intermediate such as division by zero.

The audit is a conservative numerical check, not a formal proof of global mathematical validity. It can reject a legal formula because of interval dependency or exhausted subdivision. If bounds are omitted, the model derives a box from the training inputs after any internal split; this does not establish validity outside that box. Validation points outside the box receive pointwise numerical checks only. Use `audit_domain(new_bounds)` before considering a new operating range. A low prediction error does not establish boundary legality, physical causality or extrapolation accuracy.

## CSV command line

All CSV files need a header and finite numeric rows. The fitting inputs default to every column outside `--targets`; metadata can select and order a subset. Validation and prediction columns are aligned by stored names, so their order may differ. Duplicate or missing names, nonnumeric values, NaN and infinity are rejected. Extra numeric columns are ignored when selecting named inputs. Prediction CSV headers retain target names.

```bash
dimensr-mimo fit --data train.csv --targets y1 y2 --validation validation.csv --output model.json
dimensr-mimo predict --model model.json --data new_inputs.csv --output predictions.csv
# Equivalent entry point:
python -m dimensr_mimo --help
```

Optional `--metadata metadata.json`:

```json
{
  "feature_names": ["a", "b", "c"],
  "target_names": ["y1", "y2"],
  "input_bounds": [[-2, 2], [-2, 2], [-2, 2]]
}
```

The same JSON may contain `feature_units`, `target_units` and `coefficient_units` as lists of seven-element vectors. If supplied, `target_names` must match `--targets` exactly. Optional `--config config.json` accepts the fields in the configuration table, for example:

```json
{"generations": 6, "max_features": 384, "max_terms": 4, "random_state": 42}
```

Saved JSON retains the expression trees, fitted coefficients and dimensional/domain metadata. It is the model interchange format; printed SymPy expressions are for inspection.

## Provenance and contributions

DimenSR-MIMO reuses DimenSR's MIT-licensed units, expression trees and interval auditing in `src/dimensr_mimo/_vendor/`; attribution is retained under `licenses/`. The MIMO search, estimator, CLI and tests are maintained independently in this repository. The original repository's experiments and private source documents are not included here.

For changes, run `python -m pytest` and include a small reproducible example. Synthetic examples and previously inspected problems are regression evidence, not a blind benchmark. MIT licensed; see [LICENSE](LICENSE).
