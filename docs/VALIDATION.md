# Release validation

Local validation used Windows and Python 3.12, with the dependency versions in
`experiments/smoke_v2/requirements-lock.txt`. The original DimenSR checkout and
its frozen studies were unchanged. The three vendor source files were checked
byte-for-byte against DimenSR revision 7fbcefadd105fe6266c8149cc05420de8ca74a76.

The final suite passed **63 tests** from the editable source installation and
again passed **63 tests** when imported from an independently installed wheel.
The example also completed with finite test predictions, an enclosed declared
domain, and a JSON save/load prediction roundtrip. Both wheel and source
distribution builds succeeded; the source distribution includes examples/docs,
and both packages include the required license text.

Tests cover shared nonlinear recovery, joint internal-frequency fitting, output
scale imbalance, different target dimensions, explicit dimensional coefficients,
training/validation isolation, sklearn cloning/tags, large finite input scaling,
raw-tree poles/log/sqrt/overflow, unresolved interval budgets, JSON roundtrip and
tampering rejection, failed refit state, CSV name alignment and input protection.

The four author-created systems in the development smoke study use 160 training,
40 validation and 80 test rows, with two search seeds each. All eight runs in v3
completed and passed their declared numerical domain screening and JSON prediction
roundtrip. Numerical recovery (test max absolute error <1e-6) was:

| System | Recovered runs | Largest per-output test NRMSE |
| --- | ---: | ---: |
| Shared polynomial | 2/2 | 9.42e-16 |
| Shared trigonometric | 2/2 | 4.41e-16 |
| Shared rational | 2/2 | 8.68e-16 |
| Nonlinear internal constant | 0/2 | 2.27e-3 |

These are inspected development cases, not blind evaluation. V1 and V2 results
and code remain in Git history; the failed initial metrics serialization run is
also retained. V3 source hashes, saved models, per-output metrics, budgets and
timings are under `experiments/smoke_v3/results`. The nonlinear-constant failures
remain open search/selection limitations despite a separate passing unit test
for joint constant optimization on another fixed regression case.

The wheel and source distribution are built locally. GitHub CI is configured for
Linux and Windows with Python 3.10 and 3.12; only completed CI runs establish those
platform results. There is no upstream matched-budget comparison, real-data
multi-output evaluation, or claim of improved performance over other methods.

## Subsequent real-data study

The historical v0.1.0 checks above remain intact. A separate [real-data report](REAL_DATA_BENCHMARK.md)
now records the UCI data provenance, group/time splits, sealed holdout protocol,
all development runs, frozen model hashes and final outcomes. The added optional
regularized output heads have seven new tests; the complete source suite passed
70 tests and the shared-system example passed. The overall holdout NRMSE changed
from 0.94127 to 0.93297; all dataset-level paired 95% intervals included zero.
No algorithm or hyperparameter was changed after opening those holdouts.
