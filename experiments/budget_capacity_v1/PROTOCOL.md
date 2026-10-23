# Budget and capacity diagnosis (development evidence only)

This protocol precedes scoring. Algorithm source remains the v3 release at
88acb0e; no coverage, pruning or new search mechanism is introduced. Frozen v1--v3
records and all holdout files remain untouched.

## Fixed grid

Reuse v3's three datasets (concrete_slump, parkinsons, air_quality), three nested
folds, capped training rows, transformations and fold hashes verbatim. Seeds 0
and 1 provide a bounded diagnostic, not the full five-seed confirmation study.
For every dataset/fold/seed run these three arms at 20 and 60 process CPU seconds:

* joint8: original joint search, at most 8 basis functions total.
* independent8: one search per output, at most 8 basis functions each, dividing
  the entire system CPU allowance equally without reallocating unused time.
* joint_wide: original joint search, at most 8 times the output count basis
  functions total. This matches an upper bound on total basis slots, not the
  exact representational capacity or realized graph complexity.

All other v3 parameters remain fixed, including ridge=0.001, 192 pool features,
10,000 generations, max_nodes=13, max_depth=6, 128 proposals/generation, 10 polish
evaluations, interval_max_boxes=32, complexity_weight=0.0001. There are 108 runs
and 4,320 nominal search CPU seconds. BLAS/OpenMP use one thread. Four concurrent
processes and a 300-second per-run wall timeout bound execution. Save actual
CPU/wall time, overshoot, head-fit counts, archive sizes and selected basis counts.
Fresh 20-second runs are required because v3 did not persist candidate archives;
do not mix v3 selected-model results with this round's 60-second curves.

## Curves and selection

Persist every candidate in each estimator's training-Pareto archive that passes
the original raw-output training-box interval audit and finite inner-validation
check (`pareto_front_`). No new fitting or polishing happens during curve
construction. This diagnoses the algorithm's retained candidate set, not every
candidate ever visited and not a global optimum over expressions.

For independent output models, enumerate the exact Cartesian product of their
retained archives. Count the union of value-aware expression DAG nodes plus all
nonzero output coefficients/intercepts. Identical independent subexpressions get
the same sharing credit as joint models. Do not use algebraic cancellation.

For each attainable integer complexity cap, select the model/tuple with minimum
mean inner-validation normalized squared error. Break ties by lower actual K,
then stable archive indices. This yields a complete inner-validation-selected
step frontier. The outer fold is used only after choices are fixed, to measure
per-output NRMSE/R2. Do not optimize, smooth or take a minimum over outer scores.
Keep the original v3 selection (validation MSE + 0.0001*K) as a separate default
operating point; for independent searches it retains separate output selection.

Publish every frontier breakpoint and a common integer-cap grid, along with
actual K used. A common graph-complexity cap does not force identical realized K.
Report selection/enumeration/evaluation CPU separately from search CPU: these
post-search costs are not claimed to be equal across methods.

All rows and folds must match the frozen v3 hashes. Failed runs and illegal outer
predictions stay in the record; no replacement seed and no test-driven fallback.
Aggregate only complete paired cells. Curves are equal-weight means over the
six fold/seed cells per dataset, with their descriptive spread; these are
correlated observations, not six IID data replications or confidence intervals.

## Questions and interpretation

1. Does 20 -> 60 seconds improve joint search, and shrink its paired error gap
   versus independent search at the same complexity caps?
2. Does raising only joint's basis limit improve results under those caps?
3. Do training loss improvements transfer to inner validation and outer scoring,
   or is extra search accompanied by poorer generalization?

Report paired changes at caps 10,20,30,40,60,80,120 and at the default operating
points, plus the full curves. Include actual selected K and how often the selected
basis hits its bound. No candidate adoption gate is applied and no algorithm
change follows automatically from a small mean difference. A wider search also
changes allocation within a fixed CPU budget, so this is not a pure causal proof
of capacity or negative transfer. Sampling limits and distribution shift remain
alternative explanations. No new blind claim is made; no Gas/SML test data is
opened. Finish this diagnostic before proposing step-two algorithm changes.
