# Shared typed dictionary evolution

For input X with n rows and d columns, the model predicts m outputs with a shared
basis and separate sparse linear heads:

    Y_hat[:, j] = b[j] + sum_k A[j,k] * phi_k(X; theta_k)

All outputs use the same phi_k and theta_k. This is a single coupled dictionary
search. Identical raw subtrees with identical constant values are executed once
per prediction batch. Constants in different basis trees are independently
optimized; equality of two displayed symbolic forms does not tie their parameters.

## Training objective and selection

Each output's scale s_j is its training standard deviation. A constant output uses
max(max(abs(Y_j)),1). Search minimizes mean(((Y_hat-Y)/s)^2). This prevents units
such as Pa versus MPa from setting the relative output weights. It does not
remove noise or imply statistically optimal task weighting.

1. Make an explicit or seeded train/validation split. Infer the input box from
   training inputs only unless the user declares one.
2. Build typed primitive features, scaled unary features and bounded common
   denominator features. Strict raw-tree interval screening rejects invalid and
   unresolved candidates before they enter the dictionary.
3. Rank additions using their projections on every output's normalized training
   residual. Try 12 additions per branch, keep four forward-search branches, and
   solve the allowed linear heads by column-scaled least squares. At most
   `max_terms` features can be used by the entire output system.
4. Retain a bounded pool with small structures, random diversity, selected terms,
   and training relevance. Propose typed compositions and constant mutations.
   Repeat for `generations` rounds with `proposals_per_generation` attempts each.
5. Jointly optimize up to four constant-bearing models from each generation's
   training Pareto front using SciPy least_squares. Each residual call refits all
   linear heads (variable projection). Optimized raw features pass interval
   screening again. `polish_max_nfev` is the SciPy budget **per optimized model**;
   finite-difference calls are separately counted in diagnostics.
6. Construct the training Pareto front for normalized squared error and graph
   complexity. Only now inspect validation labels, choosing the minimum of
   validation loss + complexity_weight * complexity. Do not refit on validation.

Complexity counts unique raw graph nodes, nonzero output coefficients, and
nonzero intercepts. Sharing therefore reduces the structural cost. This is a
finite Pareto filter, not DimenSR's NSGA-II or its neural policy. No Torch, neural
policy training, Rademacher regularizer, or upstream PhySO/DSO runtime is required.

## Dimensional and numerical semantics

Units use exact rational SI exponents. Feature operators follow DimenSR's typed
rules. A connection from feature k to output j is legal only if unit(Y_j) minus
unit(phi_k) is in `coefficient_units`, whose default is just the zero vector.
Intercept b_j is permitted only if unit(Y_j) is in that same list. This keeps the
default coefficients dimensionless; adding dimensional coefficient types is an
explicit relaxation. Internal nonlinear constants are dimensionless in v0.1.

Prediction uses the shared raw graph. SymPy expressions are display values and
may simplify away operations, so they are not the inference or domain authority.
The complete output heads are also interval-screened before final selection.
`audit_domain` retains enclosed/invalid/unresolved status on the declared box;
floating-point enclosures are numerical screening, not verified mathematics.
Predicting outside that box checks only the supplied rows.

## Scope and provenance

This is an independent MIT research implementation derived from the three pinned
DimenSR primitives named in THIRD_PARTY_NOTICES.md. It supports multiple inputs,
multiple outputs, differing target dimensions, model persistence, sklearn cloning,
and CSV fitting/prediction. It does not guarantee recovery of arbitrary systems,
support implicit equations, learn causal direction, or establish research novelty.
It uses closed rectangular domains; coupled physical domain constraints require a
different domain representation and are outside this version.
