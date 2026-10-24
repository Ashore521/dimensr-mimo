# Retrospective diagnosis of frozen candidate quality and model selection

Authorized after the user accepted the proposed candidate-library diagnosis.
No training, coefficient refitting, new algorithm, default change, external
holdout opening or new predictive-performance claim is allowed in this round.
All computations use contextual_transfer_v1's 45 completed cells and six
methods. Source code, inputs and this protocol are pinned before rescoring.

## Questions and evidence limits

1. Within the same actual shared-DAG complexity cap, do retained archives
   contain more accurate expressions than the frozen selected model?
2. At a common cap, does context25's retained library offer an advantage over
   independent25 and the controlled whole25 arm?
3. Do inner-validation rankings agree with observed outer rankings, and does
   the existing 1% per-output inner-MSE eligibility filter exclude alternatives?

Outer labels have already been inspected. Every outer-optimal index below is
an in-sample hindsight reference for diagnosis, NOT a deployable selector,
new test score, estimate of attainable generalization, or theoretical bound.
Larger libraries have more opportunities for chance improvement. Conclusions
concern retained, validation-finite archives, not all visited formulas or the
whole search space. Both candidate quality and selection may limit performance.

## Inputs, scoring and legality

Use all original development cells: concrete slump, Parkinson and Air Quality,
three unchanged nested folds and five search seeds. Retain original fitting
transforms, target SDs, groups, time order and row caps. Check all archive,
choice, selected-model, result, split and original source hashes. The input
inventory contains 27,420 archived candidate entries (25,843 scalar-head and
1,577 joint-system candidates). No numeric source arrays are published.

Decode every archived model and reevaluate its fixed coefficients on the
original inner and outer rows. Recompute normalized MSE, NRMSE and R2; verify
stored validation losses and all selected-model scores. Do not alter formulas,
thresholds, parameters, split membership or numerical-domain rules. Archived
candidates inherit the frozen training-box legality checks; outer finite-point
evaluation is not a new box-domain certificate. Record every nonfinite outer
candidate; exclude it only from finite hindsight minima, never silently from
the inner-only selection or failure counts.

Identical serialized representations (features, coefficients and intercepts)
are deduplicated within each archive for enumeration; aliases retain all source
indices. For per-output rank statistics, remove zero-weight bases and deduplicate
exact active head representations so redundant archive entries do not inflate
rank agreement. No approximate semantic or algebraic equivalence is assumed.

## System complexity and hindsight comparisons

Enumerate retained scalar-head combinations exactly. System K is the union of
value-aware DAG nodes plus nonzero output coefficients and intercepts, using
the original accounting. Do not sum standalone head costs, which double-counts
shared nodes. Joint candidates remain indivisible systems; no recombination of
their heads is added. Cache decoded features; run with one numerical thread.

For each method/cell evaluate three prespecified caps:

- own: the K of that method's frozen selected system;
- common: the K of independent25's frozen selected system in the same cell;
- unbounded: a labeled secondary sensitivity check, not a fair-cost comparison.

At each cap record: (a) the tuple minimizing mean inner NRMSE, with its outer
score revealed afterward; (b) the tuple minimizing mean outer NRMSE, a hindsight
reference; (c) for scalar archives only, the outer minimum among tuples satisfying
the unchanged 1% per-output inner-MSE filter. Ties favor lower K then source
indices. The inner-only cap selector is a diagnostic counterfactual, distinct
from the released compact or joint-penalty selectors. It does not consult outer
finiteness or labels to break ties. Nonfinite outer predictions remain a failed
counterfactual rather than triggering selection of a replacement.

Own-cap hindsight gaps separate the room remaining after the 1% filter from
alternatives the filter excluded. The selected model is always feasible at its
own cap. Empty common-cap/eligibility sets are explicitly recorded, not silently
replaced with a looser cap. Common-cap outer minima compare retained-library
quality only; they are never substituted for frozen study results.

## Per-output diagnosis

For each of 720 method/cell/output records, record the frozen selected errors,
active standalone head K, library size, Spearman inner/outer rank correlation,
selected outer rank percentile and own-cap hindsight error/R2. Undefined ranks
(fewer than two distinct score values) are null. Also use the corresponding
independent25 selected active-head K as a common cap; these standalone results
are not combined into a system K claim. Preserve per-output directions instead
of hiding damage in a system average. Joint head projections are diagnostic
views of archived joint systems, not additional fitted models.

## Reporting and reproducibility

Persist one completed checkpoint per cell; resume only byte-verified records
from this freeze. Refuse overwrites and preserve partial failures. Report all
six methods, all datasets/outputs, paired differences and descriptive counts.
Dataset aggregate system errors weight the three tasks equally. Head summaries
are stratified by dataset/output; do not pool unequal output counts into the
system headline. A >=1% own-cap hindsight reduction is a descriptive indicator
of material retained-library headroom, not an advancement/significance gate.
Seeds share data and are correlated; no IID p-values or population claims.

Deliver per-candidate scores, per-output diagnostics, common/own-cap system
tables, a complete source/result verification receipt and a reasoned next-step
recommendation. Rebuild the deterministic summary independently. Test constant
rank handling, nonfinite outer behavior, shared-DAG caps, strict filter effects,
joint indivisibility, deduplication, untouched input files and frozen-data guards.
No future validation selector or shared-parameter optimizer is implemented here.
