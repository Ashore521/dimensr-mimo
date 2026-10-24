# Shared/private expressions with validation size control

This protocol is frozen before real-data scoring. Existing library source and
all prior frozen results stay unchanged. This is development evidence, not a
new blind evaluation. No Gas/SML holdout is opened. No outer-fold result changes
the algorithm, allocations, selection tolerance, seed set or acceptance gates.

## Fixed comparisons

Reuse the nine nested splits in budget_capacity_v1/SPLITS.json: three datasets,
three folds, five seeds (0..4). Scales, training cap 1200, subject separation and
time gaps remain exactly v3's. Three training arms at 20 CPU seconds per system
give 135 runs and 2700 nominal CPU seconds. Four subprocess workers use one
numerical-library thread each, with a 300-second wall timeout. Preserve failures
and incomplete runs; no replacement seeds. Rerun controls under this protocol.

- joint: the original joint search, at most eight shared bases in total.
- independent: original per-output searches, eight bases per output, equal
  division of the 20-second system allowance.
- hybrid: 40% of the allowance searches a joint dictionary of at most four
  bases. Choose a domain-safe shared anchor using training MSE + 0.0001*K only.
  Another 40% searches each output's residual, equally divided across outputs,
  with at most four private bases per output. The remaining allowance covers
  training-only head refits, performed round-robin across outputs.

The hybrid head pool contains projected shared candidates, private-only heads
refitted to the original target, and refitted unions of anchor plus private
bases. Each output has at most eight bases; coefficients are output-specific.
Retain each output's training-loss/complexity Pareto set. Final basis and raw
output expressions must pass the unchanged interval audit on the training box.
No algebraic cancellation bypasses the original-tree domain audit. Exact
matching bases are stored once; independent models receive the same graph
sharing credit. This is a bounded two-stage prototype, not joint optimization
of all shared/private structures.

All hybrid stages, anchor checks, projection and head refits count toward its
20-second training cap; remaining allowance bounds subsequent stages. Unused
allocations are not reassigned. Numerical calls can overshoot a checkpoint;
record actual time and overshoot. Validation, final domain checks, graph-tuple
selection, serialization and evaluation are outside search caps for all arms
and are reported separately. Equal limits do not imply equal consumed CPU.

Other settings are v3's: pool=192, generations=10000, proposals=128,
max_nodes=13, max_depth=6, polish_max_nfev=10, interval_max_boxes=32,
head_ridge=0.001, candidate_quota=0, prune_tolerance=0. No new dependency.

## Selection ablation

All training finishes before inner validation is used. Independent and hybrid
archives each produce two models without additional fitting:

- penalty: original per-output inner normalized MSE + 0.0001*K selection.
- compact: retain candidates within 1% of each output's own minimum inner MSE
  (plus 1e-12 numerical slack), then choose the Cartesian-product combination
  with smallest exact shared-DAG K. Break ties by mean inner MSE, then stable
  indices. The 1% tolerance is a fixed rule, not a confidence interval.

Together with joint's original default, this gives five operating points and
225 evaluations. The independent compact control separates selection effects
from search-structure effects. Save all archives and selection indices. Outer
scores never select candidates or tune settings; final-model failures do not
trigger a fallback. Report train/inner/outer NRMSE and R2 per output, actual K,
shared/private basis counts, CPU and pairwise changes over matched cells.

## Prespecified advancement gates

Primary candidate: hybrid_compact. Control: independent_compact. Advancement
requires all 45 paired runs successful with matching split hashes, plus either:

1. At least 1% lower equal-dataset mean outer NRMSE, with mean K no higher than
   110% of control; or
2. At least 10% lower mean K, with mean NRMSE no higher than 100.5% of control.

Both routes require that each dataset/output's mean NRMSE increase is at most
max(0.01, 2% of that control's NRMSE), and aggregate NRMSE is no worse than the
original joint control. Compare hybrid_penalty vs independent_penalty and
compact vs penalty within each archive as explanatory ablations, not alternate
adoption candidates. No failed primary gate is replaced by a convenient arm.

Passing these gates warrants a separate external validation, not an automatic
default replacement or a claim of universal superiority. All three datasets
are previously inspected; fold/seed observations are correlated. Preserve
both successful and unsuccessful outcomes. There is no within-round retuning.
