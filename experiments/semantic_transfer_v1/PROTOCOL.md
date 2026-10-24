# Optional semantic transfer after complete independent search

Authorized by the user's request to optimize in the recommended order. Freeze
this protocol, trainer, analysis gates and tests before new real-data scoring.
Preserve all prior code and frozen studies. These are development regressions;
no Gas Turbine or SML holdout is reopened. No new dependency is required.

## Hypothesis and literature adaptation

MOKPIM (Ferreira et al., 2023, doi:10.1007/s00521-023-08335-0) motivates
independent search with optional exchange. MTGPSC (Wang et al., 2025,
doi:10.1016/j.patcog.2024.111289) motivates semantic donor selection. This
implementation adapts those ideas to typed basis functions and refitted linear
heads; it is not a reproduction of either paper or their performance claims.
No external implementation is copied. Dimensional compatibility and strict
original-tree interval auditing remain unchanged.

The hypothesis is that optional legal subexpression transfer improves the
error/complexity tradeoff without reducing independent search capacity. A
random-transfer arm isolates the effect of semantic donor ranking. Selection
is held fixed; physical parameter tying and reinforcement learning are outside
this round because the available data provide no declared shared mechanism.

## Fixed data, controls and resources

Use the same nine nested splits from budget_capacity_v1/SPLITS.json: concrete
slump, Parkinson and Air Quality, three folds, five seeds (0..4). Preserve the
v3 training-only transformations, 1200 fitting-row cap, subject separation and
24-row chronological gaps. Existing inspected data never become new blind data.

Each of 45 cells trains the following four models, yielding 180 evaluations:

- independent20: original independent search, 20 CPU seconds per system.
- independent25: original independent search, 25 CPU seconds per system;
  **primary equal-budget control**, separate search with the same seed.
- random25: exactly independent20's raw candidate archives, followed by a
  five-CPU-second optional transfer stage with random donor ordering.
- semantic25: the identical starting archives and five-second stage, ordering
  donors by squared cosine with the recipient's current training residual;
  **primary candidate**. Center features only when an intercept is legal.

Independent budgets are divided equally among outputs. Base20 is trained once
per cell and reused by the two transfer arms: nominal unique work is 55 seconds
per cell, 2475 CPU seconds total. Four isolated subprocess workers use one BLAS/
OpenMP thread each and a 300-second wall timeout per cell. Record actual search,
transfer, selection, total CPU and wall time. Equal caps do not imply equal
consumed time; do not pad early exhaustion or conceal it. Native calls may
overshoot checkpoints; record overshoot. Validation/audit/export after training
is outside training caps for all arms. Transfer-bank auditing and candidate
head auditing during transfer count inside its five seconds.

Keep v3 settings: pool 192, generations 10000, proposals 128, eight bases per
output, max nodes 13, max depth 6, polishing 10 evaluations, ridge .001,
interval_max_boxes 32, complexity_weight .0001; other options remain disabled.

## Transfer and selection

Freeze donor banks from the original per-output training archives. A donor is
a nonterminal subtree (including the full basis) originating from another
output. Extract its own constants without cancellation or protected operators.
Reject unsafe source bases before extracting subtrees; recheck subtrees,
finite values and coefficient-unit compatibility. Exact trees and constant
values define identity, matching existing DAG accounting.

For each output and current training-Pareto parent, try up to eight donors in
the declared order. Try addition when fewer than eight bases are used and
replacement of the weakest existing contribution (absolute coefficient times
training SD). Refit coefficients on original training targets. Retain a move
if it improves training normalized MSE or lowers complexity, and passes the
raw output domain audit. Process outputs round-robin; repeat on updated beams
until the deadline or a round adds no candidate. Original heads are always
preserved, even if dominated; added heads are training-Pareto filtered. This
bounded local search is not an exhaustive optimum over shared DAGs. A donor
accepted during training need not survive final validation selection.

Complete all four training arms before inner validation. Apply the unchanged
compact rule from shared_private_v1: candidates within 1% of each output's
minimum inner MSE (+1e-12), then minimum actual system DAG K, tie by mean inner
MSE and stable indices. It may reject every transfer. Retaining base candidates
does not guarantee lower outer error. Persist all archives, model hashes and
CHOICES.json before any outer metric is computed. All outputs, seeds, failures
and incomplete cells remain in the report. No result-driven within-round tune.

## Advancement and attribution

Primary semantic25 versus independent25 must have all 45 matched cells valid
and satisfy either: (a) at least 1% lower equal-dataset mean outer NRMSE with
mean K at most 110% of control; or (b) at least 10% lower K with mean error at
most 100.5% of control. Additionally, each dataset/output's mean NRMSE increase
must be at most max(.01, 2% of its control NRMSE), and aggregate error must be
no worse than independent20. No alternative arm replaces a failed primary.

Report random25 versus independent25 and semantic25 versus random25 as
ablations. If semantic25 fails to improve aggregate error or complexity over
random25, do not attribute gains to semantic ranking. Report selected transfer
heads, shared/private basis counts, donor counts and attempted/accepted moves.
All methods receive identical exact-DAG sharing credit.

Use descriptive paired differences and SD; seeds within folds are correlated,
and three datasets do not justify an IID significance claim. Passing warrants
fresh external validation, not an automatic released default change. If gates
fail, preserve the prototype and negative findings; do not launch unbounded
retuning or claim physical sharing benefits without suitable physical data.
