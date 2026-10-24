# Optional transfer inside nonlinear expression contexts

**All 45 paired cells and 270 evaluations succeeded, but the prototype did not
pass either its advancement or internal-graft attribution gate.** Relative to
equal-budget independent search, mean outer NRMSE falls by 0.160% and DAG
complexity by 7.60%. Against the controlled whole-basis arm, error falls by only
0.0087%, complexity rises by 1.42%, and one output exceeds its damage limit.
The implementation remains experimental; no released default changes.

## Results and decision

These are fresh paired runs on the original development folds. NRMSE divides
each output RMSE by its fitting-set target SD; outputs, cells and datasets are
weighted equally at their respective levels. K is actual selected complexity.
CPU-limited searches can visit different candidates across runs; these values
do not replace the frozen estimates from earlier experiments.

| Method | Concrete slump | Parkinson | Air Quality | Mean NRMSE | Mean K |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent20 | 0.663358 | 1.004039 | 0.999880 | 0.889092 | 43.33 |
| independent25 | 0.661383 | 1.004039 | 0.997371 | 0.887598 | 44.71 |
| semantic25 | 0.659881 | 1.020419 | 0.996567 | 0.892289 | 40.07 |
| whole25 | 0.658767 | 1.012401 | 0.987598 | 0.886255 | 40.73 |
| context25 | 0.656122 | 1.009700 | 0.992712 | 0.886178 | 41.31 |
| joint25 | 0.749114 | 1.022873 | 0.984150 | 0.918713 | 33.93 |

The context candidate's lowest aggregate error point estimate is too small an
improvement to pass the frozen 1% accuracy criterion. Its 7.60% compression is
below the 10% requirement. All eight output guards against independent25 pass,
as does the independent20 fallback requirement; those partial successes do not
override the failed primary gate. Context25 has lower error in 20/45 matched
cells versus independent25 and 15/45 versus whole25. These are descriptive
counts, not independent statistical trials.

| Dataset | Context error change vs independent25 | Context K change |
| --- | ---: | ---: |
| Concrete slump | -0.80% | -12.21% |
| Parkinson | +0.56% | +16.88% |
| Air Quality | -0.47% | -6.09% |

Concrete has a useful local compression tradeoff. Parkinson remains weak:
context25's mean outer R2 is -0.0506 for motor UPDRS and -0.0110 for total UPDRS,
versus -0.0207/-0.0090 for independent25. Across its 30 selected output heads,
12 are constants. The results therefore do not establish improved prediction
across subjects. Concrete's output R2 values are 0.1240, 0.3763 and 0.9366;
Air Quality's are 0.8748, 0.3434 and 0.5547. Full per-output results are retained.

The direct ablation offers little aggregate benefit from internal grafts. On
Air Quality CO, NRMSE rises from 0.381662 for whole25 to 0.393403 for context25:
the 0.011741 increase exceeds the allowed 0.010000. This fails the attribution
guard, in addition to failing both aggregate improvement routes. The lower mean
error than the older semantic arm cannot establish that internal grafting caused
the improvement, since whole25 already achieves nearly the same result.

The operation is active: 84,760 internal proposals were attempted and 5,611
were accepted during training. Of 120 selected context25 output heads, 22 have
an internal graft as their final accepted move, spread across 19/45 cells.
This diagnostic tracks the final move, not all ancestors or a causal effect.
More accepted structures and successful seeded recovery do not establish better
real-data generalization.

These findings support retaining the primitive for research, without adding
parameter binding or a larger graph-search engine on the assumption it will
improve accuracy. Cross-group stability of transfer selection remains a
separate hypothesis: this experiment did not change the selector or isolate
why validation gains fail to transfer. No post-result tuning is performed.

## What changes

Prior transfer adds a donor subtree as a whole basis, or replaces a whole basis.
This prototype can also replace a proper subtree *inside* a recipient basis.
For example, a donor `x0+x1` can replace the argument of `sin(x0)`, yielding
`sin(x0+x1)`. The surrounding function is retained, and the full output head is
refitted before assessing the move. Matching uses the insertion node's units;
a donor need not be linearly connectable to the final output by itself.

This follows the ideas of shared symbolic networks, position crossover and
structural mutation in Yan Youcheng's MIMO symbolic-regression thesis,
printed pages 24-31 (cover dated June 2024). It adapts the ideas to this
repository rather than reproducing the thesis algorithm. Its classification
experiments do not establish continuous multi-output regression performance.
The supplied private PDF is not distributed.

The released implementation already supports nonlinear basis functions,
exact shared-subtree computation and constant polishing. The new capability
is an additional local search move. It is not a new function class, a full
graph evolution engine, explicit shared-parameter optimization or proof that
the learned sharing reflects a physical mechanism.

The donor bank is frozen from original independent candidates. Unsafe source
roots cannot donate children. The modified complete basis and fitted output
must pass the existing raw-domain audit. Surrounding constants stay attached
to their own nodes; donor constants are inserted in traversal order. No
protected division or algebraic cancellation hides an illegal intermediate.
Immutable trees prevent accidental donor edits or graph cycles. Originals
remain available, so the final selector can reject all transfer candidates.

## Fixed comparison

The [protocol](../experiments/contextual_transfer_v1/PROTOCOL.md) and
[freeze](../experiments/contextual_transfer_v1/FREEZE.json) precede real fitting;
the exact frozen code is in commit
[`bd90722`](https://github.com/Ashore521/dimensr-mimo/commit/bd90722be2e38c079c7d15c11ecaf05fca812160).
Three development datasets use the same nine nested folds and five seeds:
concrete slump (7 inputs/3 outputs), Parkinson (16/2), Air Quality (8/3).
Subject separation, chronological gaps, fitting-only scaling and row caps
remain unchanged. These previously inspected observations are development
evidence; no new or previously sealed external holdout is opened.

| Method | Total search allowance | Purpose |
| --- | ---: | --- |
| independent20 | 20 CPU seconds | Common starting candidate pool |
| independent25 | 25 CPU seconds | Primary equal-budget control |
| semantic25 | independent20 + 5 seconds | Unchanged prior semantic transfer |
| whole25 | independent20 + 5 seconds | New transfer engine without internal grafts |
| context25 | independent20 + 5 seconds | Internal grafts enabled |
| joint25 | 25 CPU seconds | Original joint search and released selector |

Whole25 and context25 have the same random donor-selection policy, head
refitting and final selector. Context25 adds up to four sampled compatible
positions per donor/parent; both inspect up to eight donors. Parents and
subsequent searches may diverge. This is a comparison at equal CPU caps, not
identical attempted candidates or consumed time. The semantic arm separately
retains the older residual-correlation ordering. The joint arm retains its
released selector and is not a selector-controlled ablation.

All fitting finishes before inner selection. Independent/transfer heads use
the same compact rule as the preceding study. Archives and all six model
choices are saved before outer scoring. The study has 45 matched cells and
270 evaluations. Common base archives are fitted once: nominal unique search
work is 3825 CPU seconds, using up to four single-thread workers.

The primary advancement gate compares context25 to independent25. Attribution
to internal grafts additionally requires passing the same gate against whole25.
Each comparison requires either 1% lower error with K at most 110% of control,
or 10% lower K with error at most 100.5%; per-output damage limits and the
independent20 fallback requirement also apply. A favorable secondary arm does
not replace a failed primary comparison. K is the exact shared DAG node count
plus nonzero output coefficients and intercepts, for every symbolic method.

## Engineering evidence and reproduction

The complete suite passes 185 tests. New checks cover constant splicing,
dimensionally valid internal-only donation, nonlinear/partial/no-sharing cases,
hidden denominator zeros, limits, preservation of source candidates, JSON
round trips, freeze enforcement, gate boundaries and choice persistence before
outer scoring. The sandbox's default temporary directory was unavailable in
the first targeted run; tests use a fresh workspace-local temporary directory.
A CPU-exhaustion test uses a controlled clock to avoid Windows timer granularity.

The frozen-code [CI run](https://github.com/Ashore521/dimensr-mimo/actions/runs/40102994551)
passes Windows/Linux with Python 3.10/3.12, including tests, the existing public
example and package building. Independent loading of all 270 saved models
reproduces NRMSE and R2 across 810 train/inner/outer partitions; maximum absolute
NRMSE discrepancy is zero. Every model receives `enclosed` on its fitting box
and on the expanded box covering observed fitting/validation/outer inputs.
Expanded checks occur after selection and do not filter results. This remains
numerical interval screening, not a mathematical or physical certificate.

Unique search consumes 3825.875 CPU seconds; the complete fitting, selection,
export and scoring pipelines consume 4021.266 CPU seconds. Mean/max cell wall
times are 89.856/91.344 seconds with four workers, and the largest recorded
search-stage overshoot is 0.03125 seconds. All four analysis outputs reproduce
byte-for-byte in a second directory.

The new 29-entry freeze and all 220 entries in the five latest historical
freezes are intact. The earlier v2 manifest has two pre-existing mismatches
(`config.py`, `search.py`) against today's source; both files are unchanged
from this turn's starting commit. The verification receipt records these
differences rather than claiming that every historic manifest matches current
code.

The [seeded demonstration](../experiments/contextual_transfer_v1/synthetic/summary.json)
uses 200 fitting, 100 validation and 100 new synthetic evaluation rows. It
starts from hand-specified incomplete candidate archives, not from a fresh
unassisted discovery problem. In the internal and partial-sharing examples,
recipient NRMSE falls from 0.1691 and 0.4664 under whole-basis transfer to below
1e-14 with internal grafts. Both methods retain near-machine-precision results
on the no-sharing example. These results validate the mechanism only.

From a repository checkout with the existing development environment:

```bash
python experiments/contextual_transfer_v1/demo.py --output outputs/contextual-demo
python -m pytest tests/test_contextual_transfer.py tests/test_contextual_study.py -q
```

The demo refuses an existing output directory. Its saved models load through
the existing `MIMOSymbolicRegressor.load` API and expect the recorded feature
order. No new dependency or released training API is introduced.

The frozen real-data analysis can be reproduced without refitting:

```bash
python experiments/contextual_transfer_v1/analyze.py --output outputs/contextual-recheck
```

Analysis checks source/data hashes, complete cell identities, preserved base
archives, saved selector choices and model bytes. As in previous studies,
local raw numeric inputs are excluded from publication and are needed for the
full provenance check. Repeating a CPU-limited fit requires a separate experiment
directory and freeze; never overwrite these records.

[Full tables](../experiments/contextual_transfer_v1/analysis/TABLES.md),
[per-output scores](../experiments/contextual_transfer_v1/analysis/scores.csv),
[paired differences](../experiments/contextual_transfer_v1/analysis/paired.csv),
[gates and summary](../experiments/contextual_transfer_v1/analysis/summary.json),
[verification receipt](../experiments/contextual_transfer_v1/VERIFICATION.json)
and [run records](../experiments/contextual_transfer_v1/runs) preserve the full
outcome, including the failed advancement and attribution gates.
