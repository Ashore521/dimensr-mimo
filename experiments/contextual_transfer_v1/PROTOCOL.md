# Contextual subtree transfer: first structural development experiment

The user authorized the next step after the thesis reading and proposed plan.
Only step one is implemented: graft legal donor subexpressions into recipient
bases. No shared-parameter binding, adaptive budget allocation, changed default,
new external test, or private thesis upload is part of this round.

## Source and hypothesis

Yan Youcheng, *Research on MIMO System Modeling Based on Symbolic Regression*,
Xi'an University of Technology master's thesis, cover dated June 2024 (the
user's filename says 2025). Printed pages 24-25 motivate shared subtrees;
28-31 describe position crossover and structural mutation. This is an
adaptation, not a reproduction of that algorithm or its classification results.
No thesis text, images, PDF, or implementation is distributed here.

Hypothesis: evaluating legal donor subtrees inside nonlinear recipient contexts
improves the error/complexity tradeoff over whole-basis transfer. Existing
DimenSR-MIMO already supports nonlinear bases and exact-subtree evaluation reuse.
The new operation changes one occurrence in an immutable tree; it does not
mutate a donor, tie parameters, or introduce graph cycles.

## Mechanism and controlled ablation

Keep all original independent candidates. Build a frozen donor bank from
nonterminal subtrees of original archives, after auditing both source roots
and donor subtrees. Do not extract children of an unsafe root. Two new arms
share a random donor-order policy, full-head refitting, acceptance and final
selection; neither uses raw donor/residual correlation to exclude donors.

For each output/training-Pareto parent, sample up to eight eligible donors.
Whole-basis moves add a basis below the eight-term cap or replace the weakest
coefficient-times-training-SD contribution, matching the preceding move family.
The internal arm additionally samples up to four dimensionally matching proper
subtree positions per donor/parent. Terminal positions are allowed; the root
is excluded. Match the insertion node's units, not the final output's units.
Separate random streams choose donors and insertion positions.

Retain the recipient's surrounding tree and unaffected constants; splice the
donor's own constants, then apply existing canonical ordering without algebraic
cancellation. Enforce the same size/depth limits and audit the entire new basis
and refitted output on the fitting-domain box. No protected operators or unsafe
simplifications are added. `unresolved` does not count as a legal candidate and
does not establish illegality. The existing `enclosed` status is numerical
screening, not a formal real-arithmetic proof.

Refit output coefficients using only fitting rows. Retain a move if training
normalized MSE improves or actual complexity falls. Training-Pareto selection
uses the resulting full-model loss, not a donor-only proxy. Process outputs
round-robin; repeat until no improvement or CPU exhaustion. Added candidates
are Pareto filtered; original candidates remain available. Cached evaluations,
bank construction, grafting and all in-search audits count inside the allowance.
The bounded positional sampling is intentionally not exhaustive.

## Data, arms and fixed resource budget

Use only the three original development datasets: concrete slump (7 inputs,
3 outputs), Parkinson (16 inputs, 2 outputs), Air Quality (8 inputs, 3 outputs).
The definitive dimensions and row counts are checked against SPLITS.json at
preparation. Reuse the nine frozen nested splits in budget_capacity_v1: three
folds, five seeds 0..4, 45 matched cells. Preserve subject separation,
chronological 24-row gaps, fitting-only standardization and the 1200-row cap.
No Gas Turbine, SML or external_transfer_v1 test set is read. These are inspected
development data, never a new blind evaluation.

| Arm | System CPU allowance | Role |
| --- | ---: | --- |
| independent20 | 20 s | Common starting archive and fallback |
| independent25 | 25 s | Primary equal-budget accuracy/complexity control |
| semantic25 | Same independent20 + 5 s | Unchanged previous transfer reference |
| whole25 | Same independent20 + 5 s | New engine with internal grafts disabled |
| context25 | Same independent20 + 5 s | Primary candidate, internal grafts enabled |
| joint25 | 25 s | Original joint search and its released selector |

There are 270 evaluations. Reuse base20 once per cell: nominal unique search
work is 85 CPU seconds/cell, 3825 CPU seconds total. Use at most four isolated
subprocess workers, one BLAS/OpenMP thread each, 300-second wall timeout/cell.
No padding for unused time; record actual search, selection, total CPU/wall,
overshoot, attempts, accepted internal moves and selected internal heads.
An extra move family changes how the fixed budget is spent; this is part of
the tested algorithm and is not evidence of equal candidate-evaluation counts.

Retain v3 search settings: pool 192, generations 10000, proposals 128, max terms
8, max nodes 13, max depth 6, constant polishing 10 calls, ridge .001,
interval boxes 32, complexity weight .0001. No constant polishing is added to
the transfer stage. No package installation is needed; record environment
versions in FREEZE.json. Failures/timeouts remain visible and are never replaced.

## Selection, provenance and decision

Finish every fitting arm in a cell before accessing inner validation labels.
For independent/transfer arms, use unchanged compact selection: each output
within 1% (+1e-12) of its best inner MSE, then minimum system DAG K and mean
inner loss as tie breaker. Joint uses its released mean-inner-MSE + .0001*K
selector; it is a reference, not a selector-controlled ablation. Save archives,
models, reload equality and CHOICES.json before outer scoring. Freeze protocol,
code, tests, data and split hashes before any real fitting. Refuse overwrites;
resume only verified completed cells, preserve incomplete directories.

Primary context25 vs independent25 must complete all 45 pairs and pass either:
1% lower equal-dataset mean outer NRMSE with K <= 110% of control, or K <= 90%
with error <= 100.5%. Each dataset/output's mean NRMSE increase must be at most
max(.01, 2% of control); mean error must also be no worse than independent20.
K counts exact shared DAG nodes plus nonzero output coefficients/intercepts.

Attribute improvement specifically to internal grafts only if context25 also
passes that same accuracy-or-compression route and output guards against
whole25. Do not replace a failed primary control with a more favorable arm.
Report semantic25 and joint25 comparisons, all outputs' R2, constant models,
paired differences and descriptive SD. Correlated seeds/folds and three tasks
do not justify IID significance or population-wide generalization claims.

Synthetic checks cover internal nonlinear sharing, partial sharing, no sharing,
dimensionally valid internal-only donation, constant bookkeeping, unsafe source
rejection, hidden denominator zeros, original-candidate preservation, limits,
export/reload and the freeze-before-score pipeline. They establish mechanics,
not fresh discovery/generalization. A separately recorded seeded demonstration
also reports unseen synthetic-row errors for all three sharing cases.

If gates fail, retain the complete negative result and experimental trainer.
If they pass, the next research step still requires separate frozen evaluation;
no change to released defaults or automatic implementation of later steps.
