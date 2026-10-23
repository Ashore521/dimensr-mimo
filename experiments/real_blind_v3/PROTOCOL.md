# Real-data study v3: equal search CPU budget and separate ablations

Written before v3 scoring. Earlier v1/v2 sources, data and results are immutable.
This is a bounded development study followed by one new holdout evaluation.

## Development design

Use only the existing train and validation arrays from v1 concrete_slump,
parkinsons and air_quality. Previously opened test sets are not fresh evidence
and are not accessed here. Use five search seeds (0--4), with three fixed outer
folds per dataset. Concrete uses shuffled KFold (seed 20261024), Parkinson uses
GroupKFold by subject, and Air uses TimeSeriesSplit with a gap of 24 observed
rows (not necessarily 24 hours). Seeds reuse folds; they are not independent
data replications.

Inside each outer training fold, reserve 20% for model selection. Use a fixed
shuffled split for concrete, a subject-isolated GroupShuffleSplit for Parkinson,
and a chronological split with a further 24-observed-row gap for Air. Cap fitting
rows at 1200 by fixed fold-specific sampling shared by all arms and seeds.
Input standardization and target error scales are fitted on fitting rows only.
The outer fold is used only for scoring. No tuning against outer fold results.
All standardized inputs are dimensionless; this study does not demonstrate
discovery of equations in physical units. Save input transformations for reuse.

Four arms (45 required runs each):

* `joint`: current joint algorithm with head_ridge=0.001.
* `independent`: one otherwise identical joint algorithm per output, each given
  20/m CPU seconds, trained sequentially, without reallocating unused time.
* `coverage`: joint plus one positive-relevance shortlist slot per output,
  round-robin deduplication and deterministic rotation; shortlist remains 12.
* `prune`: joint plus backward output-connection elimination and head refitting.
  Each output's normalized training MSE may rise by at most 0.005 relative to
  its original candidate, not cumulatively. Fixed initial contribution ranking;
  no validation labels used for deletion/refitting. Remove globally unused bases.

Every arm receives 20 process CPU seconds per complete output system, with
BLAS/OpenMP limited to one thread. All search initialization, candidate creation,
domain screening, head fits, nonlinear polishing and pruning count. Pruning
reserves the final 20% of its budget to process the training Pareto archive from
lowest loss upward; completed pruned and original candidates remain eligible.
No combined coverage+pruning arm is evaluated. Configuration otherwise uses
max_features=192, generations=10000 (time cap normally binds),
proposals_per_generation=128, max_terms=8, max_nodes=13, max_depth=6,
polish_max_nfev=10, interval_max_boxes=32, complexity_weight=1e-4, default operators.
Budget checkpoints cannot interrupt an atomic numerical call; report search CPU,
overshoot, full fit CPU/wall time and diagnostics. Validation, expression display,
serialization and final audit overhead are reported outside the search budget.
Do not equate head-fit counts with FLOPs. Hardware scheduling can affect exact
time-limited discovery; persist fitted models and source/package hashes.

Budget exhaustion returns the completed archive and is normal. A legal constant
fallback is built before feature initialization. Each method run has a 120-second
wall timeout; exceptions, hard timeout, no valid model, nonfinite predictions or
missing runs are failures. Never replace a failed seed or silently omit failures.
Batch resumption skips only complete existing run records.

## Metrics and fixed selection rule

Report per-output RMSE, training-scale NRMSE and R2. Primary S averages NRMSE
equally over outputs, fold/seed and then datasets. Primary K counts the exact
shared basis DAG over all outputs plus nonzero output connections/intercepts;
constants participate by value, and independent models receive the same sharing
credit. Also report independent separate-storage complexity. No symbolic
cancellation is used for domain checks or complexity reduction.

Against `joint`, coverage qualifies if S falls at least 1% and K rises no more
than 10%; pruning qualifies if S rises at most 0.5% and K falls at least 10%.
Both additionally require all 45 runs to succeed and, for every dataset/output,
the paired fold/seed mean NRMSE increment is at most max(0.01, 0.02 times the
joint output mean). If both qualify choose lower S, then lower K. If neither
qualifies retain joint. Selection is global, not dataset-specific. Report the
independent comparison regardless of winner. These are prespecified engineering
gates, not statistical significance claims; correlated folds/seeds are not IID.

## New holdout phase

New official real multi-output datasets: UCI 551 Gas Turbine CO/NOx and UCI 274
SML2010. Sources, licenses, exclusions and split rules are in NEW_DATASETS.md and
new_data/manifest.json. Gas uses 2011--2013 train, 2014 validation, 2015 test;
SML uses whole UTC days split chronologically 60/20/20. SML is contemporaneous
thermal reconstruction, not forecasting. Seal test NPZ separately, and inspect
no test targets, statistics or performance before model freeze.

Fit joint, independent and the globally selected revision (deduplicate if joint
is selected), using seeds 0--4 and the same 20-second system budget/train cap.
Also fit fixed mean, Ridge(alpha=1) and ExtraTrees(200 trees, min_samples_leaf=2,
n_jobs=1, seeds 0--4) predictors on the same fitting rows. These predictive
baselines are accuracy references and are not claimed to have equal CPU budget.
Selection within each symbolic archive uses only the supplied validation set.
New validation results cannot change algorithm/configuration or the global
choice. Save preprocessing, source, configuration, models, provenance and hashes;
commit their freeze manifest before opening holdouts once. Never refit after
opening. Five seed models are scored individually, with aggregate results.

Report all frozen models, including negative findings, time/complexity and
training-domain numerical interval screening. Test support coverage is reported
without claiming extrapolation legality. Paired 2000-replicate bootstrap by
whole UTC days (SML) and consecutive 24-record blocks (Gas, no calendar timestamp
available) summarizes uncertainty conditional on these test periods/models.
Do not call Gas blocks calendar days or treat these intervals as future-site
generalization guarantees. Raw numeric data and ExtraTrees binaries stay local;
publish preparation scripts, hashes, JSON symbolic models, protocol and results.
