# Real-data sealed holdout protocol

Registered before any real-data model fitting on 2026-10-23. Baseline source:
`de2553a163ad0b453d17a74a350afbe862bfb6e3` (`v0.1.0`). This is an internally
sealed holdout study on public data, not an independent third-party blind trial.

## Data and isolation

Three preselected UCI datasets: Concrete Slump Test, Parkinsons Telemonitoring,
Air Quality. `DATASETS.md` and `data/manifest.json` record official provenance,
licenses, missing-value rules, row counts and split hashes. No dataset may be
discarded based on model accuracy. Parkinson clinical labels are interpolated;
the Air Quality license descriptions conflict, so raw data stays local.

Seed 20261023; 60/20/20 train/validation/test. Concrete uses random rows,
Parkinson uses disjoint subjects, Air Quality uses chronological blocks.
Preparation writes test arrays under `outputs/real_blind_v1_sealed/`. Developers
may inspect training and validation only until the final source, configurations,
models and comparison list are recorded in `FREEZE.json`. A final evaluation
script opens all test sets once; all algorithms are then frozen. Test results
will never drive another modification in this study.

## Budget and preprocessing

All methods use exactly the same training rows and validation rows. Training
sets larger than 1200 rows are reduced to a deterministic random 1200-row subset
(seed 20261023); no validation/test subsampling. Input means and standard
deviations are learned only from those training rows; zero deviations become 1.
All inputs are treated as dimensionless standardized measurements. This study
evaluates prediction and numerical domains, not discovery of physical units.
Bounds use only standardized training minima/maxima. No clipping or imputation.
Output normalization uses training standard deviations for every method.

Common symbolic configuration: max_features=192, generations=3,
proposals_per_generation=128, max_terms=8, max_nodes=13, max_depth=6,
polish_max_nfev=10, interval_max_boxes=32, complexity_weight=0.0001,
default seven operators. Seeds 0 and 1. One BLAS thread per process. A 240-second
per-fit process timeout is recorded as failure, never silently dropped.
At most two algorithm revisions may be evaluated on validation; each gets a new
experiment directory. Equal search settings and operation counts/wall times
are reported; equality of settings is not a claim of identical FLOP budgets.

Comparators: training-mean predictor, Ridge(alpha=1), ExtraTreesRegressor
(n_estimators=200,min_samples_leaf=2,n_jobs=1,random_state=seed). These are
predictive context, not symbolic-regression or matched-compute competitors.
No upstream superiority claim follows from this study.

## Selection and final reporting

Primary selection score: average per-output validation RMSE divided by that
output's training standard deviation, averaged equally across datasets/seeds.
Select one global algorithm revision (not a different revision per dataset).
All six dataset/seed runs must succeed for a revision to qualify; any failure,
timeout or missing run makes that revision ineligible (score +infinity).
Adopt an accuracy change only if its mean validation score improves by at least
1%; otherwise keep the baseline. Always
report every dataset, seed, failure, complexity and domain audit.

After freezing, compare baseline, selected revision and all comparators on the
untouched test partition. Report per-output RMSE/NRMSE/R2, per-dataset averages,
all seed results and runtime/complexity. Paired bootstrap uses 1000 resamples
(seed 20261023), resampling rows for concrete, subjects for Parkinson and days
for Air Quality. Report 95% intervals for the selected-minus-baseline mean
NRMSE difference; these conditional intervals do not capture split uncertainty.
Audit a box expanded to contain test inputs after prediction; report the result
without changing the model. Preserve failures and all prior study directories.
