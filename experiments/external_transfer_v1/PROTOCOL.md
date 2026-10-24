## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan and run, inline
- Origin Date: 2026-10-24
- Verification Status: UNVERIFIED at protocol creation
- Version Label: external_transfer_v1

# Frozen external evaluation on three new measured datasets

The user requested evaluation on other real datasets. This authorizes this
bounded implementation and run. Keep the released algorithm and all earlier
studies intact. No outcome-dependent algorithm, feature or split changes are
allowed. This is a new external holdout evaluation of a fixed prototype, not
another tuning round. These public datasets are new to this project's model
evaluation; there is no claim that they were absent from an LLM's pretraining.

## Data and fixed tasks

All three official UCI sources are CC BY 4.0. Inspect schemas and identifiers
only before freezing; do not print held-out targets or their statistics.
Download ZIPs and record ZIP/member SHA-256. Raw data and all numeric arrays
remain local and ignored by Git. Sources were checked on 2026-10-24.

| Dataset | Official source / attribution | Fixed inputs and outputs |
| --- | --- | --- |
| Tetouan power | [UCI 849](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city), Salam and El Hibaoui (2018), [DOI](https://doi.org/10.24432/C5B034) | Five weather measurements plus sine/cosine of hour and weekday; all three regional power measurements are outputs |
| Water treatment | [UCI 106](https://archive.ics.uci.edu/dataset/106/water+treatment+plant), Poch (1993), [DOI](https://doi.org/10.24432/C5FS4C) | Nine influent measurements Q-E through COND-E; effluent DBO-S, DQO-S and SS-S are outputs |
| Video transcoding | [UCI 335](https://archive.ics.uci.edu/dataset/335/online+video+characteristics+and+transcoding+time+dataset), Deneke (2014), [DOI](https://doi.org/10.24432/C58C9K) | All 17 numeric source-video/requested-format attributes; fixed four-category one-hot encodings of input/output codecs; measured memory `umem` and CPU time `utime` are outputs |

Water is a task adaptation from stage-sensor data originally offered for plant
state analysis. Exclude all intermediate/outlet measures and removal-efficiency
variables from predictors. Only the three named measured pollution indicators
are evaluated. Video uses only `transcoding_mesurment.tsv`, not the large list
of unmeasured videos. Original video ID is exclusively a split/group identifier.
Codec categories are flv, h264, mpeg4 and vp8, observed in source schema checks;
they are not selected by outcomes. Unexpected categories fail preparation.

Require finite selected numeric fields; `?` is missing. Use complete cases,
without imputing, clipping, removing response outliers or transforming targets.
Document exclusions. For time series, sort timestamps and retain the first
source record per timestamp before filtering. No causal or forecasting claim:
weather and pollutant inputs are contemporaneous measurements. No data-derived
physical units are inferred after input standardization.

Split whole eligible days chronologically 60/20/20 for Tetouan and Water. Purge
one full calendar day before each boundary for Tetouan, seven days for Water.
For Video, permute unique video IDs using seed 20261025 and assign entire IDs
60/20/20. No video may cross partitions. Cap fitting rows at 1200 by fixed
sampling using the same seed. Preserve validation/test rows in full. Fit input
mean/SD, target mean/SD and constant-column handling from fitting rows only.
The error denominator is the fitting-target SD. Store transforms, split-ID
hashes, group counts, bounds and array hashes; seal test arrays under outputs.

## Fixed algorithms, budgets and outputs

Use five search seeds 0--4 on each fixed split: 15 cells and 120 evaluations.
The five seeds share the same holdout and are not independent data replications.
Eight variants are evaluated in every cell:

- independent20, independent25, random25, semantic25: the unchanged frozen
  `semantic_transfer_v1` implementation, budgets and compact inner selector.
  independent20 is fitted once and reused by both five-second transfer stages.
- joint25: the released joint estimator with its existing validation selector,
  25 system CPU seconds and otherwise the same base configuration. This is a
  released-method reference, not a selector-controlled ablation.
- mean, Ridge(alpha=1) and ExtraTrees(200 trees, min_samples_leaf=2, n_jobs=1):
  fixed accuracy references. Fit training-standardized targets and invert this
  transform at prediction. No validation tuning. They share all fitting rows
  and input transforms, but are not claimed to have matched training CPU caps.
  Extra Trees uses the matching seed; mean/Ridge repeat deterministically across
  seeds solely to align records and do not add independent evidence.

Unchanged symbolic configuration: max_features 192, generations 10000,
proposals 128, terms 8/output, nodes 13, depth 6, polish evaluations 10,
ridge .001, interval boxes 32, complexity weight .0001, default operators.
Time allocations are system totals and independent search divides by outputs.
Original-tree dimension/domain policy is unchanged. Store all candidate
archives and selected symbolic JSONs; binary baseline fits remain local.

Nominal unique symbolic search is 80 CPU seconds/cell, 1200 seconds total.
Run four subprocess workers with one BLAS/OpenMP thread and a 300-second wall
timeout per cell. Record CPU, overshoot, selection and wall time; monitor the
batch log and child results. Only the hard timeout terminates a cell. Preserve
failures; do not replace seeds, retry failed training or silently omit rows.
Model selection/auditing/export and conventional baselines are outside search
caps. Resume only complete identical-freeze records.

## Sealing, analysis and interpretation

Freeze code, protocol, tests, packages and input hashes before fitting. All 15
cells must finish training/validation selection and all 120 model choices must
be saved in `MODELS_FROZEN.json` before the evaluator loads ANY test arrays.
Training has no test-array loader. The evaluator verifies model and source
hashes. Score once, preserve per-output RMSE/NRMSE/R2 and all failures. Store
numerical prediction/error arrays locally; public tables contain aggregate
metrics, model metadata and hashes. Test observations never select a model.

Primary comparison is semantic25 versus independent25. Average NRMSE equally
over outputs, seeds and then datasets. Complexity K uses the same exact shared
DAG plus nonzero output links/intercepts for all symbolic variants; no K is
assigned to conventional baselines. Report every dataset/output and both random
transfer and released-joint references, regardless of the primary result.

Retain prior engineering gates: all 15 primary pairs valid; either >=1% lower
mean error with K <=110% of control, or >=10% lower K with error <=100.5% of
control; each dataset/output increase <=max(.01, 2% of control); and mean error
no worse than independent20. Passing here does not erase the previous failed
development gate or automatically change defaults. Report test-point fractions
outside training bounds and expanded observed-box audits; a finite test
prediction is not a guarantee of numerical legality throughout the new box.

For the primary error difference, report a conditional paired 95% percentile
interval from 1000 bootstrap resamples, seed 20261026: whole original video IDs
for Video, seven-calendar-day blocks anchored to the first test day for both
time series. In every replicate preserve method/seed pairing, recompute RMSE
from summed squared errors and counts, then average over seeds/outputs. Models
remain fixed; these intervals do not capture refitting uncertainty or all
long-range dependence. Do not bootstrap search seeds as independent samples,
claim an IID p-value, pool rows across datasets, or select a winner by its CI.

Rebuild analysis into a second directory and require identical artifacts.
Verify earlier freezes unchanged, model round trips, preprocessing isolation,
holdout-opening gate, metrics and bootstrap arithmetic. Save a report and
verification receipt. No algorithm tuning or new dataset substitution after
any test score is observed. Public publication remains within the existing
repository authorization and excludes raw datasets, binaries and private files.

## Commands and artifacts

From the repository root, using its existing isolated Python environment:

```text
python experiments/external_transfer_v1/data.py
python experiments/external_transfer_v1/study.py freeze
python experiments/external_transfer_v1/study.py batch --workers 4
python experiments/external_transfer_v1/evaluate.py
python experiments/external_transfer_v1/analyze.py
```

Expected outputs: `data/manifest.json`, `FREEZE.json`, 15 `runs/*/result.json`,
`MODELS_FROZEN.json`, `evaluation/results.json`, `analysis/`, and a report.
No overwritten artifact is considered a successful resume.
