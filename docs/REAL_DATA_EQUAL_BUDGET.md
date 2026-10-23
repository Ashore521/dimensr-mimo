# Equal-budget MIMO evaluation and a new real-data holdout

The two proposed revisions did not meet the prespecified development gates.
Per-output candidate coverage increased aggregate NRMSE by **0.27%**. Connection
pruning reduced complexity by **9.90%** but increased NRMSE by **0.59%**. Neither
was adopted. Both remain opt-in experimental settings, disabled by default.

The independent-output control had **3.29% lower** development NRMSE than joint
search, with **33.48% higher** complexity under the same shared-DAG accounting.
Joint search therefore demonstrated a useful compactness tradeoff on two of the
three development datasets, rather than an overall accuracy advantage.

## Development protocol and results

The [protocol](../experiments/real_blind_v3/PROTOCOL.md) was committed at
`5e0df6a` before scoring. Each method received a **20 CPU-second search cap per
complete multi-output system** with single-thread BLAS. Independent models
split that allowance equally across outputs. Each of three existing datasets
used three outer folds and five search seeds, with a separate inner validation
partition for selecting the symbolic model. All **180/180** runs succeeded;
none timed out or were replaced. The maximum summed search overshoot was
0.015625 CPU seconds.

Only the old train/validation observations entered this development study.
Parkinson folds isolate subjects; Air Quality folds preserve order and use a
24-observed-row gap at both outer and inner boundaries. Fitting is capped at
1,200 rows. Scales and transformations are learned from fitting rows only.
The seeds repeat fixed folds and are not 15 independent data partitions.

NRMSE is RMSE divided by the fitting-set target standard deviation, averaged
equally over outputs and then fold/seed. Lower is better. K counts exact shared
basis nodes, nonzero output connections and intercepts; independent models
receive the same sharing credit. All figures below are outer-fold scores.

| Dataset | Joint NRMSE / K | Independent NRMSE / K | Coverage NRMSE / K | Pruning NRMSE / K |
| --- | ---: | ---: | ---: | ---: |
| Concrete slump | 0.751266 / 35.47 | 0.661566 / 48.47 | 0.748796 / 30.47 | 0.757738 / 29.20 |
| Parkinson | 1.022873 / 6.20 | 1.008612 / 5.33 | 1.031386 / 6.60 | 1.031386 / 6.60 |
| Air Quality | 0.990900 / 49.93 | 1.003813 / 68.47 | 0.992247 / 57.53 | 0.992170 / 46.73 |
| Equal dataset mean | **0.921680 / 30.53** | **0.891330 / 40.76** | **0.924143 / 31.53** | **0.927098 / 27.51** |

Coverage failed its required 1% accuracy improvement and the per-output guard
on Air Quality's third target. Pruning missed both the maximum 0.5% error
increase and minimum 10% complexity reduction, and failed the per-output guard
on concrete's third target. These thresholds were retained even where the miss
was small. The global selection was **joint**, with head ridge 0.001 as specified
for all four arms; the library's default ridge remains zero.

The arms had equal limits, not identical consumed time: mean search CPU was
20.000 / 20.010 / 20.000 / 16.113 seconds for joint / independent / coverage /
pruning. Pruning reserves the final 20% for deletion/refitting, which usually
finishes early. Full fit/validation/audit/serialization times and separate-storage
complexities are preserved in the run records. CPU limits can produce different
search trajectories on different machines; saved models identify the actual
evaluated results.

## New real-data evaluation

The selected algorithm and configuration were fixed before training on two new
official UCI datasets. Gas Turbine uses nine inputs and two emission outputs,
with 2011--2013 training, 2014 validation and 2015 testing. SML2010 uses nineteen
inputs and two indoor temperatures, with whole UTC dates split chronologically
60/20/20. SML is concurrent thermal reconstruction, not a forecasting benchmark.
Sources, licenses, units and exclusions are recorded in
[NEW_DATASETS.md](../experiments/real_blind_v3/NEW_DATASETS.md).

Joint and independent searches use five seeds each. Fixed mean, Ridge and Extra
Trees predictors provide predictive reference points; these baselines do not
have the symbolic search's equal-CPU constraint. The new validation scores do
not change the chosen algorithm, configuration or seeds.

All 34 training runs were frozen in commit **`a4042f1`** before the single
holdout opening. All **34/34** evaluations succeeded. The table averages the
five seed models' errors, not predictions from an ensemble; mean and Ridge each
have one deterministic fitted model.

| New holdout | Joint | Independent | Mean | Ridge | Extra Trees |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gas Turbine, 7,384 rows | 1.023925 | 1.009603 | 1.213966 | 1.054736 | **0.922214** |
| SML2010, 797 rows | 0.450208 | 0.460987 | 1.502537 | **0.408178** | 0.627795 |

Gas Turbine favors independent symbolic search slightly, while Extra Trees has
the lowest error among these comparators. Joint search's CO/NOx NRMSE is
0.714798/1.333053. **NOx remains a failure case**: its mean test R2 is -0.678863,
and its NRMSE is slightly worse than the training-mean predictor's 1.313761.
The aggregate gain over the mean baseline does not mean every output improved.

SML2010 favors joint over independent search on aggregate error, but Ridge has
the lowest point estimate. Joint per-room NRMSE is 0.454053/0.446362, with mean
test R2 of 0.715095/0.722410. Joint/independent mean K is **31.8/44.4** on Gas
Turbine and **29.8/28.6** on SML2010; shared search is not always more compact.

Paired group-bootstrap intervals use 2,000 resamples and hold the fitted seed
models fixed. Entries below are **comparator minus joint** mean NRMSE, so a
negative interval favors the comparator.

| Comparator | Gas Turbine 95% interval | SML2010 95% interval |
| --- | ---: | ---: |
| Independent symbolic | [-0.026841, -0.002159] | [-0.005049, 0.028055] |
| Ridge | [0.012708, 0.050137] | [-0.121823, 0.036236] |
| Extra Trees | [-0.121231, -0.082853] | [0.069422, 0.301679] |

Gas uses 308 consecutive 24-record blocks, whose calendar identity and mutual
independence are not established. SML has only **nine UTC-day groups**; its
joint-versus-independent and joint-versus-Ridge intervals cross zero. These
intervals describe observed periods and frozen models, without accounting for
all training-set variation or future-site shift. They do not establish general
algorithm superiority.

Every symbolic model passed the numerical interval screen on both its training
box and the expanded box enclosing the test inputs. Nevertheless, 455 Gas rows
and 47 SML rows fell outside the fitting box. This is numerical domain-screening
evidence, not a guarantee of predictive accuracy or a formal domain certificate.
No model or search parameter changed after the holdout opening.

## Evidence and reuse

- [CV summary and every selection gate](../experiments/real_blind_v3/CV_SUMMARY.json)
- [All development models, transformations and run records](../experiments/real_blind_v3/cv/)
- [New holdout summary](../experiments/real_blind_v3/HOLDOUT_SUMMARY.json)
- [Per-model holdout results and intervals](../experiments/real_blind_v3/holdout/results.json)
- [Freeze manifest](../experiments/real_blind_v3/FREEZE.json) and [opening receipt](../experiments/real_blind_v3/HOLDOUT_OPENED.json)
- [New-data models and transformations](../experiments/real_blind_v3/new_fits/)
- [Training and batch code](../experiments/real_blind_v3/run_study.py)
- [Frozen-source protocol](../experiments/real_blind_v3/PROTOCOL.md)

The training source is commit `5e0df6a`; subsequent changes tightened holdout
provenance checks without changing the training algorithm or harness. Saved
symbolic JSONs expect standardized inputs: apply `(X - input_mean) / input_scale`
from the adjacent result record before predicting. Predictions remain in the
original target units. Standardized inputs were treated as dimensionless, so
this study does not establish recovery of physical-unit equations.

The v1/v2 studies and original DimenSR repository remain unchanged. Raw numeric
data and fitted tree binaries are local, excluded from Git; preparation scripts
and official source hashes are published. The datasets use CC BY 4.0 separately
from the software's MIT license.

The implementation passed **117 tests**, including budget interruption,
candidate coverage, pruning, nested split isolation, model reload and freeze
tamper rejection. A wheel and source distribution built successfully; the wheel
contains the new modules and excludes raw datasets and private documents.

For a separate replication, first regenerate the local data from the published
preparation scripts, respecting their immutable-file checks. Use a new output
directory, retain this study's frozen records, and run:

```bash
python experiments/real_blind_v3/run_study.py cv --output outputs/replication/cv --source . --workers 4
python experiments/real_blind_v3/summarize_cv.py --input outputs/replication/cv --output outputs/replication/CV_SUMMARY.json
python experiments/real_blind_v3/run_study.py new --output outputs/replication/new_fits --source . --selection outputs/replication/CV_SUMMARY.json --workers 4
```

The existing holdout-opening marker intentionally blocks reusing this study as
a new blind evaluation. Existing test cases are now regression evidence; a new
algorithm iteration requires new experiment artifacts and fresh holdout data.

These are bounded single-site experiments. Small concrete samples, subject
shift, temporal drift, limited search time and the training cap constrain the
conclusions. Numerical interval screening on original expressions is retained;
it is not a formal global-domain proof. No claim of overall MIMO superiority or
statistically established algorithmic improvement follows from this round.
