# External evaluation on power, water treatment and video measurements

**All 120 external evaluations succeeded. Semantic transfer improves mean
NRMSE by 0.71% and reduces K by 8.01% against equal-budget independent search,
but again misses the fixed advancement gate.** The released joint model is
the strongest symbolic method on aggregate; Extra Trees has the lowest overall
error. Semantic ranking does not outperform random transfer on this batch.

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent, inline
- Origin Mode: external evaluation and verification
- Origin Date: 2026-10-24
- Version Label: external_transfer_v1
- Verification Status: VERIFIED for deterministic artifact reproduction;
  statistical interpretation remains conditional on this design

## Results and decision

The following values are means over five fixed search seeds on each dataset's
single external holdout. NRMSE uses the fitting-set target SD, not test variance.

| Method | Tetouan | Water | Video | Equal-dataset mean | Mean symbolic K |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent20 | 0.903822 | 0.486589 | 0.566642 | 0.652351 | 47.00 |
| independent25 | 0.904305 | 0.486589 | 0.564781 | 0.651892 | 47.47 |
| random25 | 0.906284 | 0.486513 | 0.539160 | 0.643986 | 43.53 |
| semantic25 | 0.908798 | **0.486012** | 0.546899 | 0.647236 | 43.67 |
| Released joint25 | 0.859469 | 0.486589 | 0.536002 | 0.627353 | **31.07** |
| Mean | 1.174248 | 0.486589 | 1.173187 | 0.944675 | n/a |
| Ridge | 0.916894 | 0.488945 | 1.021588 | 0.809142 | n/a |
| Extra Trees | **0.830684** | 0.490508 | **0.462810** | **0.594668** | n/a |

Semantic25's error is 3.17% higher than released joint25 and 8.84% higher than
Extra Trees on aggregate. Random25 has both a lower mean error (0.643986 versus
0.647236) and slightly lower K (43.53 versus 43.67) than semantic25. This batch
therefore provides no aggregate advantage attributable to semantic donor ranking.
The references are reported as specified; none replaces a failed primary control.

| Primary comparison: semantic25 minus independent25 | Relative error | Relative K | Conditional 95% interval for absolute NRMSE difference |
| --- | ---: | ---: | ---: |
| Tetouan | +0.50% | -15.61% | +0.00208 to +0.00709 |
| Water | -0.12% | +140.00% | -0.00137 to +0.00101 |
| Video | -3.17% | -6.84% | -0.02084 to -0.01525 |
| Equal-dataset mean | -0.71% | -8.01% | not estimated across a task population |

The fixed accuracy route requires at least 1% lower aggregate error and the
compression route requires at least 10% lower aggregate K. Both fail. All eight
dataset/output damage guards pass, as does the requirement to be no worse than
independent20. A favorable dataset subset does not change the global decision.
No defaults or algorithm parameters were changed after these scores.

Video shows the clearest conditional transfer benefit over independent search,
but random transfer and the original joint search are still better on its mean
error. Tetouan becomes smaller at a modest error cost; its first two zones and
third zone behave very differently. Water's apparent low NRMSE is not strong
predictive evidence: independent and joint selection return constants, and the
mean predictor has the same score. Semantic transfer changes only 3 of 15 output
heads across its five water runs, with negligible error improvement and higher K.

| Output | Semantic25 test R2 | Released joint25 test R2 |
| --- | ---: | ---: |
| Tetouan zone 1 | 0.637 | 0.632 |
| Tetouan zone 2 | 0.266 | 0.145 |
| Tetouan zone 3 | -5.155 | -3.199 |
| Water BOD | -0.553 | -0.562 |
| Water COD | -0.466 | -0.466 |
| Water suspended solids | -1.032 | -1.041 |
| Video memory | 0.902 | 0.895 |
| Video CPU time | 0.643 | 0.673 |

The negative R2 values expose weak generalization that an aggregate NRMSE alone
could hide. All compared methods also have negative R2 for Tetouan zone 3. A
training-SD-normalized error below one does not imply an advantage over the
unavailable test-mean oracle. These are held-out predictive results, not causal
claims about process mechanisms.

The prototype remains experimental. If development continues, transfer
acceptance across validation groups and choosing when to share are more directly
motivated hypotheses than assuming semantic ranking is universally superior.
No such follow-up was run here. These three test sets are now inspected evidence
and must not later be presented as unseen tests of an algorithm tuned on them.

## Fixed experiment

This evaluation responds to the request to try other real datasets. The
algorithm is unchanged from commit `01eefe1`. The new data preparation,
protocol, executor, analysis and tests were frozen in
[`ddfe872`](https://github.com/Ashore521/dimensr-mimo/commit/ddfe8726a345333f785d1d848b09b12b66ef6068)
before fitting. The [protocol](../experiments/external_transfer_v1/PROTOCOL.md)
contains all predeclared choices and the
[source/input freeze](../experiments/external_transfer_v1/FREEZE.json) pins
their bytes. No previous development dataset, Gas Turbine or SML holdout is
used in this evaluation.

| Dataset | Measured task | Inputs / outputs | Fitting / validation / test rows |
| --- | --- | ---: | ---: |
| Tetouan power | Weather/calendar to power in three city distribution networks | 9 / 3 | 1200 / 10368 / 10512 |
| Water treatment | Influent measurements to effluent BOD, COD and suspended solids | 9 / 3 | 250 / 80 / 86 |
| Video transcoding | Video characteristics and requested format to measured memory and CPU time | 25 / 2 | 1200 / 16665 / 15340 |

Official sources and attribution:

- [Tetouan, UCI 849](https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city),
  Salam and El Hibaoui (2018), [doi:10.24432/C5B034](https://doi.org/10.24432/C5B034).
- [Water Treatment Plant, UCI 106](https://archive.ics.uci.edu/dataset/106/water+treatment+plant),
  Poch (1993), [doi:10.24432/C5FS4C](https://doi.org/10.24432/C5FS4C).
- [Video transcoding, UCI 335](https://archive.ics.uci.edu/dataset/335/online+video+characteristics+and+transcoding+time+dataset),
  Deneke (2014), [doi:10.24432/C58C9K](https://doi.org/10.24432/C58C9K).

All sources show CC BY 4.0; their ZIP/member hashes and preparation counts are
in the [manifest](../experiments/external_transfer_v1/data/manifest.json).
The pinned Tetouan CSV contains 52,416 data rows, whereas the UCI headline says
52,417. Water has 527 source records, of which 101 lack a finite selected field;
426 complete records remain, despite the page's no-missing-values badge.
Video uses all 68,784 measured transcoding records before splitting, rather
than the separate unmeasured video catalogue. These source discrepancies are
recorded rather than silently correcting the data to match a webpage count.

Whole days are split chronologically 60/20/20 for the two time series, with
one-day gaps for Tetouan and seven-day gaps for Water. Gaps exclude 288 and
10 observed rows respectively. Video IDs are assigned whole to 60/20/20
partitions with seed 20261025. Training rows are capped at 1200 using fixed
sampling; every method receives identical rows. The video fitting cap retains
63 of the 659 IDs assigned to the training partition because IDs have unequal
record counts. Validation and test each contain 220 different IDs. This cap
limits coverage of videos and is part of the reported operating point.

All transforms come from fitting data. Tetouan includes sine/cosine calendar
features declared before scoring; Water uses only influent predictors and
excludes all outlet/removal-efficiency leakage; Video IDs are never predictors.
Codec one-hot columns use the four declared formats. Two video fitting columns
(`b`, `b_size`) are constant and use the fixed unit-scale fallback. Targets are
not clipped or logged. Standardized inputs have dimensionless declarations;
these models are not claimed to identify physical equations. The time-series
tasks use contemporaneous measurements, not a claimed forecasting horizon.

## Comparison and sealing

Five seeds share each dataset's single fixed holdout. Each of 15 cells fits
independent20, independent25, random25, semantic25, joint25, mean, Ridge and
Extra Trees. Mean/Ridge repeats are deterministic bookkeeping, not additional
data replications. All fitting and validation selection precede the global
freeze of all 120 model choices. Only then can the evaluator load sealed test
arrays. A changed model or incomplete grid prevents opening them.

The independent and transfer algorithms, donor ranking and compact selector
are exactly those in the [prior experiment](SEMANTIC_TRANSFER_EXPERIMENT.md).
Their shared initial independent20 archive is reused without discarding original
candidates. The primary comparison is semantic25 against independent25, each
with a 25-CPU-second system allowance. The 20-second base, random transfer and
released joint25 are secondary references. Joint25 retains the released
validation selector, so it is not a selector-controlled ablation. The nominal
unique symbolic training work is 80 CPU seconds per cell, 1200 seconds overall.

Ridge uses alpha=1. Extra Trees uses 200 trees, minimum leaf size 2 and one
thread. Both use fitting-only target standardization, inverted at prediction;
Extra Trees therefore balances outputs in its split objective. These fixed
predictive references receive the same fitting rows and no validation tuning,
but do not have matched symbolic-search CPU caps. No symbolic DAG complexity
is assigned to them. Four workers, single-thread BLAS and a 300-second cell
wall timeout are fixed. Selection, export, scoring and final audits are outside
the search cap. No seed replacement or failed-run deletion is allowed.

NRMSE divides each output RMSE by its fitting-target SD. Aggregate means weight
outputs, seeds and datasets equally. K counts exact shared DAG nodes, nonzero
output connections and intercepts. The primary gate is unchanged: at least 1%
lower error with K at most 110% of control, or at least 10% lower K with error
at most 100.5% of control; all primary pairs must succeed, no output may degrade
more than max(0.01, 2% of its control NRMSE), and aggregate error must be no worse
than independent20. A new result does not erase earlier failed gates.

Primary per-dataset error differences have paired conditional 95% percentile
intervals from 1000 resamples. Resampling units are whole video IDs or
seven-calendar-day blocks; all five fixed models and outputs stay paired.
RMSE is recomputed from group squared-error sums and counts. These intervals
do not represent refitting uncertainty, independence of search seeds, all
seasonal dependence or generalization across a population of datasets.

## Domain and engineering verification

Every selected symbolic model passes the unchanged training-domain audit. All
75 symbolic models also receive `enclosed` on the expanded box covering their
training range and observed test inputs. The fractions of test points outside
the original training box are 2.47% for Tetouan, 8.14% for Water and 11.17% for
Video. Expanded audits were performed only after model selection and did not
filter results. `enclosed` is the existing numerical interval-screening status,
not a formal mathematical proof or a claim about unobserved operating ranges.

The 15 training cells and 120 evaluations have zero failures. The complete
model-choice manifest was saved at 12:08:04 UTC and test opening at 12:10:05 UTC
on 2026-10-24. All 120 saved models reproduce their validation predictions
exactly. Actual unique symbolic search consumes 1200.484 CPU seconds; the
largest recorded search-stage overshoot is 0.03125 seconds. The complete fitting
pipeline consumes 1284.922 CPU seconds, with mean/max cell wall time
86.135/87.297 seconds. Recorded baseline mean training times are 0.000, 0.001 and 0.391
seconds for mean, Ridge and Extra Trees respectively; the CPU timer can record
zero for a short fit. These timings do not establish zero cost or matched budgets.

All **167 tests** pass, including group/time isolation, target-column mapping,
fitting-only preprocessing, model-integrity checks before test loading, all
eight methods on synthetic data, paired group-bootstrap arithmetic and gate
boundaries. Frozen-code CI passes Windows/Linux with Python 3.10/3.12, covering
tests, example execution and package building. Deterministic preparation was
repeated without changing artifacts. All 220 current/prior freeze entries
remain intact; all three analysis files reproduce byte-for-byte; independently
reaggregating grouped errors reproduces NRMSE for all 120 evaluations.

The [verification receipt](../experiments/external_transfer_v1/VERIFICATION.json),
[frozen choices](../experiments/external_transfer_v1/MODELS_FROZEN.json),
[per-output scores](../experiments/external_transfer_v1/analysis/scores.csv),
[full method tables](../experiments/external_transfer_v1/analysis/TABLES.md),
[summary and intervals](../experiments/external_transfer_v1/analysis/summary.json)
and [run records](../experiments/external_transfer_v1/runs) retain the evidence.

## Statistical interpretation audit

All 11 experiment-agent checks were considered. No p-values or population-wide
significance claims are made. This is a convenience sample of three tasks,
five fixed models per method and one holdout split per task.

| Check | Finding or scope |
| --- | --- |
| Simpson's paradox | Dataset/output directions differ; all strata and equal-dataset weighting are shown. No row-pooled claim replaces them. |
| Ecological fallacy | Conclusions concern these measured tasks, not all cities, plants, videos or people. |
| Berkson's paradox | Water complete-case selection excludes 101/527 records; dependence on missingness remains possible. |
| Collider bias | No causal adjustment claim; outlet/efficiency variables are excluded from water predictors, but latent process factors are not identified. |
| Base-rate neglect | Classification prevalence is not applicable; mean predictors and per-output R2 expose reference performance. |
| Regression to the mean | No outcome-based selection of extreme cases or pre/post intervention claim. |
| Survivorship bias | All runs and exclusions are accounted for; no failed model is removed. Complete-case selection remains a limitation. |
| Look-elsewhere effect | All eight prespecified methods and all eight target variables are retained, without selecting the best seed. |
| Garden of forking paths | Protocol/code frozen before fitting; global model freeze precedes test loading; no post-score tuning. Task selection is still purposive. |
| Correlation versus causation | Predictive comparisons do not establish physical or clinical mechanisms. |
| Reverse causality | Contemporaneous power/water inputs support reconstruction, not directional causal or future-horizon claims. |

Conditional intervals use only 11 Tetouan time blocks, 19 Water blocks and
220 Video IDs. They should not be promoted into confidence about unseen tasks,
new hardware, new plants, refitted models or all long-term temporal changes.

## Reproduction

In an isolated environment installed with `pip install -e ".[dev]"`, data
preparation downloads only pinned official sources and refuses inconsistent
existing artifacts:

```bash
python experiments/external_transfer_v1/data.py
python experiments/external_transfer_v1/analyze.py --output outputs/external-transfer-recheck
```

The second command rebuilds analysis from the saved evaluation and local grouped
error arrays. These arrays and baseline binary fits are excluded from the public
repository, so a fresh clone requires a separate frozen training/evaluation run
to regenerate them. The public score table and aggregate summary remain directly
inspectable. Never overwrite this run to repeat a CPU-limited fit: create a new
experiment directory and freeze, because scheduling can change searched models.
Symbolic inference models use the existing JSON format and expect the stored
input standardization and feature order from the manifest. Raw datasets,
binary fitted trees and private source documents are not published.
