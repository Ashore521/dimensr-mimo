# Optional subexpression transfer after independent search

**Semantic transfer reduces mean complexity by 7.31% with 0.065% higher outer
NRMSE than independent search at the same 25-second allowance. It does not
pass the frozen advancement gate.** All 45 cells and 180 evaluations succeeded.
The prototype remains available for experimentation; this result does not
justify replacing the released algorithm.

This experiment follows the shared/private prototype with complete independent
search and optional cross-output transfer. The implementation lives in
[`experiments/semantic_transfer_v1`](../experiments/semantic_transfer_v1).
It reuses the released expression representation, domain audit, output fitting
and JSON format. The released estimator and its defaults are unchanged.

## Results and decision

| Method | Outer NRMSE | Actual DAG K | Mean training CPU seconds |
| --- | ---: | ---: | ---: |
| independent20 | 0.889900 | 42.58 | 20.010 |
| independent25, primary control | **0.888640** | 42.87 | 25.021 |
| random25 | 0.895412 | 43.67 | 25.011 |
| semantic25, primary candidate | 0.889222 | **39.73** | 24.997 |

Compared with the primary control, semantic transfer changes mean NRMSE by
+0.000582 (paired descriptive SD 0.030711) and K by -3.133 (SD 13.533). It has
lower error in 18/45 matched cells. Compared with random transfer, its mean
error is 0.69% lower and K is 9.01% lower. This ablation favors semantic ranking
on the aggregate tradeoff, but cannot substitute for passing the primary
comparison or establish a general benefit across tasks.

| Dataset | independent25 NRMSE / K | semantic25 NRMSE / K | Error change | K change |
| --- | ---: | ---: | ---: | ---: |
| Concrete slump | 0.664305 / 46.80 | 0.659439 / 42.47 | -0.73% | -9.26% |
| Parkinson | 1.004039 / 5.13 | 1.017052 / 5.93 | +1.30% | +15.58% |
| Air Quality | 0.997576 / 76.67 | 0.991174 / 70.80 | -0.64% | -7.65% |

Concrete and Air Quality improve modestly in both aggregate metrics, while
Parkinson shows negative transfer. Across all datasets, semantic transfer
reduces inner NRMSE from 0.655303 to 0.644829, but that gain does not translate
into better aggregate outer error. Keeping independent candidates and selecting
on one inner split therefore does not eliminate negative transfer.

The frozen accuracy route requires at least 1% lower error with K at most 110%
of control. The compression route requires at least 10% lower K with error at
most 100.5% of control. Neither route passes: accuracy does not improve, and
the 7.31% compression is below 10%. Both additional safeguards pass: aggregate
error is no worse than independent20, and all eight dataset/output error
increases are within `max(0.01, 2% of control NRMSE)`.

| Output | Control NRMSE | Semantic NRMSE | Allowed increase | Guard |
| --- | ---: | ---: | ---: | --- |
| Concrete: slump | 0.902950 | 0.907344 | 0.018059 | pass |
| Concrete: flow | 0.837264 | 0.816315 | 0.016745 | pass |
| Concrete: compressive strength | 0.252700 | 0.254658 | 0.010000 | pass |
| Parkinson: motor UPDRS | 1.015603 | 1.031557 | 0.020312 | pass |
| Parkinson: total UPDRS | 0.992475 | 1.002546 | 0.019850 | pass |
| Air Quality: CO | 0.389451 | 0.383557 | 0.010000 | pass |
| Air Quality: NOx | 1.742945 | 1.726793 | 0.034859 | pass |
| Air Quality: NO2 | 0.860333 | 0.863173 | 0.017207 | pass |

The evidence supports retaining semantic transfer as a compact-model research
direction, with cross-split transfer acceptance as a specific next hypothesis.
Such a follow-up would need a new freeze and separate evaluation; it was not
performed here. The present prototype is not promoted, and neither physical
parameter tying nor reinforcement learning is added on these results.

## Method and fixed comparison

The [protocol](../experiments/semantic_transfer_v1/PROTOCOL.md), implementation,
tests and advancement gates were frozen in commit
[`454f42f`](https://github.com/Ashore521/dimensr-mimo/commit/454f42f54199b0b0eb404cd91ff543847c2020d5)
before this real-data batch. Source and input hashes are recorded in
[FREEZE.json](../experiments/semantic_transfer_v1/FREEZE.json).

There are 45 matched cells: three existing development datasets, three nested
folds and five seeds. Each cell yields four operating points:

| Method | Training allowance per output system | Purpose |
| --- | --- | --- |
| independent20 | 20 CPU seconds | Original independent-search baseline |
| independent25 | 25 CPU seconds | Primary control with the same total allowance |
| random25 | The same independent20 archives + 5 CPU seconds | Random donor-order control |
| semantic25 | The same independent20 archives + 5 CPU seconds | Primary candidate with residual-based donor ranking |

Independent search divides its allowance equally among outputs. The baseline
archives are trained once per cell and reused by both transfer arms; unique
nominal training work is 55 CPU seconds per cell, or 2475 seconds overall.
Four subprocess workers use single-thread BLAS/OpenMP and a 300-second wall
timeout per cell. Model selection, final auditing, serialization and scoring
are outside the training allowance for every method. Donor-bank construction
and the audits performed during transfer count inside its five seconds.

Transfer donors are nonterminal subtrees, including whole bases, from other
outputs' original archives. Constants stay attached to their own subtrees.
Both source roots and donor subtrees must pass the original-tree interval
audit, finite-value checks and declared coefficient-unit compatibility.
There are no protected divisions, absolute-value square roots or algebraic
cancellations that would erase illegal intermediates.

For each training-Pareto parent, semantic transfer ranks donors by squared
cosine with its current training residual. Signs are irrelevant because the
output coefficients are refitted. Features are centered only when an intercept
is legal. Random transfer shuffles the same donor pool. Both try up to eight
donors per parent, adding a basis below the eight-term limit or replacing the
weakest contribution, then refitting on the original training target. A move
must improve training loss or reduce complexity and pass the raw output audit.
Outputs are processed round-robin. All original candidates are retained;
training-Pareto additions extend the archive. These bounded local moves do not
constitute exhaustive DAG optimization.

All four arms finish fitting before inner-validation labels are used. The
unchanged compact selector keeps candidates within 1% of each output's best
inner MSE, then minimizes actual system DAG complexity. The 1% tolerance is a
selection rule, not a confidence interval. Candidate archives, all selected
models and `CHOICES.json` are saved before outer scores are computed. The
selector can reject every transferred candidate; retaining baseline heads
does not guarantee a lower outer error.

## Transfer activity and verification

| Transfer stage | Mean donor count summed over outputs | Mean attempted refits | Mean training-accepted moves | Final selected transfer heads | Cells selecting transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 561.58 | 1618.89 | 825.04 | 50/120 | 32/45 |
| Semantic | 561.58 | 1028.47 | 708.96 | 46/120 | 35/45 |

Donor counts match in every cell. Training-accepted moves are intermediate
local improvements, not unique final discoveries. Semantic ranking adds work
within the same cap; this comparison does not match the number of refits.
Mean selected whole bases used by multiple outputs are 0.600 for independent25,
0.556 for random25 and 0.667 for semantic25. K also credits shared subtrees;
its reduction cannot be attributed entirely to sharing whole bases.

Random transfer uses 5.001 CPU seconds on average and semantic transfer uses
4.986 seconds. One semantic stage exhausts its search before the cap; unused
time is not padded. Maximum recorded stage overshoot is 0.015625 seconds for
each transfer arm and 0.046875 seconds summed across independent25 outputs.
Total unique training CPU is 2477.953 seconds. Including selection, auditing,
serialization and scoring, the mean cell consumes 58.101 CPU seconds and
58.494 wall seconds; the largest cell wall time is 59.656 seconds.

The full suite passes **156 tests**, covering retained independent archives,
subtree constants, dimension/domain rejection, both transfer rules, exhausted
budgets, validation rejection, pre-scoring persistence and gate boundaries.
The frozen commit's CI passes on Windows/Linux with Python 3.10/3.12, including
tests, the example and package builds. A five-second synthetic smoke run gives
validation NRMSE 0.000742/0.000681 and K=68 with exact JSON reload predictions.
This is an engineering demonstration, not external validation.

All 180 real-data models pass exact validation-prediction reload checks.
All four analysis files rebuild byte-for-byte into a separate directory.
The 27 new freeze entries and 162 prior freeze entries remain intact, and an
independent aggregation of saved per-output scores matches the reported means.
The [verification receipt](../experiments/semantic_transfer_v1/VERIFICATION.json)
records checks and hashes.

[Full tables](../experiments/semantic_transfer_v1/analysis/TABLES.md),
[per-cell/output scores](../experiments/semantic_transfer_v1/analysis/scores.csv),
[paired changes](../experiments/semantic_transfer_v1/analysis/paired.csv) and
[gates and file hashes](../experiments/semantic_transfer_v1/analysis/summary.json)
retain every outcome. [Run records](../experiments/semantic_transfer_v1/runs)
contain the candidate archives, pre-scoring choices and portable models.

## Data and interpretation

The experiment uses only the original train/validation pools of Concrete Slump,
Parkinsons Telemonitoring and Air Quality. The nine
[nested splits](../experiments/budget_capacity_v1/SPLITS.json) are unchanged:
mixture-level random folds for concrete, subject separation for Parkinsons,
and chronological inner/outer splits with 24-row gaps for Air Quality. Fitting
uses at most 1200 rows and training-only input standardization. Original source
and cleaning details remain in the [dataset record](../experiments/real_blind_v1/DATASETS.md).

NRMSE is RMSE divided by the fitting-set target standard deviation, averaged
equally across outputs within each cell and equally across the three datasets.
K counts exact shared expression DAG nodes, nonzero output connections and
nonzero intercepts. Every method receives the same sharing credit. Cross-fold
and cross-seed differences are descriptive; the 45 cells are not 45 independent
datasets, and this study does not claim an IID significance test.

These previously inspected datasets provide development regression evidence.
No new blind evaluation was performed, and Gas Turbine and SML holdouts were
not reopened. Standardized real-data runs use dimensionless declarations;
the unit-compatibility tests establish software behavior, not physical validity
of these fitted models. Physical parameter tying remains deferred until data
with a declared common mechanism and compatible units are available.

## Use and reproduction

Run the synthetic example from a clone installed in an isolated environment:

```bash
python -m pip install -e ".[dev]"
python experiments/semantic_transfer_v1/demo.py --seconds 10 --output outputs/semantic-demo.json
```

The demonstration prints validation errors, K, consumed training CPU and
accepted training transfers, and checks exact predictions after JSON reload.
It refuses to overwrite an existing output. The experimental trainer is a
repository script rather than a stable public estimator API.

The saved real-data models expect `(X - input_mean) / input_scale`, in the
feature order recorded for that fold in `SPLITS.json`; their outputs retain
the original target scale. Use `MIMOSymbolicRegressor.load(...)` for inference.
The saved bounds cover fitting inputs. Outside those bounds, finite predictions
alone do not establish interval validity; audit a new operating box before
relying on it.

After preparing the original development inputs with the pinned
[`prepare_data.py`](../experiments/real_blind_v1/prepare_data.py), rebuild the
analysis without refitting:

```bash
python experiments/semantic_transfer_v1/analyze.py --output outputs/semantic-transfer-recheck
```

The destination must be new. The analysis verifies frozen source/data hashes,
all 45 identities, split hashes, archive preservation, model hashes and saved
choices before applying the fixed gates. CPU-limited training may vary across
machines even at the same seed; the archived models preserve this run. New
training or algorithm changes require a separate experiment directory and
freeze, preserving these records. Raw numeric datasets, fitted tree binaries
and private source documents are excluded from publication.

## Literature basis

- Ferreira et al. (2023), MOKPIM,
  [doi:10.1007/s00521-023-08335-0](https://doi.org/10.1007/s00521-023-08335-0):
  independent search islands and optional expression exchange motivate the
  complete independent baseline and migration stage.
- Wang et al. (2025), MTGPSC,
  [doi:10.1016/j.patcog.2024.111289](https://doi.org/10.1016/j.patcog.2024.111289):
  semantic subtree transfer motivates donor ranking and the random-transfer
  ablation. Its [public code](https://github.com/yuriwong33/MTGPSC) was inspected;
  no implementation was copied.

This is an adaptation to typed bases and refitted linear output heads, not a
reproduction of either paper or an independent confirmation of its results.
