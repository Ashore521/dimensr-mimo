# New real-data holdouts for real_blind_v3

This source and preparation protocol was fixed before fitting any model to the two new datasets. They were not used in the preceding v1/v2 development. Baseline code commit: `cfea3b1`. Preparation seed: `20261024`; both splits below are deterministic chronological/group splits and do not use random shuffling. Official sources were checked during this preparation before accessing the numeric records.

## Official sources and selection

- **Gas Turbine CO and NOx Emission Data Set:** [UCI 551](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set), [DOI 10.24432/C5WC95](https://doi.org/10.24432/C5WC95). UCI describes actual hourly aggregates from a Turkish gas turbine over 2011–2015, with 36,733 records. The data license is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Raw files are divided by year and documented as chronologically ordered; individual timestamps are not supplied.
- **SML2010:** [UCI 274](https://archive.ics.uci.edu/dataset/274/sml2010), [DOI 10.24432/C5RS3S](https://doi.org/10.24432/C5RS3S). Pablo Romeu-Guallart and Francisco Zamora-Martinez (2014) provide measurements from a monitor installed in a real domotic house. The source reports roughly 40 monitoring days, minute acquisition and 15-minute averaged records, with UTC timestamps. License: CC BY 4.0. The task here reconstructs two simultaneous indoor temperatures; it is not future forecasting.

SARCOS was considered because the [official GPML data page](https://gaussianprocess.org/gpml/data/) describes 21 inputs, seven robot-joint torques, and official training/test files. That page does not state a dataset license or establish sampling/time order and units. It was therefore excluded from this round instead of treating a third-party software license as permission for the data. Neither selected replacement is a simulation dataset.

## Frozen feature and target rules

Arrays retain published numeric values. Preparation performs no standardization, target transform, interpolation, outlier trimming, model fit or evaluation. Any later normalization must use training data only. Physical labels below do not imply that all columns have interchangeable units.

### Gas turbine

Nine inputs, in order: `AT`, `AP`, `AH`, `AFDP`, `GTEP`, `TIT`, `TAT`, `TEY`, `CDP`. Outputs: `CO`, `NOX`. Year is split metadata only. Neither output enters the input matrix.

Units as stated by UCI: AT/TIT/TAT are degrees Celsius; AP/AFDP/GTEP/CDP are labelled mbar; AH is percent; TEY is labelled MWH; both emissions are mg/m³. The script preserves source values and these labels without inferring corrections to potentially unusual labels. These are concurrent plant-state measurements, so this is emissions estimation from the same recorded interval, not forecasting or a causal model.

Read `gt_2011.csv` through `gt_2015.csv` in year order, preserving source row order within each file. Drop a record only if any of the eleven chosen values is missing, nonnumeric or nonfinite. UCI reports no missing data. Do not inspect output magnitudes to select rows and do not remove zeros. Source headers, ZIP bytes and individual CSV members are hash-checked.

### SML2010

Outputs are the published indoor dining-room and room temperatures (source columns 3 and 4), both in degrees Celsius. Predictors are exactly source columns 5–23: weather-forecast temperature; both CO₂, humidity and lighting readings; rain proportion; dusk indicator; wind; three facade light measurements; solar irradiance; three ventilation/enthalpic-motor controls; outdoor temperature and humidity. Date, time and computed weekday are excluded from features. These nineteen recorded covariates do not include either output or transformations/lags of either output. Weather-forecast temperature is a separately supplied covariate; its exact issuance history is undocumented, and the benchmark does not claim a forecast-vintage guarantee.

Use the source's UTC date/time to concatenate both files chronologically. For a repeated timestamp keep the first source-file occurrence in lexicographic filename/physical-row order, before complete-case filtering. Discard invalid timestamps and rows with missing, nonnumeric or nonfinite values in any chosen predictor or target. No undocumented numeric sentinel is invented, and unused-column missingness does not remove a row. No smoothing beyond the supplied 15-minute averaging is added. Header descriptions define feature units; the dusk field has no explicit physical unit and is labelled accordingly.

## Fixed split and bootstrap groups

- **Gas turbine:** 2011–2013 training, 2014 validation, 2015 sealed test. This uses a different validation role for 2014 than the original paper's combined 2014–2015 testing recommendation; results must not be described as reproducing that original evaluation. Bootstrap groups are nonoverlapping blocks of 24 consecutive original source records within each year, created before missing-row removal. They are **record blocks, not proven calendar days**, because timestamps and possible gaps are unavailable. No records are shuffled.
- **SML2010:** sort unique eligible UTC dates, with cut positions `floor(0.6*number_of_dates)` and `floor(0.8*number_of_dates)`. Assign whole dates to training, validation and test in that order; preserve chronological records within each split. Each UTC date is a bootstrap group, and no date crosses split boundaries. The percentages refer to observed dates, not necessarily equal row counts. Gaps in recordings remain gaps; no extra temporal purge is introduced.

Serial dependence, single-site observations, nonstationarity and incomplete knowledge of sensor calibration limit generalization. No independence across adjacent record blocks is asserted.

## Artifacts and reproducibility

`new_data/{gas_turbine,sml2010}.npz` contains `X_train`, `Y_train`, `X_val`, `Y_val`, `train_ids`, `val_ids`, `train_groups`, and `val_groups`. `outputs/real_blind_v3_sealed/{gas_turbine,sml2010}.npz` contains `X_test`, `Y_test`, `test_ids`, and `test_groups`. Stable IDs use source filename and physical line number. All arrays load with `allow_pickle=False`.

Raw archives are cached under `outputs/real_blind_v3_raw/`. Raw archives and processed NPZ data remain local and must not be committed. `prepare_new_data.py` and `new_data/manifest.json` may be published with this attribution. The MIT software license does not replace the datasets' CC BY 4.0 terms.

`new_data/manifest.json` records source/licensing details, declared units, raw/member/file SHA-256, split row/group counts, ordered ID hashes and training-only input/target summary statistics. It contains no test values or target distribution summaries. Preparation refuses to overwrite incompatible existing artifacts. Rebuild from the repository root with `python experiments/real_blind_v3/prepare_new_data.py`; source files must match their pinned hashes.

## Preparation receipt

Both preparations completed, and a second run reproduced all archive and manifest bytes exactly. An independent check verified archive hashes, NPZ keys, NPY shapes and non-object dtypes, plus train/validation finiteness and row/group separation. For sealed files that check inspected only hashes and NPY headers, without loading numerical test arrays.

| Dataset | Inputs / outputs | Train / validation / sealed test rows | Train / validation / sealed test groups |
| --- | --- | --- | --- |
| Gas turbine | 9 / 2 | 22,191 / 7,158 / 7,384 | 925 / 299 / 308 record blocks |
| SML2010 | 19 / 2 | 2,545 / 795 / 797 | 27 / 9 / 9 whole UTC dates |

All 36,733 gas-turbine rows and all 4,137 SML2010 rows were retained. No missing selected values or invalid timestamps required removal, and no duplicate SML2010 timestamp was found. The SML2010 files span 45 observed UTC dates, consistent with the official approximate-duration description. No test outcome distribution or model performance was computed during preparation.
