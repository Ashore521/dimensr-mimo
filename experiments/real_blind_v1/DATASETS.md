# Real measured data: frozen preparation protocol

This protocol was written before running any model on these datasets. Preparation uses seed `20261023`; it never trains a model, scores a prediction, prints held-out targets, or computes held-out target summaries. The sealed data are for one final evaluation after selecting the algorithm using training and validation only.

## Sources and attribution

| Dataset | Official source | Original measurements | License shown by UCI |
| --- | --- | --- | --- |
| Concrete Slump Test | [UCI 182](https://archive.ics.uci.edu/dataset/182/concrete+slump+test), [DOI](https://doi.org/10.24432/C5FG7D), I-Cheng Yeh (2007) | Concrete mixtures and laboratory slump, flow and 28-day compressive strength | CC BY 4.0 |
| Parkinsons Telemonitoring | [UCI 189](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring), [DOI](https://doi.org/10.24432/C5ZS3N), Athanasios Tsanas and Max Little (2009) | Home voice recordings from 42 people enrolled in a six-month study | CC BY 4.0 |
| Air Quality | [UCI 360](https://archive.ics.uci.edu/dataset/360/air+quality), [DOI](https://doi.org/10.24432/C59K5F), Saverio De Vito (2008) | Field sensor responses and colocated reference-analyzer concentrations in an Italian city | CC BY 4.0 banner, conflicting legacy research-only restriction |

Official source pages were checked on 2026-10-23. Air Quality's description separately excludes commercial use, despite its CC BY 4.0 banner. This experiment uses it for local research only; the conflict is not resolved here. Raw data, processed numeric datasets and sealed test arrays are excluded from publication. Public code downloads the official sources and records their hashes. Appropriate data attribution remains necessary independently of this repository's MIT code license.

## Fixed columns and cleaning

All arrays retain published numeric values; this script performs no standardization, target transformation, interpolation, clipping or imputation. Physical unit labels document source meaning and are not a declaration that every feature has a known SI dimension. Any model-side normalization must be fit on training data only.

- **Concrete Slump Test:** seven inputs are Cement, Slag, Fly ash, Water, SP, Coarse aggregate and Fine aggregate, in kg per cubic metre of concrete. Outputs are SLUMP (cm), FLOW (cm) and 28-day Compressive Strength (MPa). Exclude the record-number column. Require finite values in all ten selected columns; drop a row only if any selected field is missing or nonnumeric. No response-based outlier removal. The source contains 103 mixtures, comprising an initial 78 and 25 obtained later. Random splitting is a mixture-level generalization check and does not establish independence of production batches.
- **Parkinsons Telemonitoring:** use exactly the 16 published voice measures (`Jitter(%)`, `Jitter(Abs)`, `Jitter:RAP`, `Jitter:PPQ5`, `Jitter:DDP`, `Shimmer`, `Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `Shimmer:APQ11`, `Shimmer:DDA`, `NHR`, `HNR`, `RPDE`, `DFA`, `PPE`). Outputs are `motor_UPDRS` and `total_UPDRS`. Subject ID is used only for grouping; age, sex, recruitment time and both output columns are excluded from predictors. Require a valid integer subject ID and finite selected measurements/targets. No imputation or response-based filtering. The UPDRS labels are **linearly interpolated clinician scores**, so recordings are real measurements but their per-recording target labels are not independent clinical observations. The two targets are also related clinical scales. The source does not specify a complete authoritative unit mapping for all 16 measures; names containing `%` or `dB` preserve that notation, and the rest are documented as unspecified voice-measure units.
- **Air Quality:** inputs are `PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`, `PT08.S5(O3)`, `T`, `RH`, `AH`. Outputs are reference-analyzer `CO(GT)` (mg/m³), `NOx(GT)` (ppb), `NO2(GT)` (µg/m³). Other ground-truth pollutant columns are excluded from inputs. Parse `Date`/`Time` as local source timestamps; discard empty or malformed timestamps, sort chronologically, and retain the first source-file row for a repeated timestamp **before** complete-case filtering. Discard a row if any of these eleven selected numeric fields is missing, nonfinite or the source sentinel `-200`. Missingness of unused fields is irrelevant. No interpolation or response-based outlier removal. `T` is °C and `RH` is percent; the source does not unambiguously establish SI units for sensor responses or `AH`. Sensor drift, cross-sensitivity, serial correlation and complete-case selection limit interpretation. Timestamp ordering treats the source as naive local time; no timezone or daylight-saving reconstruction is attempted.

## Fixed splits and sealing

For a population of size `N`, cuts are `floor(0.6*N)` and `floor(0.8*N)`. The first segment is training, the second validation and the third test. Integer rounding can slightly alter percentages.

1. Concrete: independently initialize NumPy `default_rng(20261023)` and permute eligible source-row indices, then apply the cuts.
2. Parkinsons: sort eligible unique subject IDs, independently initialize the same RNG seed, permute subjects, then apply the cuts to subjects. Preserve source order within each split. No subject can occur in more than one split.
3. Air Quality: apply the cuts to the cleaned chronological records, with no shuffling. Distinct timestamps cannot cross split boundaries. No time-gap purge is imposed; adjacent boundaries and repeated site measurements remain a limitation.

`data/{concrete_slump,parkinsons,air_quality}.npz` contains `X_train`, `Y_train`, `X_val`, `Y_val`, `train_ids`, `val_ids`, `train_groups` and `val_groups`. `outputs/real_blind_v1_sealed/` contains corresponding NPZ files with `X_test`, `Y_test`, `test_ids` and `test_groups`. Groups for later bootstrap resampling are individual source rows for concrete, subject IDs for Parkinsons, and date strings for Air Quality. Air Quality splits follow hourly records, so a boundary day may occur in two adjacent splits. All arrays load with `allow_pickle=False`. Source row IDs are stable and retained through filtering and splitting.

`data/manifest.json` records source and raw-member SHA-256, column metadata, eligible/dropped counts, split counts, ordered split-ID hashes, array-file SHA-256 and **training-only** input ranges. It contains no held-out values, means, variances, extrema or performance. Local raw downloads live under `outputs/real_blind_v1_raw/`.

Rebuild from the repository root using `python experiments/real_blind_v1/prepare_data.py`. Existing raw files are reused only if their pinned hashes match. Existing prepared arrays and manifest must match deterministic contents; incompatible files are not silently overwritten. Preparation asserts disjoint splits, source provenance, finite arrays and group/time isolation without displaying target values.

## Preparation receipt

Preparation and a deterministic second run completed successfully using the pinned official ZIP files. Train/validation archives were independently checked with `allow_pickle=False` for schema, finite values, disjoint row IDs and manifest hashes; that check did not open the sealed archives.

| Dataset | Eligible rows | Train / validation / sealed test rows | Split groups |
| --- | ---: | --- | --- |
| Concrete Slump Test | 103 | 61 / 21 / 21 | One source row per group |
| Parkinsons Telemonitoring | 5,875 | 3,514 / 1,019 / 1,342 | 25 / 8 / 9 subjects |
| Air Quality | 6,941 | 4,164 / 1,388 / 1,389 | Dates retained for group bootstrap |

Concrete and Parkinsons required no row deletion. The Air Quality CSV has 9,471 physical data records: 114 have invalid/empty timestamps, no timestamp duplicates were found, and 2,416 remaining records fail the predeclared selected-column complete-case rule. Its 9,357 valid-timestamp records differ by one from the UCI page's stated 9,358 instances; the pinned CSV, not that descriptive count, controls this experiment. No test outcome statistics or model performance were computed during preparation.
