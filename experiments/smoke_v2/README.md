# Development regression v2

The v1 run and implementation are frozen in commit `ea64aec`; no v1 measurements
were overwritten. V2 reuses exactly the four systems, splits, seeds and budgets
defined by `../smoke_v1/run.py`, after adding a four-branch forward search and
bounded generic scaled-unary/common-denominator initialization. These cases have
already been inspected. V2 is regression evidence, not blind evaluation.

```bash
python experiments/smoke_v1/run.py --output outputs/my_smoke_v2
```

The shared runner still identifies its dataset protocol as smoke_v1. Source
hashes in `results/results.json` identify the actual v2 algorithm. Every saved
model has an exact JSON prediction roundtrip; test results and failed recoveries
are reported without selection. No comparisons with other systems were run.
