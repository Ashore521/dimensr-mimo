# Development smoke protocol v1

This protocol was written before its first run. Four synthetic systems have three
inputs and two outputs, with common polynomial, trigonometric, rational, or
nonlinearly parameterized terms. Data use RNG seed 20261023, 280 uniform points
in [-1,1]^3: the first 160 train, the next 40 select, and the last 80 score.
Search seeds are 0 and 1. Every run uses the same declared configuration in
`run.py`; all runs, including unsuccessful recoveries, are recorded.

Run from the repository root after installation:

```bash
python experiments/smoke_v1/run.py --output outputs/my_smoke_v1
```

An existing output directory is refused to preserve prior results. Every fit saves
its model and checkpointed results, including source hashes, dependency versions,
test metrics, interval screening, and exact JSON prediction roundtrip. Recovery
means maximum absolute error <1e-6 on these 80 test rows; it is not symbolic
equivalence, an arbitrary-domain proof, or performance on unknown tasks.

These author-created cases are implementation/development regression evidence.
They are not a blind benchmark, a matched-budget comparison with original
DimenSR, or evidence of research novelty. See `results_release/results.json` for the
recorded release run. Future algorithm changes must use a new result directory.
