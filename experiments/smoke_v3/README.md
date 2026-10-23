# Development regression v3

V2 is frozen at commit `ee61b35`. V3 changes only nonlinear polishing: up to four
constant-bearing models on each generation's training Pareto front are optimized,
instead of only the deepest selected dictionary. This allows a simpler nonlinear
model to compete with a higher-order approximation after its constants are fitted.

The cases and protocol are reused from smoke_v1. All test rows have already been
inspected; these are regression results and cannot be treated as blind evidence.

```bash
python experiments/smoke_v1/run.py --output outputs/my_smoke_v3
```

See `results/results.json` for every run, source hashes, configurations, package
versions, and numerical recovery flags. The runner's `protocol=smoke_v1` records
the unchanged dataset/split protocol, not an assertion of unchanged algorithm.
