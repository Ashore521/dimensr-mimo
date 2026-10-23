"""Open the sealed holdouts once, only after all fitted models are frozen."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from run_development import ROOT, metrics, sha


def paired_interval(y, before, after, scales, groups):
    # Group sums retain within-subject/day dependence; no iid-row assumption.
    labels, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    def grouped_errors(predictions):
        errors = np.stack([((p-y)/scales)**2 for p in predictions])
        return np.stack([np.stack([np.bincount(inverse, weights=e[:, j], minlength=len(labels))
                                  for j in range(y.shape[1])], axis=1) for e in errors])
    left, right = grouped_errors(before), grouped_errors(after)
    rng = np.random.default_rng(20261023)
    differences = []
    for _ in range(1000):
        sampled = rng.integers(len(labels), size=len(labels))
        denominator = counts[sampled].sum()
        a = np.sqrt(left[:, sampled].sum(axis=1)/denominator).mean()
        b = np.sqrt(right[:, sampled].sum(axis=1)/denominator).mean()
        differences.append(float(b-a))
    return dict(groups=len(labels), resamples=1000, selected_minus_baseline_nrmse_95ci=
                np.quantile(differences, [.025, .975]).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
    for path, expected in frozen["files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen file changed: {path}")
    # Creation is the irreversible opening marker; retries must retain its failure record.
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(freeze_sha256=sha(args.freeze), opened_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=[])
    output = args.output / "results.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sys.path.insert(0, str(ROOT / "src"))
    from dimensr_mimo import MIMOSymbolicRegressor
    predictions = defaultdict(lambda: defaultdict(list))
    tests = {}
    with threadpool_limits(limits=1):
        for entry in frozen["models"]:
            dataset = entry["dataset"]
            if dataset not in tests:
                path = ROOT / "outputs/real_blind_v1_sealed" / (dataset + ".npz")
                if sha(path) != frozen["sealed_sha256"][dataset]:
                    raise ValueError(f"Sealed data changed: {dataset}")
                with np.load(path, allow_pickle=False) as data:
                    tests[dataset] = {key: data[key] for key in data.files}
            data = tests[dataset]
            directory = ROOT / entry["directory"]
            dev = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            X = (data["X_test"]-dev["input_mean"])/dev["input_scale"]
            Y, scales = data["Y_test"], np.asarray(dev["target_scale"])
            result = dict(entry, test_rows=len(X), fit_seconds=dev["fit_seconds"],
                          complexity=dev.get("complexity"), diagnostics=dev.get("diagnostics"))
            try:
                start = time.monotonic()
                if entry["method"] == "symbolic":
                    model = MIMOSymbolicRegressor.load(directory / "model.json")
                    pred = model.predict(X)
                    result["prediction_seconds"] = time.monotonic()-start
                    bounds = np.column_stack((np.minimum(model.input_bounds_[:, 0], X.min(axis=0)),
                                              np.maximum(model.input_bounds_[:, 1], X.max(axis=0))))
                    result["expanded_domain"] = model.audit_domain(bounds)["status"]
                    result["outside_training_box_rows"] = int(np.any((X < model.input_bounds_[:, 0]) |
                                                                 (X > model.input_bounds_[:, 1]), axis=1).sum())
                else:
                    model = joblib.load(directory / "model.joblib")
                    pred = np.tile(model, (len(X), 1)) if isinstance(model, np.ndarray) else model.predict(X)
                    if entry["method"] in ("ridge", "extra_trees"):
                        pred = pred*scales
                    result["prediction_seconds"] = time.monotonic()-start
                result.update(status="ok", test=metrics(Y, pred, scales))
                predictions[dataset][entry["role"]].append(pred)
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["runs"].append(result)
            output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(dataset, entry["role"], entry["seed"], result["status"], result.get("test", {}).get("mean_nrmse"), flush=True)
        report["paired_intervals"] = {}
        for dataset, roles in predictions.items():
            if len(roles["baseline"]) == len(roles["selected"]) == 2:
                dev_entry = next(e for e in frozen["models"] if e["dataset"] == dataset)
                dev = json.loads((ROOT / dev_entry["directory"] / "result.json").read_text(encoding="utf-8"))
                report["paired_intervals"][dataset] = paired_interval(tests[dataset]["Y_test"],
                    roles["baseline"], roles["selected"], np.asarray(dev["target_scale"]), tests[dataset]["test_groups"])
        report["complete"] = True
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
