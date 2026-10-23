"""Freeze reviewed artifacts, then open new holdouts exactly once."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from runner import ROOT, metrics, sha
from dimensr_mimo import MIMOSymbolicRegressor

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def paired_interval(y, before, after, scales, groups):
    """Resample observed groups; average errors over seed models, not ensembles."""
    labels, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)

    def errors(predictions):
        return np.asarray([[np.bincount(inverse, weights=((p[:, j] - y[:, j]) / scales[j]) ** 2,
                                        minlength=len(labels)) for j in range(y.shape[1])]
                           for p in predictions]).transpose(0, 2, 1)

    left, right = errors(before), errors(after)
    rng = np.random.default_rng(20261024)
    differences = []
    for _ in range(2000):
        sampled = rng.integers(len(labels), size=len(labels))
        n = counts[sampled].sum()
        differences.append(float(np.sqrt(right[:, sampled].sum(axis=1) / n).mean()
                                 - np.sqrt(left[:, sampled].sum(axis=1) / n).mean()))
    return dict(groups=len(labels), resamples=2000, after_minus_joint_nrmse_95ci=
                np.quantile(differences, [.025, .975]).tolist(),
                interpretation="Conditional on frozen seed models and observed test groups; not IID rows.")


def freeze(runs, output, selection):
    if output.exists():
        raise FileExistsError(output)
    chosen = json.loads(selection.read_text(encoding="utf-8"))["selected_method"]
    manifest = json.loads((HERE / "new_data/manifest.json").read_text(encoding="utf-8"))
    methods = list(dict.fromkeys(("joint", "independent", chosen)))
    expected = {(d, m, s) for d in ("gas_turbine", "sml2010") for m in methods for s in range(5)}
    expected |= {(d, m, s) for d in ("gas_turbine", "sml2010") for m in ("mean", "ridge", "extra_trees")
                 for s in (range(5) if m == "extra_trees" else (0,))}
    entries, files, actual = [], {}, set()
    for directory in sorted(p for p in runs.iterdir() if p.is_dir()):
        result = directory / ("timeout.json" if (directory / "timeout.json").exists() else "result.json")
        report = json.loads(result.read_text(encoding="utf-8"))
        key = report["dataset"], report["method"], report["seed"]
        if key in actual:
            raise ValueError(f"Duplicate run: {key}")
        actual.add(key)
        names = report.get("models", [])
        if report["status"] == "ok" and not names:
            raise ValueError(f"Missing fitted models: {result}")
        if report["status"] == "ok" and any(sha(directory / name) != report["model_sha256"][name] for name in names):
            raise ValueError(f"Model changed after training: {result}")
        for path in [result] + [result.parent / name for name in names]:
            files[path.resolve().relative_to(ROOT).as_posix()] = sha(path)
        provenance = report["provenance"]
        if provenance["selection"]["sha256"] != sha(selection) or provenance["selection"]["selected_method"] != chosen:
            raise ValueError(f"Selection changed after training: {result}")
        if report.get("data_sha256", manifest["datasets"][key[0]]["trainval_sha256"]) != manifest["datasets"][key[0]]["trainval_sha256"]:
            raise ValueError(f"Training data provenance changed: {result}")
        source = Path(provenance["source_directory"])
        for name, expected_hash in provenance["source_sha256"].items():
            path = source / "src" / name
            if sha(path) != expected_hash or path.read_text(encoding="utf-8") != (ROOT / "src" / name).read_text(encoding="utf-8"):
                raise ValueError(f"Training source changed or differs from evaluation source: {path}")
            files[path.resolve().relative_to(ROOT).as_posix()] = expected_hash
        for name, expected_hash in provenance["study_sha256"].items():
            if sha(HERE / name) != expected_hash:
                raise ValueError(f"Training harness changed: {name}")
        entries.append(dict(dataset=key[0], method=key[1], seed=key[2],
                            directory=result.parent.resolve().relative_to(ROOT).as_posix(),
                            status=report["status"], models=names))
    if actual != expected:
        raise ValueError(f"Run inventory differs: missing={expected-actual}; extra={actual-expected}")
    for path in list((ROOT / "src/dimensr_mimo").rglob("*.py")) + list(HERE.glob("*.py")) + [
            HERE / "PROTOCOL.md", HERE / "NEW_DATASETS.md", HERE / "new_data/manifest.json", selection]:
        files[path.resolve().relative_to(ROOT).as_posix()] = sha(path)
    for item in manifest["datasets"].values():
        for field in ("trainval", "sealed"):
            path = ROOT / item[field + "_file"]
            if sha(path) != item[field + "_sha256"]:
                raise ValueError(f"Data hash mismatch: {path}")
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    write_json(output, dict(selected_method=chosen, files=files, models=entries,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        instruction="Commit this manifest before invoking evaluate. No refits after holdout opening."))
    print(f"Frozen {len(entries)} runs and {len(files)} files")


def evaluate(freeze_path, output):
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    for path, expected in frozen["files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Frozen file changed: {path}")
    relative_freeze = freeze_path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative_freeze], cwd=ROOT)
    if committed != freeze_path.read_bytes():
        raise ValueError("Freeze manifest must be committed, with byte-identical content.")
    # A fixed marker prevents reopening the same study under a different output path.
    if output.exists():
        raise FileExistsError(output)
    with (HERE / "HOLDOUT_OPENED.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(freeze_sha256=sha(freeze_path), output=str(output),
                       opened_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), stream, indent=2)
    output.mkdir(parents=True, exist_ok=False)
    result_path = output / "results.json"
    report = dict(freeze_sha256=sha(freeze_path), runs=[], complete=False,
                  opened_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write_json(result_path, report)
    manifest = json.loads((HERE / "new_data/manifest.json").read_text(encoding="utf-8"))
    tests, predictions = {}, defaultdict(lambda: defaultdict(list))
    scales_by_dataset = {}
    with threadpool_limits(limits=1):
        for entry in frozen["models"]:
            dataset, method = entry["dataset"], entry["method"]
            result = dict(entry)
            if entry["status"] != "ok":
                result.update(status="training_failed")
                report["runs"].append(result)
                write_json(result_path, report)
                continue
            if dataset not in tests:
                with np.load(ROOT / manifest["datasets"][dataset]["sealed_file"], allow_pickle=False) as data:
                    tests[dataset] = {key: data[key] for key in data.files}
            data = tests[dataset]
            directory = ROOT / entry["directory"]
            dev = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            X = (data["X_test"] - dev["input_mean"]) / dev["input_scale"]
            Y, scales = data["Y_test"], np.asarray(dev["target_scale"])
            if dataset in scales_by_dataset:
                np.testing.assert_array_equal(scales_by_dataset[dataset], scales)
            scales_by_dataset[dataset] = scales
            result.update(test_rows=len(X), complexity=dev.get("complexity"),
                          search_cpu_seconds=dev.get("search_cpu_seconds"),
                          fit_cpu_seconds=dev.get("fit_cpu_seconds"), fit_wall_seconds=dev.get("fit_wall_seconds"))
            try:
                start = time.monotonic()
                if method in ("mean", "ridge", "extra_trees"):
                    model = joblib.load(directory / "model.joblib")
                    pred = np.tile(model, (len(X), 1)) if method == "mean" else model.predict(X) * scales
                else:
                    models = [MIMOSymbolicRegressor.load(directory / name) for name in entry["models"]]
                    pred = np.column_stack([model.predict(X) for model in models])
                    lower, upper = models[0].input_bounds_.T
                    result["outside_training_box_rows"] = int(np.any((X < lower) | (X > upper), axis=1).sum())
                    expanded = np.column_stack((np.minimum(lower, X.min(axis=0)), np.maximum(upper, X.max(axis=0))))
                    result["expanded_domain"] = [model.audit_domain(expanded)["status"] for model in models]
                result.update(status="ok", prediction_seconds=time.monotonic() - start,
                              test=metrics(Y, pred, scales))
                predictions[dataset][method].append(pred)
            except Exception as exc:
                result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["runs"].append(result)
            write_json(result_path, report)
            print(dataset, method, entry["seed"], result["status"], result.get("test", {}).get("mean_nrmse"), flush=True)
        report["paired_intervals"] = {}
        for dataset, methods in predictions.items():
            report["paired_intervals"][dataset] = {}
            if len(methods["joint"]) != 5:
                continue
            expected_methods = {e["method"] for e in frozen["models"] if e["dataset"] == dataset} - {"joint"}
            for method in sorted(expected_methods):
                values = methods[method]
                required = 1 if method in ("mean", "ridge") else 5
                if len(values) == required:
                    report["paired_intervals"][dataset][method] = paired_interval(
                        tests[dataset]["Y_test"], methods["joint"], values,
                        scales_by_dataset[dataset], tests[dataset]["test_groups"])
                else:
                    report["paired_intervals"][dataset][method] = dict(status="unavailable", successful_models=len(values), required_models=required)
        report["complete"] = True
        write_json(result_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("freeze")
    first.add_argument("--runs", type=Path, required=True)
    first.add_argument("--selection", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    second = commands.add_parser("evaluate")
    second.add_argument("--freeze", type=Path, required=True)
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.runs, args.output, args.selection)
    else:
        evaluate(args.freeze, args.output)


if __name__ == "__main__":
    main()
