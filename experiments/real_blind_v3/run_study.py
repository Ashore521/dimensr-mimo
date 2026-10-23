"""Immutable v3 development/new-data batches; this module never opens holdouts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CV_DATASETS = ("concrete_slump", "parkinsons", "air_quality")
NEW_DATASETS = ("gas_turbine", "sml2010")
SYMBOLIC = ("joint", "independent", "coverage", "prune")
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
TIMEOUT = 120
BUDGET = 20.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Publish a completed record atomically, never replacing an existing record."""
    path = Path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
    temporary.replace(path)


def provenance(source):
    package = source / "src/dimensr_mimo"
    if not (package / "__init__.py").is_file():
        raise ValueError("--source must contain src/dimensr_mimo/__init__.py")
    return dict(
        source_directory=str(source),
        source_sha256={p.relative_to(source / "src").as_posix(): sha(p)
                       for p in sorted(package.rglob("*.py"))},
        study_sha256={name: sha(HERE / name) for name in ("run_study.py", "runner.py", "PROTOCOL.md")},
        python=platform.python_version(), executable=sys.executable, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "sympy", "scikit-learn", "threadpoolctl", "joblib")},
        thread_environment=THREAD_ENV, search_cpu_seconds=BUDGET, wall_timeout_seconds=TIMEOUT,
    )


def load_runtime(source):
    # Insert the frozen package before importing either runner or dimensr_mimo.
    sys.path.insert(0, str(source / "src"))
    import dimensr_mimo
    import runner
    if Path(dimensr_mimo.__file__).resolve().parent != (source / "src/dimensr_mimo").resolve():
        raise RuntimeError("Imported symbolic package does not match --source.")
    return runner


def run_name(job):
    fold = f"_f{job['fold']}" if job["fold"] is not None else ""
    return f"{job['dataset']}_{job['method']}{fold}_s{job['seed']}"


def jobs_for(mode, methods=None, selected=None):
    if mode == "cv":
        return [dict(dataset=dataset, method=method, fold=fold, seed=seed)
                for dataset in CV_DATASETS for method in dict.fromkeys(methods or SYMBOLIC)
                for fold in range(3) for seed in range(5)]
    methods = list(dict.fromkeys(("joint", "independent", selected)))
    jobs = [dict(dataset=dataset, method=method, fold=None, seed=seed)
            for dataset in NEW_DATASETS for method in methods for seed in range(5)]
    jobs += [dict(dataset=dataset, method=method, fold=None, seed=seed)
             for dataset in NEW_DATASETS for method in ("mean", "ridge", "extra_trees")
             for seed in (range(5) if method == "extra_trees" else (0,))]
    return jobs


def baseline_run(runner, split, method, seed):
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits

    if method == "mean":
        model = split["Y"].mean(axis=0)
    elif method == "ridge":
        model = Ridge(alpha=1.)
    elif method == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=seed)
    else:
        raise ValueError(f"Unknown baseline: {method}")
    start_wall, start_cpu = time.monotonic(), time.process_time()
    with threadpool_limits(limits=1):
        if method != "mean":
            model.fit(split["X"], split["Y"] / split["target_scale"])
        report = dict(split["metadata"], method=method, seed=seed, status="ok",
                      fit_cpu_seconds=time.process_time() - start_cpu,
                      fit_wall_seconds=time.monotonic() - start_wall,
                      configs=[dict(strategy="mean") if method == "mean" else model.get_params()],
                      diagnostics=[], complexity=None,
                      data_sha256=split["metadata"]["source_sha256"],
                      target_transform="Mean stored in original units; other baselines fit Y/training target scale.",
                      budget_note="Predictive accuracy baseline; no equal-search-budget claim.")
        for name, suffix in (("train", ""), ("validation", "_val"), ("eval", "_eval")):
            if "X" + suffix in split:
                prediction = (np.tile(model, (len(split["X" + suffix]), 1)) if method == "mean"
                              else np.asarray(model.predict(split["X" + suffix])) * split["target_scale"])
                if prediction.ndim == 1:
                    prediction = prediction[:, None]
                report[name] = runner.metrics(split["Y" + suffix], prediction, split["target_scale"])
    return report, model


def execute_job(job, mode, directory, source):
    """Child process entry point; the parent has exclusively created its directory."""
    started = json.loads((directory / "started.json").read_text(encoding="utf-8"))
    if any(started[key] != value for key, value in dict(job, mode=mode).items()):
        raise ValueError("Child job differs from the immutable started record.")
    if any(path.name not in ("started.json", "process.log") for path in directory.iterdir()):
        raise FileExistsError(f"Child refuses to overwrite existing artifacts: {directory}")
    report = dict(job, status="failed", models=[])
    start_wall, start_cpu = time.monotonic(), time.process_time()
    try:
        import numpy as np
        runner = load_runtime(source)
        if mode == "cv":
            split = runner.prepare_split(job["dataset"], job["fold"])
        else:
            metadata = json.loads((HERE / "new_data/manifest.json").read_text(encoding="utf-8"))["datasets"][job["dataset"]]
            split = runner.prepare_trainval(HERE / "new_data" / f"{job['dataset']}.npz",
                                            metadata, dataset=job["dataset"])
        if job["method"] in SYMBOLIC:
            report, models = runner.train_run(split, job["method"], job["seed"], budget=BUDGET)
            report["models"] = []
            for index, model in enumerate(models):
                filename = f"model_{index}.json"
                model.save(directory / filename)
                report["models"].append(filename)
                restored = runner.MIMOSymbolicRegressor.load(directory / filename)
                for key in ("X", "X_val", "X_eval"):
                    if key in split:
                        np.testing.assert_array_equal(restored.predict(split[key]), model.predict(split[key]))
            report["reload_prediction_exact"] = report["status"] == "ok"
        else:
            import joblib
            report, model = baseline_run(runner, split, job["method"], job["seed"])
            joblib.dump(model, directory / "model.joblib")
            report["models"] = ["model.joblib"]
            restored = joblib.load(directory / "model.joblib")
            if job["method"] == "mean":
                np.testing.assert_array_equal(restored, model)
            else:
                np.testing.assert_array_equal(restored.predict(split["X_val"]), model.predict(split["X_val"]))
            report["reload_prediction_exact"] = True
        report["model_sha256"] = {name: sha(directory / name) for name in report["models"]}
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    report.update(job, mode=mode, source_sha256=started["source_sha256"],
                  provenance=started, process_cpu_seconds=time.process_time() - start_cpu,
                  process_wall_seconds=time.monotonic() - start_wall)
    write_json(directory / "child_result.json", report)
    return 0 if report["status"] == "ok" else 1


def existing_status(directory, expected):
    if not directory.exists():
        return None
    record = directory / "timeout.json"
    if not record.exists():
        record = directory / "result.json"
    if not record.exists():
        raise FileExistsError(f"Incomplete run cannot be overwritten: {directory}")
    actual = json.loads(record.read_text(encoding="utf-8"))
    started = json.loads((directory / "started.json").read_text(encoding="utf-8"))
    for key in ("mode", "dataset", "method", "fold", "seed", "source_sha256", "study_sha256", "selection"):
        if started.get(key) != expected.get(key):
            raise ValueError(f"Existing run has incompatible {key}: {directory}")
    if actual.get("status") not in ("ok", "failed", "timeout"):
        raise ValueError(f"Existing record is not terminal: {record}")
    return actual["status"]


def launch_job(job, args, shared):
    directory = args.output / run_name(job)
    directory.mkdir()
    started = dict(shared, **job, mode=args.mode, started_utc=datetime.now(timezone.utc).isoformat())
    write_json(directory / "started.json", started)
    command = [sys.executable, str(Path(__file__).resolve()), args.mode,
               "--output", str(args.output), "--source", str(args.source),
               "--job", job["dataset"], job["method"], str(job["fold"] if job["fold"] is not None else -1), str(job["seed"])]
    if args.selection:
        command += ["--selection", str(args.selection)]
    environment = dict(os.environ, **THREAD_ENV, PYTHONHASHSEED="0", PYTHONUTF8="1")
    with (directory / "process.log").open("x", encoding="utf-8") as log:
        try:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, timeout=TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            write_json(directory / "timeout.json", dict(job, mode=args.mode, status="timeout",
                       timeout_seconds=TIMEOUT, error="Child exceeded its fixed wall-time limit.", provenance=started))
            return "timeout"
        except OSError as exc:
            write_json(directory / "result.json", dict(job, mode=args.mode, status="failed", models=[],
                       error=f"Cannot launch child: {type(exc).__name__}: {exc}", provenance=started))
            return "failed"
    child_path = directory / "child_result.json"
    if child_path.exists():
        result = json.loads(child_path.read_text(encoding="utf-8"))
    else:
        result = dict(job, mode=args.mode, status="failed", models=[],
                      error="Child exited without a completed record.", provenance=started)
    if completed.returncode:
        result.update(status="failed", returncode=completed.returncode,
                      process_error=f"Child exited with code {completed.returncode}.")
    write_json(directory / "result.json", result)
    return result["status"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cv", "new"))
    parser.add_argument("--output", type=Path, required=True, help="Root for immutable run directories.")
    parser.add_argument("--source", type=Path, default=ROOT, help="Repository/snapshot containing src/dimensr_mimo.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=SYMBOLIC, help="CV arms; omitted means all four.")
    parser.add_argument("--selection", type=Path, help="Required for new: frozen CV summary with selected_method.")
    parser.add_argument("--job", nargs=4, metavar=("DATASET", "METHOD", "FOLD", "SEED"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive.")
    args.output, args.source = args.output.resolve(), args.source.resolve()
    if args.mode == "new" and (args.selection is None or args.methods is not None):
        parser.error("new requires --selection and does not accept --methods.")
    selection = None
    if args.selection:
        args.selection = args.selection.resolve()
        summary = json.loads(args.selection.read_text(encoding="utf-8"))
        if summary.get("selection_status") != "selected" or summary.get("protocol_errors"):
            parser.error("--selection must be a completed CV selection without protocol errors.")
        selected = summary["selected_method"]
        if selected not in ("joint", "coverage", "prune"):
            parser.error("selected_method must be joint, coverage or prune.")
        selection = dict(path=str(args.selection), sha256=sha(args.selection), selected_method=selected)
    jobs = jobs_for(args.mode, args.methods, selection["selected_method"] if selection else None)
    if args.job:
        dataset, method, fold, seed = args.job
        job = dict(dataset=dataset, method=method, fold=None if int(fold) == -1 else int(fold), seed=int(seed))
        if job not in jobs:
            parser.error("Child job is outside the declared study grid.")
        return execute_job(job, args.mode, args.output / run_name(job), args.source)
    shared = dict(provenance(args.source), selection=selection)
    pending, statuses = [], []
    for job in jobs:
        status = existing_status(args.output / run_name(job), dict(shared, **job, mode=args.mode))
        if status is None:
            pending.append(job)
        else:
            statuses.append(status)
            print(f"skip {run_name(job)}: {status}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(launch_job, job, args, shared): job for job in pending}
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            print(f"done {run_name(futures[future])}: {status}", flush=True)
    print(f"finished {len(statuses)} runs; failures={sum(status != 'ok' for status in statuses)}", flush=True)
    return int(any(status != "ok" for status in statuses))


if __name__ == "__main__":
    raise SystemExit(main())
