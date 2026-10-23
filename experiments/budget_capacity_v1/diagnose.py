"""Immutable paired budget/capacity runs; imports frozen v3 splitting code."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from curves import validation_frontier, predict_tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("frozen_v3_runner", ROOT / "experiments/real_blind_v3/runner.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
DATASETS = ("concrete_slump", "parkinsons", "air_quality")
ARMS = ("joint8", "independent8", "joint_wide")
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def name(job):
    d, arm, budget, fold, seed = job
    return f"{d}_{arm}_b{budget}_f{fold}_s{seed}"


def jobs():
    # Interleave both time limits and representations within paired fold/seed cells.
    return [(d, a, b, f, s) for d in DATASETS for f in range(3) for s in (0, 1)
            for b in (20, 60) for a in ARMS]


def check_freeze():
    frozen = json.loads((HERE / "FREEZE.json").read_text(encoding="utf-8"))
    for path, digest in frozen["files"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen diagnostic input changed: {path}")
    return frozen


def prepare():
    if (HERE / "FREEZE.json").exists():
        raise FileExistsError("This diagnostic is already frozen.")
    splits = {}
    for dataset in DATASETS:
        for fold in range(3):
            split = v3.prepare_split(dataset, fold)
            previous = json.loads((ROOT / f"experiments/real_blind_v3/cv/{dataset}_joint_f{fold}_s0/result.json").read_text(encoding="utf-8"))
            assert split["metadata"]["fold_sha256"] == previous["fold_sha256"]
            splits[f"{dataset}_f{fold}"] = split["metadata"]
    save(HERE / "SPLITS.json", splits)
    paths = list((ROOT / "src/dimensr_mimo").rglob("*.py")) + [
        HERE / "diagnose.py", HERE / "curves.py", HERE / "PROTOCOL.md", HERE / "SPLITS.json",
        ROOT / "experiments/real_blind_v3/runner.py", ROOT / "experiments/real_blind_v1/data/manifest.json"]
    paths += [ROOT / f"experiments/real_blind_v1/data/{dataset}.npz" for dataset in DATASETS]
    save(HERE / "FREEZE.json", dict(files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        jobs=[name(job) for job in jobs()], created_at_utc=datetime.now(timezone.utc).isoformat(),
        packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "sympy", "scikit-learn", "threadpoolctl")}))
    print("Frozen 108 jobs and 9 identical v3 splits.")


def fit_job(job, output):
    check_freeze()
    dataset, arm, budget, fold, seed = job
    split = v3.prepare_split(dataset, fold)
    frozen_split = json.loads((HERE / "SPLITS.json").read_text(encoding="utf-8"))[f"{dataset}_f{fold}"]
    assert split["metadata"]["fold_sha256"] == frozen_split["fold_sha256"]
    m = split["Y"].shape[1]
    count = m if arm == "independent8" else 1
    cfg_base = dict(v3.BASE_CONFIG, random_state=seed, search_seconds=budget / count,
                    max_terms=8 * m if arm == "joint_wide" else 8)
    models, archives = [], []
    report = dict(dataset=dataset, arm=arm, budget=budget, fold=fold, seed=seed,
                  fold_sha256=frozen_split["fold_sha256"], freeze_sha256=sha(HERE / "FREEZE.json"),
                  status="running", configs=[], diagnostics=[], models=[])
    cpu, wall = time.process_time(), time.monotonic()
    with threadpool_limits(limits=1):
        for j in range(count):
            columns = slice(j, j + 1) if count > 1 else slice(None)
            model = MIMOSymbolicRegressor(SearchConfig(**cfg_base), feature_names=split["feature_names"],
                                          target_names=split["target_names"][columns])
            model.fit(split["X"], split["Y"][:, columns], X_val=split["X_val"], Y_val=split["Y_val"][:, columns])
            filename = f"model_{j}.json"
            model.save(output / filename)
            restored = MIMOSymbolicRegressor.load(output / filename)
            np.testing.assert_array_equal(restored.predict(split["X_val"]), model.predict(split["X_val"]))
            models.append(model)
            archives.append(model.pareto_front_)
            report["configs"].append(asdict(model.config_))
            report["diagnostics"].append(model.diagnostics_)
            report["models"].append(filename)
        report.update(fit_cpu_seconds=time.process_time()-cpu, fit_wall_seconds=time.monotonic()-wall,
                      search_cpu_seconds=sum(model.diagnostics_["search_cpu_seconds"] for model in models),
                      overshoot_seconds=sum(model.diagnostics_["budget_overshoot_seconds"] for model in models))
        save(output / "archives.json", archives)
        start = time.process_time()
        frontier, combinations = validation_frontier(archives)
        report["curve_selection_cpu_seconds"] = time.process_time()-start
        report["archive_counts"] = [len(a) for a in archives]
        report["enumerated_combinations"] = combinations
        report["default"] = dict(v3.model_complexity(models),
            selected_terms=[len(model.features_) for model in models],
            training_archive_max_terms=[max(len(c["features"]) for c in model.training_archive_) for model in models],
            validation_loss=float(np.mean([model.diagnostics_["selected_validation_loss"] for model in models])))
        # All curve indices are fixed above, before any outer-score computation.
        start = time.process_time()
        for field, suffix in (("train", ""), ("inner", "_val"), ("outer", "_eval")):
            pred = np.column_stack([model.predict(split["X"+suffix]) for model in models])
            report["default"][field] = v3.metrics(split["Y"+suffix], pred, split["target_scale"])
        for point in frontier:
            try:
                for field, suffix in (("train", ""), ("inner", "_val"), ("outer", "_eval")):
                    pred = predict_tuple(archives, point["indices"], split["X"+suffix])
                    point[field] = v3.metrics(split["Y"+suffix], pred, split["target_scale"])
                point["status"] = "ok"
            except ValueError as exc:
                point.update(status="failed", error=str(exc))
        report.update(frontier=frontier, score_cpu_seconds=time.process_time()-start,
                      status="ok", training_domain="enclosed", model_sha256={p:sha(output/p) for p in report["models"]},
                      archives_sha256=sha(output/"archives.json"))
    report.update(total_cpu_seconds=time.process_time()-cpu, total_wall_seconds=time.monotonic()-wall)
    return report


def launch(job):
    output = HERE / "runs" / name(job)
    if output.exists():
        path = output / "result.json"
        if not path.exists():
            raise FileExistsError(f"Incomplete run preserved: {output}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["freeze_sha256"] != sha(HERE / "FREEZE.json"):
            raise ValueError("Existing run has a different freeze.")
        return result["status"]
    output.mkdir(parents=True)
    save(output / "started.json", dict(job=list(job), freeze_sha256=sha(HERE/"FREEZE.json"),
                                       time_utc=datetime.now(timezone.utc).isoformat()))
    with (output / "process.log").open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "job", *map(str, job)],
                stdout=log, stderr=subprocess.STDOUT, timeout=300,
                env=dict(os.environ, **THREAD_ENV, PYTHONHASHSEED="0", PYTHONUTF8="1"))
            child = output / "child.json"
            report = json.loads(child.read_text(encoding="utf-8")) if child.exists() else dict(status="failed", error="No completed child record")
            if result.returncode:
                report.update(status="failed", returncode=result.returncode)
        except subprocess.TimeoutExpired:
            report = dict(status="timeout", error="300-second wall timeout")
        except OSError as exc:
            report = dict(status="failed", error=f"Could not launch process: {exc}")
    report.update(dataset=job[0], arm=job[1], budget=job[2], fold=job[3], seed=job[4], freeze_sha256=sha(HERE/"FREEZE.json"))
    save(output / "result.json", report)
    return report["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "batch", "job"))
    parser.add_argument("identity", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "job":
        d, a, b, f, s = args.identity
        job = (d, a, int(b), int(f), int(s))
        assert job in jobs()
        output = HERE / "runs" / name(job)
        try:
            report = fit_job(job, output)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            report = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(output / "child.json", report)
        return int(report["status"] != "ok")
    else:
        check_freeze()
        if args.workers < 1:
            parser.error("workers must be positive")
        statuses = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(launch, job): job for job in jobs()}
            for future in as_completed(futures):
                status = future.result()
                statuses.append(status)
                print(name(futures[future]), status, flush=True)
        print(f"Finished {len(statuses)} runs; failures={sum(s!='ok' for s in statuses)}", flush=True)
        return int(any(s != "ok" for s in statuses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
