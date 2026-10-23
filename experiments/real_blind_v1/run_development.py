"""One immutable training/validation run; deliberately never opens test files."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = dict(max_features=192, generations=3, proposals_per_generation=128,
                   max_terms=8, max_nodes=13, max_depth=6, polish_max_nfev=10,
                   interval_max_boxes=32, complexity_weight=1e-4)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, prediction, scales):
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    rmse = np.sqrt(np.mean((prediction-y)**2, axis=0))
    return dict(rmse=rmse.tolist(), nrmse=(rmse/scales).tolist(),
                mean_nrmse=float(np.mean(rmse/scales)),
                r2=np.asarray(r2_score(y, prediction, multioutput="raw_values")).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["concrete_slump", "parkinsons", "air_quality"])
    p.add_argument("--method", default="symbolic", choices=["symbolic", "ridge", "extra_trees", "mean"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", type=Path, default=ROOT)
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve() / "src"
    sys.path.insert(0, str(source))
    from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig

    data_path = ROOT / "experiments/real_blind_v1/data" / (args.dataset + ".npz")
    data = np.load(data_path, allow_pickle=False)
    X, Y = data["X_train"], data["Y_train"]
    indices = np.arange(len(X))
    if len(X) > 1200:
        indices = np.sort(np.random.default_rng(20261023).choice(len(X), 1200, replace=False))
    X, Y = X[indices], Y[indices]
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    X, X_val, Y_val = (X-mean)/scale, (data["X_val"]-mean)/scale, data["Y_val"]
    yscale = np.std(Y, axis=0)
    yscale = np.where(yscale > 0, yscale, np.maximum(np.max(np.abs(Y), axis=0), 1.))
    config = dict(BASE_CONFIG, random_state=args.seed)
    if args.config:
        config.update(json.loads(args.config.read_text(encoding="utf-8")))
    metadata = json.loads((data_path.parent / "manifest.json").read_text(encoding="utf-8"))["datasets"][args.dataset]
    report = dict(dataset=args.dataset, method=args.method, seed=args.seed,
                  data_sha256=sha(data_path), config=config, train_rows=len(X),
                  validation_rows=len(X_val), train_indices=indices.tolist(),
                  input_mean=mean.tolist(), input_scale=scale.tolist(), target_scale=yscale.tolist(),
                  python=platform.python_version(), packages={k: importlib.metadata.version(k) for k in
                    ("numpy", "scipy", "sympy", "scikit-learn", "joblib")},
                  source_sha256={f.relative_to(source).as_posix(): sha(f)
                                 for f in sorted((source / "dimensr_mimo").rglob("*.py"))})
    (args.output / "started.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    start = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            if args.method == "symbolic":
                model = MIMOSymbolicRegressor(SearchConfig(**config),
                    feature_names=metadata["feature_names"], target_names=metadata["target_names"])
                model.fit(X, Y, X_val=X_val, Y_val=Y_val)
                model.save(args.output / "model.json")
                report.update(complexity=model.model_["complexity"], expressions=[str(e) for e in model.expressions_],
                              diagnostics=model.diagnostics_, history=model.history_,
                              domain=model.audit_domain()["status"])
            elif args.method == "ridge":
                model = Ridge(alpha=1.).fit(X, Y/yscale)
            elif args.method == "extra_trees":
                model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                    n_jobs=1, random_state=args.seed).fit(X, Y/yscale)
            else:
                model = None
            if args.method != "symbolic":
                joblib.dump(model if model is not None else Y.mean(axis=0), args.output / "model.joblib")
            report["fit_seconds"] = time.monotonic()-start
            train_pred = model.predict(X) if model is not None else np.tile(Y.mean(axis=0), (len(X), 1))
            val_pred = model.predict(X_val) if model is not None else np.tile(Y.mean(axis=0), (len(X_val), 1))
            if args.method in ("ridge", "extra_trees"):
                train_pred, val_pred = train_pred*yscale, val_pred*yscale
            report.update(status="ok", train=metrics(Y, train_pred, yscale),
                          validation=metrics(Y_val, val_pred, yscale))
            if args.method == "symbolic":
                np.testing.assert_array_equal(MIMOSymbolicRegressor.load(args.output / "model.json").predict(X_val), val_pred)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-start)
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(args.dataset, args.method, args.seed, report["status"], report.get("validation", {}).get("mean_nrmse"), flush=True)
    if report["status"] != "ok":
        raise SystemExit(report["error"])


if __name__ == "__main__":
    main()
