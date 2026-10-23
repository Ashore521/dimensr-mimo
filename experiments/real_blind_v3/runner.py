"""Single-run v3 fitting and nested development splits; never opens holdouts."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import dimensr_mimo
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split, TimeSeriesSplit
from threadpoolctl import threadpool_limits

from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo.features import graph_size


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments/real_blind_v1/data"
SPLIT_SEED = 20261024
TRAIN_CAP = 1200
METHODS = ("joint", "independent", "coverage", "prune")
BASE_CONFIG = dict(max_features=192, generations=10000, proposals_per_generation=128,
                   max_terms=8, max_nodes=13, max_depth=6, polish_max_nfev=10,
                   interval_max_boxes=32, complexity_weight=1e-4, head_ridge=.001)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested_indices(dataset, n_rows, groups, fold):
    """Return fitting, inner validation and outer scoring positions before capping."""
    if fold not in range(3):
        raise ValueError("fold must be 0, 1 or 2.")
    rows = np.arange(n_rows)
    if dataset == "concrete_slump":
        outer = KFold(3, shuffle=True, random_state=SPLIT_SEED).split(rows)
    elif dataset == "parkinsons":
        outer = GroupKFold(3).split(rows, groups=groups)
    elif dataset == "air_quality":
        outer = TimeSeriesSplit(3, gap=24).split(rows)
    else:
        raise ValueError(f"Unknown development dataset: {dataset}")
    outer_train, evaluate = list(outer)[fold]
    if dataset == "concrete_slump":
        train, validation = train_test_split(outer_train, test_size=.2, random_state=SPLIT_SEED + fold)
    elif dataset == "parkinsons":
        a, b = next(GroupShuffleSplit(1, test_size=.2, random_state=SPLIT_SEED + fold)
                    .split(outer_train, groups=np.asarray(groups)[outer_train]))
        train, validation = outer_train[a], outer_train[b]
    else:
        cut = int(.8 * len(outer_train))
        if cut <= 24:
            raise ValueError("Too few chronological training rows for the inner 24-row gap.")
        train, validation = outer_train[:cut - 24], outer_train[cut:]
    return {"train": np.sort(train), "val": np.sort(validation), "eval": evaluate}


def _standardize(parts, metadata, dataset, fold, source_sha256):
    """Use only capped fitting rows to learn all numerical transformations."""
    train = parts["train"]
    available = len(train[0])
    selected = np.arange(available)
    if available > TRAIN_CAP:
        selected = np.sort(np.random.default_rng(SPLIT_SEED + (fold or 0))
                           .choice(available, TRAIN_CAP, replace=False))
    parts = dict(parts, train=tuple(a[selected] for a in train))
    X, Y = parts["train"][:2]
    if len(X) < 5:
        raise ValueError("At least five fitting rows are required.")
    mean, scale, target_scale = X.mean(axis=0), X.std(axis=0), Y.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    target_scale = np.where(target_scale > 0, target_scale, np.maximum(np.abs(Y).max(axis=0), 1.))
    if not np.isfinite(np.concatenate((mean, scale, target_scale))).all():
        raise ValueError("Nonfinite training transformations.")
    rows = {name: part[2].tolist() for name, part in parts.items()}
    flat = [row for values in rows.values() for row in values]
    if len(set(flat)) != len(flat):
        raise ValueError("Row identifiers overlap within or across split partitions.")
    info = dict(dataset=dataset, fold=fold, source_sha256=source_sha256,
                feature_names=list(metadata["feature_names"]), target_names=list(metadata["target_names"]),
                input_mean=mean.tolist(), input_scale=scale.tolist(), target_scale=target_scale.tolist(),
                row_ids=rows, group_ids={name: part[3].tolist() for name, part in parts.items()},
                row_counts={name: len(part[0]) for name, part in parts.items()},
                available_fitting_rows=available, train_cap=TRAIN_CAP,
                fitting_subset_positions=selected.tolist(), split_seed=SPLIT_SEED + (fold or 0),
                units_policy="Standardized inputs are dimensionless; no inferred physical units.")
    info["fold_sha256"] = hashlib.sha256(json.dumps(info, sort_keys=True, allow_nan=False).encode()).hexdigest()
    split = dict(feature_names=info["feature_names"], target_names=info["target_names"],
                 target_scale=target_scale, metadata=info)
    for name, (inputs, targets, _, _) in parts.items():
        if (inputs.ndim != 2 or targets.ndim != 2 or len(inputs) != len(targets) or
                inputs.shape[1] != len(info["feature_names"]) or targets.shape[1] != len(info["target_names"]) or
                not np.isfinite(inputs).all() or not np.isfinite(targets).all()):
            raise ValueError(f"Invalid arrays in {name} partition.")
        suffix = "" if name == "train" else "_" + name
        split["X" + suffix], split["Y" + suffix] = (inputs - mean) / scale, targets
    return split


def prepare_split(dataset, fold):
    """Prepare one of three fixed outer folds from only v1's train+val NPZ."""
    if dataset not in ("concrete_slump", "parkinsons", "air_quality"):
        raise ValueError(f"Unknown development dataset: {dataset}")
    metadata = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))["datasets"][dataset]
    path = DATA_DIR / f"{dataset}.npz"
    digest = sha(path)
    if digest != metadata["trainval_sha256"]:
        raise ValueError("Development source hash differs from the frozen manifest.")
    with np.load(path, allow_pickle=False) as data:
        X, Y = (np.concatenate([data[f"{key}_train"], data[f"{key}_val"]]) for key in ("X", "Y"))
        ids, groups = (np.concatenate([data[f"train_{key}"], data[f"val_{key}"]]) for key in ("ids", "groups"))
    indices = nested_indices(dataset, len(X), groups, fold)
    parts = {name: (X[index], Y[index], ids[index], groups[index]) for name, index in indices.items()}
    split = _standardize(parts, metadata, dataset, fold, digest)
    # Positions refer to the concatenated frozen train+val file, not new observations.
    selected = split["metadata"]["fitting_subset_positions"]
    split["metadata"]["pool_indices"] = {name: (index[selected] if name == "train" else index).tolist()
                                           for name, index in indices.items()}
    return split


def prepare_trainval(path, metadata, *, dataset):
    """Prepare an explicitly supplied train/val NPZ, without looking up a test path."""
    digest = sha(path)
    if metadata.get("trainval_sha256", digest) != digest:
        raise ValueError("Train/validation source hash differs from its manifest.")
    with np.load(path, allow_pickle=False) as data:
        parts = {}
        for name in ("train", "val"):
            X, Y = data[f"X_{name}"], data[f"Y_{name}"]
            ids = data[f"{name}_ids"] if f"{name}_ids" in data else np.array([f"{name}:{i}" for i in range(len(X))])
            groups = data[f"{name}_groups"] if f"{name}_groups" in data else ids
            parts[name] = X, Y, ids, groups
    return _standardize(parts, metadata, dataset, None, digest)


def metrics(y, prediction, scales):
    if prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or nonfinite predictions.")
    rmse = np.sqrt(np.mean((prediction - y) ** 2, axis=0))
    nrmse = rmse / scales
    if not np.isfinite(nrmse).all():
        raise ValueError("Nonfinite normalized errors.")
    r2 = np.atleast_1d(r2_score(y, prediction, multioutput="raw_values")) if len(y) > 1 else np.full(y.shape[1], np.nan)
    return dict(rmse=rmse.tolist(), nrmse=nrmse.tolist(), mean_nrmse=float(nrmse.mean()),
                worst_nrmse=float(nrmse.max()), r2=[float(value) if np.isfinite(value) else None for value in r2])


def model_complexity(models):
    """Give independent models the same value-aware DAG reuse credit as joint ones."""
    features, connections, offsets, storage = [], 0, 0, 0
    for model in models:
        features.extend(model.features_)
        edges, biases = int(np.count_nonzero(model.coef_)), int(np.count_nonzero(model.intercept_))
        connections += edges
        offsets += biases
        storage += graph_size(model.features_) + edges + biases
    nodes = graph_size(features)
    return dict(complexity=nodes + connections + offsets, shared_dag_nodes=nodes,
                nonzero_connections=connections, nonzero_intercepts=offsets, storage_complexity=storage)


def train_run(split, method, seed, budget=20.):
    """Fit and score one system; return (JSON-ready report, fitted model list).

    The caller owns persistence and the 120-second process timeout. Search CPU
    checkpoints cannot interrupt native numerical calls or post-search work.
    """
    if method not in METHODS or not np.isfinite(budget) or budget <= 0:
        raise ValueError("Expected a declared method and positive finite CPU budget.")
    output_count = split["Y"].shape[1]
    allocations = [budget / output_count] * output_count if method == "independent" else [budget]
    package_dir = Path(dimensr_mimo.__file__).resolve().parent
    report = dict(split["metadata"], method=method, seed=seed, status="running",
                  cpu_budget_seconds=float(budget), allocated_search_seconds=allocations,
                  python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in
                      ("numpy", "scipy", "sympy", "scikit-learn", "threadpoolctl")},
                  source_sha256={p.relative_to(package_dir.parent).as_posix(): sha(p)
                                 for p in sorted(package_dir.rglob("*.py"))},
                  configs=[], diagnostics=[], model_fit_cpu_seconds=[], model_fit_wall_seconds=[])
    # Preserve both provenance namespaces: source code and the numeric data file.
    report["data_sha256"] = split["metadata"]["source_sha256"]
    models = []
    wall_start, cpu_start = time.monotonic(), time.process_time()
    try:
        with threadpool_limits(limits=1):
            for j, allocation in enumerate(allocations):
                config = dict(BASE_CONFIG, random_state=seed, search_seconds=allocation,
                              candidate_quota=int(method == "coverage"),
                              prune_tolerance=.005 if method == "prune" else 0.)
                report["configs"].append(config)
                columns = slice(j, j + 1) if method == "independent" else slice(None)
                model = MIMOSymbolicRegressor(SearchConfig(**config), feature_names=split["feature_names"],
                                               target_names=split["target_names"][columns])
                start_wall, start_cpu = time.monotonic(), time.process_time()
                model.fit(split["X"], split["Y"][:, columns], X_val=split["X_val"], Y_val=split["Y_val"][:, columns])
                report["model_fit_cpu_seconds"].append(time.process_time() - start_cpu)
                report["model_fit_wall_seconds"].append(time.monotonic() - start_wall)
                models.append(model)
                report["diagnostics"].append(model.diagnostics_)
            report["fit_cpu_seconds"] = time.process_time() - cpu_start
            report["fit_wall_seconds"] = time.monotonic() - wall_start
            report.update(model_complexity(models))
            report["expressions"] = [str(expression) for model in models for expression in model.expressions_]
            report["history"] = [model.history_ for model in models]
            for name, suffix in (("train", ""), ("validation", "_val"), ("eval", "_eval")):
                if "X" + suffix in split:
                    prediction = np.column_stack([model.predict(split["X" + suffix]) for model in models])
                    report[name] = metrics(split["Y" + suffix], prediction, split["target_scale"])
            observed = np.vstack([split[key] for key in ("X", "X_val", "X_eval") if key in split])
            bounds = np.column_stack((observed.min(axis=0), observed.max(axis=0)))
            report["raw_domain"] = dict(training=[model.audit_domain() for model in models],
                                         observed_box=[model.audit_domain(bounds) for model in models],
                                         note="Observed-box audits occur after fitting and do not select a model.")
            if any(audit["status"] != "enclosed" for audit in report["raw_domain"]["training"]):
                raise ValueError("Selected raw model is not enclosed on its training domain.")
            report["status"] = "ok"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    report["search_cpu_seconds"] = sum(d.get("search_cpu_seconds", 0.) for d in report["diagnostics"])
    report["budget_overshoot_seconds"] = sum(d.get("budget_overshoot_seconds", 0.) for d in report["diagnostics"])
    report["total_cpu_seconds"] = time.process_time() - cpu_start
    report["total_wall_seconds"] = time.monotonic() - wall_start
    return report, models
