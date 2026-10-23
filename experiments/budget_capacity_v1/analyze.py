"""Aggregate frozen development runs without fitting or reselecting models."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import statistics

from diagnose import ARMS, DATASETS, HERE, check_freeze, jobs, name, sha
from curves import select_cap

CAPS = (10, 20, 30, 40, 60, 80, 120)


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.pstdev(values),
                min=min(values), max=max(values), median=statistics.median(values))


def point_row(record, point, cap):
    row = {k: record[k] for k in ("dataset", "arm", "budget", "fold", "seed")}
    row.update(cap=cap, status=record["status"])
    if record["status"] != "ok":
        return row
    row["status"] = point.get("status", "ok") if point else "unattainable"
    if row["status"] != "ok":
        return row
    row.update(complexity=point["complexity"], validation_loss=point["validation_loss"])
    for split in ("train", "inner", "outer"):
        row[f"{split}_nrmse"] = point[split]["mean_nrmse"]
        row[f"{split}_worst_nrmse"] = point[split]["worst_nrmse"]
        for metric in ("nrmse", "r2"):
            for j, value in enumerate(point[split][metric]):
                row[f"{split}_{metric}_{j}"] = value
    return row


def aggregate(rows, expected=6):
    good = [row for row in rows if row["status"] == "ok"]
    result = dict(expected=expected, available=len(good), complete=len(good) == expected)
    # Do not silently average a subset with easier folds or more valid predictions.
    if result["complete"]:
        fields = set.intersection(*(set(r) for r in good)) - {
            "dataset", "arm", "budget", "fold", "seed", "cap", "status"}
        result["metrics"] = {key: stats([row[key] for row in good])
                             for key in sorted(fields) if isinstance(good[0][key], (int, float))}
    return result


def paired_rows(rows, left, right, label):
    """Difference is left minus right; negative error favors left. Never impute."""
    def index(selector):
        return {(r["dataset"], r["fold"], r["seed"], r["cap"]): r for r in rows
                if (r["arm"], r["budget"]) == selector}
    a, b = index(left), index(right)
    paired = []
    for key in sorted(a.keys() | b.keys(), key=str):
        ra, rb = a.get(key), b.get(key)
        dataset, fold, seed, cap = key
        row = dict(dataset=dataset, fold=fold, seed=seed, cap=cap, comparison=label,
                   status="ok" if ra and rb and ra["status"] == rb["status"] == "ok" else "incomplete")
        if row["status"] == "ok":
            for field in sorted(ra.keys() & rb.keys()):
                if field == "complexity" or field.startswith(("train_", "inner_", "outer_")):
                    row[f"delta_{field}"] = ra[field] - rb[field]
            row.update(left_nrmse=ra["outer_nrmse"], right_nrmse=rb["outer_nrmse"],
                       left_complexity=ra["complexity"], right_complexity=rb["complexity"])
        paired.append(row)
    return paired


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tables(summary):
    lines = ["# Frozen diagnostic tables", "", "NRMSE is lower-is-better; K is actual selected complexity. All means require six complete fold/seed cells.", "",
             "## Default operating points", "",
             "| Dataset | Arm | CPU cap | Train | Inner | Outer | K | Basis slots used | Selected bound hits / components |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary["default"]:
        m = row.get("metrics")
        values = "incomplete | | | | |" if not m else (
            f'{m["train_nrmse"]["mean"]:.6f} | {m["inner_nrmse"]["mean"]:.6f} | {m["outer_nrmse"]["mean"]:.6f} | '
            f'{m["complexity"]["mean"]:.2f} | {m["selected_terms"]["mean"]:.2f} | '
            f'{round(m["selected_bound_hits"]["mean"]*6)} / {round(m["components"]["mean"]*6)}')
        lines.append(f'| {row["dataset"]} | {row["arm"]} | {row["budget"]} | {values} |')
    lines += ["", "## Prespecified common complexity caps", "", "Each entry is outer NRMSE / actual K. Models are selected using inner validation alone.", "",
              "| Dataset | CPU cap | K cap | Joint8 | Independent8 | Joint wide |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    index = {(r["dataset"], r["arm"], r["budget"], r["cap"]): r for r in summary["curves"]}
    for dataset, budget, cap in product(DATASETS, (20, 60), CAPS):
        values = []
        for arm in ARMS:
            m = index[dataset, arm, budget, cap].get("metrics")
            values.append(f'{m["outer_nrmse"]["mean"]:.6f} / {m["complexity"]["mean"]:.2f}' if m else "incomplete")
        lines.append(f'| {dataset} | {budget} | {cap} | ' + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def analyze(out):
    frozen = check_freeze()
    split_hashes = json.loads((HERE / "SPLITS.json").read_text(encoding="utf-8"))
    expected = {name(job) for job in jobs()}
    paths = {p.parent.name: p for p in (HERE / "runs").glob("*/result.json")}
    if expected != set(paths):
        raise ValueError(f"Incomplete grid: missing={sorted(expected-set(paths))}, unexpected={sorted(set(paths)-expected)}")
    records = []
    inputs = {}
    for job in jobs():
        path = paths[name(job)]
        r = json.loads(path.read_text(encoding="utf-8"))
        assert tuple(r[k] for k in ("dataset", "arm", "budget", "fold", "seed")) == job
        assert r["freeze_sha256"] == sha(HERE / "FREEZE.json")
        inputs[path.relative_to(HERE).as_posix()] = sha(path)
        if r["status"] == "ok":
            assert r["fold_sha256"] == split_hashes[f'{r["dataset"]}_f{r["fold"]}']["fold_sha256"]
            for file, digest in r["model_sha256"].items():
                assert sha(path.parent / file) == digest
            assert sha(path.parent / "archives.json") == r["archives_sha256"]
        records.append(r)
    max_cap = max(120, max(p["complexity"] for r in records if r["status"] == "ok" for p in r["frontier"]))
    defaults, curves = [], []
    for r in records:
        row = point_row(r, r.get("default"), "default")
        if r["status"] == "ok":
            point = r["default"]
            limits = [c["max_terms"] for c in r["configs"]]
            archives = json.loads((HERE / "runs" / name(tuple(r[k] for k in (
                "dataset", "arm", "budget", "fold", "seed"))) / "archives.json").read_text(encoding="utf-8"))
            row.update(selected_terms=sum(point["selected_terms"]), components=len(limits),
                       selected_bound_hits=sum(t == lim for t, lim in zip(point["selected_terms"], limits)),
                       archive_bound_hits=sum(t == lim for t, lim in zip(point["training_archive_max_terms"], limits)),
                       head_fits=sum(d["head_fits"] for d in r["diagnostics"]),
                       feature_attempts=sum(d["feature_attempts"] for d in r["diagnostics"]),
                       archive_min_train_loss=statistics.mean(min(c["train_loss"] for c in a) for a in archives),
                       archive_min_inner_loss=statistics.mean(min(c["validation_loss"] for c in a) for a in archives),
                       archive_candidates=sum(r["archive_counts"]), enumerated_combinations=r["enumerated_combinations"])
            for field in ("search_cpu_seconds", "fit_cpu_seconds", "fit_wall_seconds", "overshoot_seconds",
                          "curve_selection_cpu_seconds", "score_cpu_seconds", "total_cpu_seconds", "total_wall_seconds"):
                row[field] = r[field]
        defaults.append(row)
        for cap in range(max_cap + 1):
            point = select_cap(r["frontier"], cap) if r["status"] == "ok" else None
            curves.append(point_row(r, point, cap))
    comparisons = [( (arm, 60), (arm, 20), f"{arm}:60-20") for arm in ARMS]
    comparisons += [((arm, budget), ("joint8", budget), f"{arm}-joint8:{budget}")
                    for budget in (20, 60) for arm in ("joint_wide", "independent8")]
    pairs = [row for left, right, label in comparisons
             for row in paired_rows(defaults + curves, left, right, label)]
    summary = dict(freeze_sha256=sha(HERE / "FREEZE.json"), source_commit=frozen["source_commit"],
                   result_sha256=inputs, expected_runs=len(expected), successful_runs=sum(r["status"] == "ok" for r in records),
                   failures=[{k:r[k] for k in ("dataset", "arm", "budget", "fold", "seed", "status")} for r in records if r["status"] != "ok"],
                   failed_frontier_points=sum(p["status"] != "ok" for r in records for p in r.get("frontier", [])),
                   max_cap=max_cap, prespecified_caps=CAPS, default=[], curves=[], paired=[])
    for dataset, arm, budget in product(DATASETS, ARMS, (20, 60)):
        def matching(rows):
            return [r for r in rows if (r["dataset"], r["arm"], r["budget"]) == (dataset, arm, budget)]
        identity = dict(dataset=dataset, arm=arm, budget=budget)
        summary["default"].append(dict(identity, **aggregate(matching(defaults))))
        selected = matching(curves)
        for cap in range(max_cap + 1):
            summary["curves"].append(dict(identity, cap=cap, **aggregate([r for r in selected if r["cap"] == cap])))
    for dataset, (_, _, label), cap in product(DATASETS, comparisons, ("default", *CAPS)):
        selected = [r for r in pairs if (r["dataset"], r["comparison"], r["cap"]) == (dataset, label, cap)]
        group = dict(dataset=dataset, comparison=label, cap=cap, **aggregate(selected))
        if group["complete"]:
            group["relative_mean_error_change_pct"] = 100 * group["metrics"]["delta_outer_nrmse"]["mean"] / group["metrics"]["right_nrmse"]["mean"]
            group["left_error_lower_cells"] = sum(r["delta_outer_nrmse"] < 0 for r in selected)
        summary["paired"].append(group)
    out.mkdir(exist_ok=False)
    for filename, rows in (("default.csv", defaults), ("curves.csv", curves), ("paired.csv", pairs)):
        write_csv(out / filename, rows)
    summary["analysis_source_sha256"] = sha(__file__)
    summary["csv_sha256"] = {p.name: sha(p) for p in out.glob("*.csv")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (out / "TABLES.md").write_text(tables(summary), encoding="utf-8")
    print(f'Aggregated {summary["successful_runs"]}/{len(expected)} runs; failed curve points={summary["failed_frontier_points"]}; caps=0..{max_cap}.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "analysis", help="new directory; never overwrites existing results")
    analyze(parser.parse_args().output)
