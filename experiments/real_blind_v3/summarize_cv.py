"""Audit every declared CV run and apply only the frozen v3 selection gates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATASETS = {"concrete_slump": 3, "parkinsons": 2, "air_quality": 3}
METHODS = ("joint", "independent", "coverage", "prune")


def _read_run(folder, identity):
    result_path, timeout_path = folder / "result.json", folder / "timeout.json"
    path = timeout_path if timeout_path.exists() else result_path
    row = dict(identity, status="missing", path=str(path))
    if not path.exists():
        row["error"] = "Required result and timeout records are both missing."
        return row
    content = path.read_bytes()
    row["record_sha256"] = hashlib.sha256(content).hexdigest()
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Run record must be a JSON object.")
        if any(data.get(key) != value for key, value in identity.items()):
            raise ValueError("Run identity differs from its required directory.")
        row.update(status=data.get("status", "failed"), fold_sha256=data.get("fold_sha256"))
        digest = row["fold_sha256"]
        if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or
                                   any(char not in "0123456789abcdef" for char in digest)):
            row["fold_sha256"] = None
            raise ValueError("Invalid fold_sha256.")
        if timeout_path.exists():
            row.update(status="timeout", error=data.get("error", "Wall timeout recorded."))
        elif row["status"] != "ok":
            row.update(status="failed", error=data.get("error", "Run did not finish successfully."))
        else:
            nrmse = np.asarray(data["eval"]["nrmse"], dtype=float)
            if nrmse.shape != (DATASETS[identity["dataset"]],) or not np.isfinite(nrmse).all() or np.any(nrmse < 0):
                raise ValueError("Invalid per-output outer-fold NRMSE.")
            for key in ("complexity", "storage_complexity", "search_cpu_seconds", "budget_overshoot_seconds"):
                value = data[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {key}.")
                row[key] = value
            if row["fold_sha256"] is None:
                raise ValueError("Missing or invalid fold_sha256.")
            row["nrmse"] = nrmse.tolist()
            row["mean_nrmse"] = float(nrmse.mean())
    except (ValueError, TypeError, KeyError) as exc:
        row.update(status="failed", error=f"Invalid run record: {exc}")
    return row


def _aggregate(rows):
    datasets = {}
    for dataset, outputs in DATASETS.items():
        part = [row for row in rows if row["dataset"] == dataset]
        succeeded = sum(row["status"] == "ok" for row in part)
        entry = dict(required_runs=15, successful_runs=succeeded, complete=succeeded == 15)
        if entry["complete"]:
            per_output = np.asarray([row["nrmse"] for row in part]).mean(axis=0)
            entry.update(per_output_nrmse=per_output.tolist(), S=float(per_output.mean()),
                         worst_output_nrmse=float(per_output.max()))
            for output_key, input_key in (("K", "complexity"), ("storage_complexity", "storage_complexity"),
                                          ("search_cpu_seconds", "search_cpu_seconds"),
                                          ("budget_overshoot_seconds", "budget_overshoot_seconds")):
                entry[output_key] = float(np.mean([row[input_key] for row in part]))
            entry["max_budget_overshoot_seconds"] = max(row["budget_overshoot_seconds"] for row in part)
        else:
            entry.update(per_output_nrmse=None, S=None, K=None,
                         reason="Incomplete: failed or missing runs are never omitted from means.")
        datasets[dataset] = entry
    complete = all(entry["complete"] for entry in datasets.values())
    report = dict(required_runs=45, successful_runs=sum(row["status"] == "ok" for row in rows),
                  complete=complete, datasets=datasets, S=None, K=None)
    if complete:
        for key in ("S", "K", "storage_complexity", "search_cpu_seconds", "budget_overshoot_seconds"):
            report[key] = float(np.mean([entry[key] for entry in datasets.values()]))
        report["max_budget_overshoot_seconds"] = max(entry["max_budget_overshoot_seconds"] for entry in datasets.values())
    return report


def _at_most(value, limit):
    # Averaging must not turn equality at a declared gate into a rounding failure.
    return bool(value <= limit or np.isclose(value, limit, rtol=1e-12, atol=1e-12))


def _gate(method, control, candidate, protocol_errors):
    gate = dict(eligible=False, all_45_succeeded=candidate["complete"],
                control_all_45_succeeded=control["complete"], comparable_splits=not protocol_errors)
    if not candidate["complete"] or not control["complete"] or protocol_errors:
        gate["reason"] = "Incomplete required runs or inconsistent protocol records."
        return gate
    if method == "coverage":
        score = candidate["S"] < control["S"] and _at_most(candidate["S"], .99 * control["S"])
        complexity = _at_most(candidate["K"], 1.1 * control["K"])
    else:
        score = _at_most(candidate["S"], 1.005 * control["S"])
        complexity = candidate["K"] < control["K"] and _at_most(candidate["K"], .9 * control["K"])
    guards = []
    for dataset in DATASETS:
        baseline = control["datasets"][dataset]["per_output_nrmse"]
        revised = candidate["datasets"][dataset]["per_output_nrmse"]
        for output, (old, new) in enumerate(zip(baseline, revised)):
            limit = max(.01, .02 * old)
            guards.append(dict(dataset=dataset, output_index=output, baseline_nrmse=old,
                               candidate_nrmse=new, increase=new - old, allowed_increase=limit,
                               passed=_at_most(new - old, limit)))
    gate.update(score_passed=bool(score), complexity_passed=bool(complexity), output_guards=guards,
                worst_output_passed=all(item["passed"] for item in guards),
                relative_score_change=(candidate["S"] / control["S"] - 1) if control["S"] else None,
                relative_complexity_change=(candidate["K"] / control["K"] - 1) if control["K"] else None)
    gate["eligible"] = bool(score and complexity and gate["worst_output_passed"])
    return gate


def summarize(cv_dir):
    cv_dir = Path(cv_dir)
    rows, expected_paths = [], set()
    for dataset in DATASETS:
        for method in METHODS:
            for fold in range(3):
                for seed in range(5):
                    identity = dict(dataset=dataset, method=method, fold=fold, seed=seed)
                    folder = cv_dir / f"{dataset}_{method}_f{fold}_s{seed}"
                    expected_paths.update((folder / "result.json", folder / "timeout.json"))
                    rows.append(_read_run(folder, identity))
    errors = []
    extra_paths = (set(cv_dir.glob("*/result.json")) | set(cv_dir.glob("*/timeout.json"))) - expected_paths
    if extra_paths:
        errors.append(dict(kind="unexpected_records", paths=sorted(map(str, extra_paths))))
    for dataset in DATASETS:
        for fold in range(3):
            matched = [row for row in rows if row["dataset"] == dataset and row["fold"] == fold and row.get("fold_sha256")]
            hashes = sorted({row["fold_sha256"] for row in matched})
            if len(hashes) > 1:
                errors.append(dict(kind="split_mismatch", dataset=dataset, fold=fold, hashes=hashes))
    methods = {method: _aggregate([row for row in rows if row["method"] == method]) for method in METHODS}
    gates = {method: _gate(method, methods["joint"], methods[method], errors) for method in ("coverage", "prune")}
    selected = None
    if not errors and methods["joint"]["complete"]:
        qualifying = [method for method in gates if gates[method]["eligible"]]
        selected = min(qualifying, key=lambda method: (methods[method]["S"], methods[method]["K"])) if qualifying else "joint"
    counts = {status: sum(row["status"] == status for row in rows) for status in ("ok", "failed", "timeout", "missing")}
    return dict(protocol="real_blind_v3", required_runs=180, counts=counts,
                all_required_succeeded=counts["ok"] == 180 and not errors,
                methods=methods, gates=gates, selected_method=selected,
                selection_status="selected" if selected is not None else "blocked",
                protocol_errors=errors, runs=rows,
                aggregation="Equal weight over outputs, fold/seed, then datasets; no incomplete-method means.",
                interpretation="Engineering gates on development data; folds and seeds are not IID replicates.",
                summarize_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="CV run directory")
    parser.add_argument("--output", type=Path, required=True, help="New summary JSON; existing files are refused")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = summarize(args.input)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)
    print(json.dumps(dict(counts=result["counts"], selected_method=result["selected_method"])))
    if result["selection_status"] == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
