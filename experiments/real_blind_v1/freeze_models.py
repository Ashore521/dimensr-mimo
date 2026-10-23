"""Validate the predeclared decision and hash all artifacts before opening test data."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
from run_development import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Existing freeze must not be overwritten")
    models, files, scores = [], {}, {}
    for role, directory in (("baseline", args.baseline), ("selected", args.selected)):
        scores[role] = []
        for dataset in ("concrete_slump", "parkinsons", "air_quality"):
            for seed in (0, 1):
                run = directory / f"{dataset}_symbolic_s{seed}"
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                if result["status"] != "ok" or (run / "timeout.json").exists():
                    raise ValueError(f"Ineligible run: {run}")
                scores[role].append(result["validation"]["mean_nrmse"])
                models.append(dict(role=role, dataset=dataset, seed=seed, method="symbolic",
                                   directory=run.resolve().relative_to(ROOT).as_posix()))
                if role == "selected":
                    for path, expected in result["source_sha256"].items():
                        if sha(ROOT / "src" / path) != expected:
                            raise ValueError(f"Selected source changed after training: {path}")
    averages = {role: statistics.mean(values) for role, values in scores.items()}
    if averages["selected"] > .99*averages["baseline"]:
        raise ValueError("Selected revision fails the predeclared 1% improvement gate")
    for dataset in ("concrete_slump", "parkinsons", "air_quality"):
        for method in ("mean", "ridge", "extra_trees"):
            for seed in ((0, 1) if method == "extra_trees" else (0,)):
                run = args.baseline / f"{dataset}_{method}_s{seed}"
                if json.loads((run / "result.json").read_text(encoding="utf-8"))["status"] != "ok":
                    raise ValueError(f"Failed comparator: {run}")
                models.append(dict(role=method, method=method, dataset=dataset, seed=seed,
                                   directory=run.resolve().relative_to(ROOT).as_posix()))
    for entry in models:
        run = ROOT / entry["directory"]
        for path in (run / "result.json", run / ("model.json" if entry["method"] == "symbolic" else "model.joblib")):
            files[path.relative_to(ROOT).as_posix()] = sha(path)
    protocol = Path(__file__).parent
    for path in list((ROOT / "src/dimensr_mimo").rglob("*.py")) + list(protocol.glob("*.py")) + [protocol / "PROTOCOL.md", protocol / "data/manifest.json"]:
        files[path.relative_to(ROOT).as_posix()] = sha(path)
    manifest = json.loads((protocol / "data/manifest.json").read_text(encoding="utf-8"))
    document = dict(protocol="real_blind_v1", frozen_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    selection_rule="One global revision; six successful runs; validation mean NRMSE improves >=1%",
                    validation_scores=averages, files=files, models=models,
                    sealed_sha256={key: value["sealed_sha256"] for key, value in manifest["datasets"].items()})
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(validation=averages, frozen_models=len(models), freeze_sha256=sha(args.output))))


if __name__ == "__main__":
    main()
