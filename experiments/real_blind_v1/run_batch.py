"""Run bounded, resumable development jobs; each result directory is immutable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--comparators", action="store_true")
    args = p.parse_args()
    runner = Path(__file__).with_name("run_development.py")
    jobs = [(d, "symbolic", s) for d in ("concrete_slump", "parkinsons", "air_quality") for s in (0, 1)]
    if args.comparators:
        jobs += [(d, m, s) for d in ("concrete_slump", "parkinsons", "air_quality")
                 for m in ("mean", "ridge", "extra_trees") for s in ((0, 1) if m == "extra_trees" else (0,))]

    def run(job):
        dataset, method, seed = job
        dest = args.output / f"{dataset}_{method}_s{seed}"
        if dest.exists():
            return f"Preserved existing {dest.name}"
        cmd = [sys.executable, str(runner), "--dataset", dataset, "--method", method,
               "--seed", str(seed), "--source", str(args.source), "--output", str(dest)]
        if args.config:
            cmd += ["--config", str(args.config)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "process.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            return result.stdout.strip() or f"{dest.name}: exit={result.returncode}"
        except subprocess.TimeoutExpired:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "timeout.json").write_text(json.dumps(dict(status="timeout", seconds=240)), encoding="utf-8")
            return f"{dest.name}: timeout after 240s"
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(run, jobs):
            print(result, flush=True)


if __name__ == "__main__":
    main()
