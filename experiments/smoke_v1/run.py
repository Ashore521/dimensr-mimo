"""Small development regression study, not a blind benchmark or novelty claim."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig


def cases(X):
    a, b, c = X.T
    return {
        "shared_polynomial": np.column_stack((2*a*b+.5*c, -a*b+3*c)),
        "shared_trigonometric": np.column_stack((2*np.sin(a)+.5*c, -np.sin(a)+3*c)),
        "shared_rational": np.column_stack((2*a/(1+b*b)+.5*c, -a/(1+b*b)+3*c)),
        "nonlinear_constant": np.column_stack((2*np.sin(1.7*a)+.5*c, -np.sin(1.7*a)+3*c)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261023)
    X = rng.uniform(-1, 1, (280, 3))
    config = SearchConfig(generations=5, proposals_per_generation=256, max_features=256, max_terms=4)
    report = dict(protocol="smoke_v1", purpose="Development regression evidence; no blind evaluation or comparative claim",
                  split=dict(train=160, validation=40, test=80), input_bounds=[[-1, 1]]*3,
                  python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in
                    ("numpy", "scipy", "sympy", "scikit-learn", "dimensr-mimo")}, runs=[])
    source = Path(__file__).resolve().parents[2] / "src" / "dimensr_mimo"
    report["source_sha256"] = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source.rglob("*.py"))}
    for name, Y in cases(X).items():
        for seed in (0, 1):
            config.random_state = seed
            start = time.monotonic()
            model = MIMOSymbolicRegressor(config).fit(X[:160], Y[:160], X_val=X[160:200], Y_val=Y[160:200], input_bounds=[[-1, 1]]*3)
            prediction = model.predict(X[200:])
            rmse = np.sqrt(np.mean((prediction-Y[200:])**2, axis=0))
            nrmse = rmse / model.target_scales_
            filename = f"{name}_s{seed}.json"
            model.save(args.output / filename)
            restored = MIMOSymbolicRegressor.load(args.output / filename)
            np.testing.assert_array_equal(restored.predict(X[200:]), prediction)
            entry = dict(case=name, seed=seed, config=asdict(config), test_rmse=rmse.tolist(),
                         test_nrmse=nrmse.tolist(), max_abs_error=float(np.max(np.abs(prediction-Y[200:]))),
                         numerical_recovery=bool(np.max(np.abs(prediction-Y[200:])) < 1e-6),
                         domain=model.audit_domain()["status"], expressions=[str(e) for e in model.expressions_],
                         model=filename, diagnostics=model.diagnostics_, elapsed_seconds=time.monotonic()-start)
            report["runs"].append(entry)
            (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(name, seed, "test NRMSE", nrmse, "recovery", entry["numerical_recovery"], flush=True)


if __name__ == "__main__":
    main()
