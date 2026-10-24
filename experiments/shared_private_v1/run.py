"""Frozen development study for shared/private expressions; never opens tests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo._vendor.units import ZERO
from hybrid import search_hybrid, validate_archives, select_indices, combine, inference_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("shared_private_v3_split", ROOT/"experiments/real_blind_v3/runner.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)
DATASETS = ("concrete_slump", "parkinsons", "air_quality")
ARMS = ("joint", "independent", "hybrid")
THREAD_ENV = {key:"1" for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def jobs():
    return [(d,a,f,s) for d in DATASETS for f in range(3) for s in range(5) for a in ARMS]


def name(job):
    d,a,f,s = job
    return f"{d}_{a}_f{f}_s{s}"


def check_freeze():
    frozen = json.loads((HERE/"FREEZE.json").read_text())
    for path,digest in frozen["files"].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return frozen


def prepare():
    old = ROOT/"experiments/budget_capacity_v1"
    splits = json.loads((old/"SPLITS.json").read_text())
    for d in DATASETS:
        for f in range(3):
            assert v3.prepare_split(d,f)["metadata"]["fold_sha256"] == splits[f"{d}_f{f}"]["fold_sha256"]
    files = list((ROOT/"src/dimensr_mimo").rglob("*.py")) + [HERE/f for f in ("hybrid.py","run.py","PROTOCOL.md")]
    files += [old/"SPLITS.json",old/"curves.py",ROOT/"experiments/real_blind_v3/runner.py",ROOT/"experiments/real_blind_v1/data/manifest.json"]
    files += [ROOT/f"experiments/real_blind_v1/data/{d}.npz" for d in DATASETS]
    save(HERE/"FREEZE.json",dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        created_at_utc=datetime.now(timezone.utc).isoformat(),jobs=[name(j) for j in jobs()],
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
        packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","sympy","scikit-learn","threadpoolctl")}))
    print("Frozen 135 runs, five operating points, 45 paired cells.")


def fit_job(job, out):
    check_freeze()
    d,arm,fold,seed = job
    split = v3.prepare_split(d,fold)
    splits = json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    assert split["metadata"]["fold_sha256"] == splits[f"{d}_f{fold}"]["fold_sha256"]
    X,Y = split["X"],split["Y"]
    n,m = X.shape[1],Y.shape[1]
    units,targets,coefficients = (ZERO,)*n,(ZERO,)*m,(ZERO,)
    bounds = np.column_stack((X.min(axis=0),X.max(axis=0)))
    cfg = SearchConfig(**dict(v3.BASE_CONFIG,search_seconds=20.,random_state=seed))
    cpu,wall = time.process_time(),time.monotonic()
    result = dict(dataset=d,arm=arm,fold=fold,seed=seed,status="running",fold_sha256=split["metadata"]["fold_sha256"],
                  freeze_sha256=sha(HERE/"FREEZE.json"),config=asdict(cfg))
    models,diagnostics,archives = [],[],[]
    with threadpool_limits(limits=1):
        if arm == "hybrid":
            pools,counts = search_hybrid(X,Y,bounds,units,targets,coefficients,cfg)
            diagnostics.append(counts)
            archives = validate_archives(pools,split["X_val"],split["Y_val"],split["target_scale"],bounds,targets,cfg.interval_max_boxes)
        else:
            count = m if arm == "independent" else 1
            for j in range(count):
                columns = slice(j,j+1) if count>1 else slice(None)
                local_cfg = SearchConfig(**dict(asdict(cfg),search_seconds=20./count))
                model = MIMOSymbolicRegressor(local_cfg,feature_names=split["feature_names"],target_names=split["target_names"][columns])
                model.fit(X,Y[:,columns],X_val=split["X_val"],Y_val=split["Y_val"][:,columns])
                models.append(model)
                diagnostics.append(model.diagnostics_)
                archives.append(model.pareto_front_)
        result.update(diagnostics=diagnostics,fit_cpu_seconds=time.process_time()-cpu,
            search_cpu_seconds=sum(c["search_cpu_seconds"] for c in diagnostics),
            overshoot_seconds=sum(c["budget_overshoot_seconds"] for c in diagnostics),
            archive_counts=list(map(len,archives)))
        save(out/"archives.json",archives)
        selection_start = time.process_time()
        variants = {"joint":models[0]} if arm == "joint" else {}
        indices = {}
        if arm != "joint":
            for rule in ("penalty","compact"):
                label = f"{arm}_{rule}"
                indices[label] = select_indices(archives,rule,cfg.complexity_weight)
                candidate = combine(archives,indices[label])
                variants[label] = inference_model(candidate,cfg,bounds,split["target_scale"],split["feature_names"],
                    split["target_names"],units,targets,coefficients)
            if arm == "independent":
                np.testing.assert_allclose(variants["independent_penalty"].predict(split["X_val"]),
                    np.column_stack([model.predict(split["X_val"]) for model in models]),rtol=1e-12,atol=1e-12)
        result["selection_cpu_seconds"] = time.process_time()-selection_start
        # Persist every choice before touching any outer-fold label.
        frozen_choices = {}
        for label,model in variants.items():
            file = f"{label}.json"
            model.save(out/file)
            restored = MIMOSymbolicRegressor.load(out/file)
            np.testing.assert_array_equal(restored.predict(split["X_val"]),model.predict(split["X_val"]))
            counts = np.count_nonzero(model.coef_,axis=0)
            frozen_choices[label] = dict(model=file,model_sha256=sha(out/file),indices=indices.get(label),
                complexity=model.model_["complexity"],shared_basis=int(np.count_nonzero(counts>1)),
                private_basis=int(np.count_nonzero(counts==1)),connections_per_output=np.count_nonzero(model.coef_,axis=1).tolist())
        save(out/"CHOICES.json",frozen_choices)
        scoring = time.process_time()
        evaluated = {}
        for label,model in variants.items():
            entry = dict(frozen_choices[label],status="ok")
            try:
                for field,suffix in (("train",""),("inner","_val"),("outer","_eval")):
                    entry[field] = v3.metrics(split["Y"+suffix],model.predict(split["X"+suffix]),split["target_scale"])
            except ValueError as exc:
                entry.update(status="failed",error=str(exc))
            evaluated[label] = entry
        result.update(variants=evaluated,score_cpu_seconds=time.process_time()-scoring,
            status="ok" if all(v["status"]=="ok" for v in evaluated.values()) else "failed",
            archives_sha256=sha(out/"archives.json"),choices_sha256=sha(out/"CHOICES.json"))
    result.update(total_cpu_seconds=time.process_time()-cpu,total_wall_seconds=time.monotonic()-wall)
    return result


def launch(job):
    out = HERE/"runs"/name(job)
    if out.exists():
        p = out/"result.json"
        if not p.exists():
            raise FileExistsError(f"Incomplete run preserved: {out}")
        r = json.loads(p.read_text())
        if r["freeze_sha256"] != sha(HERE/"FREEZE.json"):
            raise ValueError("Existing run has a different freeze.")
        return r["status"]
    out.mkdir(parents=True)
    save(out/"started.json",dict(job=job,freeze_sha256=sha(HERE/"FREEZE.json"),time_utc=datetime.now(timezone.utc).isoformat()))
    with (out/"process.log").open("x",encoding="utf-8") as log:
        try:
            child = subprocess.run([sys.executable,str(Path(__file__).resolve()),"job",*map(str,job)],
                stdout=log,stderr=subprocess.STDOUT,timeout=300,env=dict(os.environ,**THREAD_ENV,PYTHONHASHSEED="0",PYTHONUTF8="1"))
            p = out/"child.json"
            r = json.loads(p.read_text()) if p.exists() else dict(status="failed",error="No completed child record")
            if child.returncode:
                r.update(status="failed",returncode=child.returncode)
        except subprocess.TimeoutExpired:
            r = dict(status="timeout",error="300-second wall timeout")
        except OSError as exc:
            r = dict(status="failed",error=str(exc))
    r.update(dataset=job[0],arm=job[1],fold=job[2],seed=job[3],freeze_sha256=sha(HERE/"FREEZE.json"))
    save(out/"result.json",r)
    return r["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","batch","job"))
    parser.add_argument("identity",nargs="*")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "job":
        d,a,f,s = args.identity
        job = d,a,int(f),int(s)
        assert job in jobs()
        out = HERE/"runs"/name(job)
        try:
            r = fit_job(job,out)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = dict(status="failed",error=f"{type(exc).__name__}: {exc}")
        save(out/"child.json",r)
        return int(r["status"]!="ok")
    else:
        check_freeze()
        if args.workers<1:
            parser.error("workers must be positive")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(launch,j):j for j in jobs()}
            statuses = []
            for future in as_completed(futures):
                status = future.result();statuses.append(status)
                print(name(futures[future]),status,flush=True)
        print(f"Finished {len(statuses)} runs; failures={sum(s!='ok' for s in statuses)}",flush=True)
        return int(any(s!="ok" for s in statuses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
