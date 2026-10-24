"""Frozen optional-transfer development experiment; no holdout paths are read."""
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
from transfer import independent, transfer, shared

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("transfer_v3_split",ROOT/"experiments/real_blind_v3/runner.py")
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
DATASETS = ("concrete_slump","parkinsons","air_quality")
METHODS = ("independent20","independent25","random25","semantic25")
BASE_SECONDS, EXTRA_SECONDS, CONTROL_SECONDS = 20.,5.,25.
THREAD_ENV = {k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def jobs():
    return [(d,f,s) for d in DATASETS for f in range(3) for s in range(5)]


def name(job):
    return f"{job[0]}_f{job[1]}_s{job[2]}"


def check_freeze():
    frozen=json.loads((HERE/"FREEZE.json").read_text())
    for path,digest in frozen["files"].items():
        if sha(ROOT/path)!=digest:
            raise ValueError(f"Frozen input changed: {path}")
    return frozen


def prepare():
    splits=json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    for d in DATASETS:
        for f in range(3):
            assert v3.prepare_split(d,f)["metadata"]["fold_sha256"]==splits[f"{d}_f{f}"]["fold_sha256"]
    paths=list((ROOT/"src/dimensr_mimo").rglob("*.py"))+list(HERE.glob("*.py"))
    paths += [HERE/"PROTOCOL.md",ROOT/"experiments/shared_private_v1/hybrid.py",
        ROOT/"experiments/budget_capacity_v1/curves.py",ROOT/"experiments/budget_capacity_v1/SPLITS.json",
        ROOT/"experiments/real_blind_v3/runner.py",ROOT/"experiments/real_blind_v1/data/manifest.json"]
    paths += [ROOT/f"experiments/real_blind_v1/data/{d}.npz" for d in DATASETS]
    paths += list((ROOT/"tests").glob("test_semantic_transfer*.py"))
    save(HERE/"FREEZE.json",dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
        jobs=[name(j) for j in jobs()],created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","sympy","scikit-learn","threadpoolctl")}))
    print("Frozen 45 paired cells, 180 evaluations, 2475 nominal unique training CPU seconds.")


def fit_job(job,out):
    check_freeze()
    d,fold,seed=job
    split=v3.prepare_split(d,fold)
    splits=json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    assert split["metadata"]["fold_sha256"]==splits[f"{d}_f{fold}"]["fold_sha256"]
    X,Y=split["X"],split["Y"]
    units,targets,coefficients=(ZERO,)*X.shape[1],(ZERO,)*Y.shape[1],(ZERO,)
    bounds=np.column_stack((X.min(axis=0),X.max(axis=0)))
    cfg=SearchConfig(**dict(v3.BASE_CONFIG,search_seconds=BASE_SECONDS,random_state=seed))
    cpu,wall=time.process_time(),time.monotonic()
    result=dict(dataset=d,fold=fold,seed=seed,status="running",config=asdict(cfg),
        fold_sha256=split["metadata"]["fold_sha256"],freeze_sha256=sha(HERE/"FREEZE.json"))
    with threadpool_limits(limits=1):
        pools,counts={},{}
        pools["independent20"],counts["independent20"]=independent(X,Y,bounds,units,targets,coefficients,cfg)
        cfg25=SearchConfig(**dict(asdict(cfg),search_seconds=CONTROL_SECONDS))
        pools["independent25"],counts["independent25"]=independent(X,Y,bounds,units,targets,coefficients,cfg25)
        for rule in ("random","semantic"):
            label=rule+"25"
            pools[label],counts[label]=transfer(pools["independent20"],X,Y,bounds,targets,coefficients,cfg,
                seconds=EXTRA_SECONDS,rule=rule)
        result["unique_training_cpu_seconds"]=time.process_time()-cpu
        # All fitting is complete before any inner-validation labels are used.
        archives,choices,models={}, {}, {}
        for label in METHODS:
            selection=time.process_time()
            archives[label]=shared.validate_archives(pools[label],split["X_val"],split["Y_val"],
                split["target_scale"],bounds,targets,cfg.interval_max_boxes)
            indices=shared.select_indices(archives[label],"compact")
            candidate=shared.combine(archives[label],indices)
            model=shared.inference_model(candidate,cfg,bounds,split["target_scale"],split["feature_names"],
                split["target_names"],units,targets,coefficients)
            file=label+".json"
            model.save(out/file)
            restored=MIMOSymbolicRegressor.load(out/file)
            np.testing.assert_array_equal(restored.predict(split["X_val"]),model.predict(split["X_val"]))
            connections=np.count_nonzero(model.coef_,axis=0)
            extra=counts[label]["search_cpu_seconds"]
            training=extra+(counts["independent20"]["search_cpu_seconds"] if label in ("random25","semantic25") else 0.)
            choices[label]=dict(indices=indices,model=file,model_sha256=sha(out/file),
                complexity=model.model_["complexity"],shared_basis=int(np.count_nonzero(connections>1)),
                private_basis=int(np.count_nonzero(connections==1)),
                selected_transfer_heads=sum(i>=len(archives["independent20"][j]) for j,i in enumerate(indices))
                    if label in ("random25","semantic25") else 0,
                training_cpu_seconds=training,selection_cpu_seconds=time.process_time()-selection)
            models[label]=model
        save(out/"archives.json",archives)
        save(out/"CHOICES.json",choices)
        evaluated={}
        for label,model in models.items():
            entry=dict(choices[label],status="ok")
            try:
                for part,suffix in (("train",""),("inner","_val"),("outer","_eval")):
                    entry[part]=v3.metrics(split["Y"+suffix],model.predict(split["X"+suffix]),split["target_scale"])
            except ValueError as exc:
                entry.update(status="failed",error=str(exc))
            evaluated[label]=entry
    result.update(variants=evaluated,diagnostics=counts,
        archives_sha256=sha(out/"archives.json"),choices_sha256=sha(out/"CHOICES.json"),
        status="ok" if all(v["status"]=="ok" for v in evaluated.values()) else "failed",
        total_cpu_seconds=time.process_time()-cpu,total_wall_seconds=time.monotonic()-wall)
    return result


def launch(job):
    out=HERE/"runs"/name(job)
    if out.exists():
        if not (out/"result.json").exists():
            raise FileExistsError(f"Incomplete run preserved: {out}")
        r=json.loads((out/"result.json").read_text())
        if r["freeze_sha256"]!=sha(HERE/"FREEZE.json"):
            raise ValueError("Existing result belongs to another freeze.")
        return r["status"]
    out.mkdir(parents=True)
    save(out/"started.json",dict(job=job,freeze_sha256=sha(HERE/"FREEZE.json"),time_utc=datetime.now(timezone.utc).isoformat()))
    with (out/"process.log").open("x",encoding="utf-8") as log:
        try:
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),"job",*map(str,job)],
                stdout=log,stderr=subprocess.STDOUT,timeout=300,
                env=dict(os.environ,**THREAD_ENV,PYTHONHASHSEED="0",PYTHONUTF8="1"))
            r=json.loads((out/"child.json").read_text()) if (out/"child.json").exists() else dict(status="failed",error="No child result")
            if child.returncode:
                r.update(status="failed",returncode=child.returncode)
        except subprocess.TimeoutExpired:
            r=dict(status="timeout",error="300-second wall timeout")
        except OSError as exc:
            r=dict(status="failed",error=str(exc))
    r.update(dataset=job[0],fold=job[1],seed=job[2],freeze_sha256=sha(HERE/"FREEZE.json"))
    save(out/"result.json",r)
    return r["status"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","batch","job"))
    p.add_argument("identity",nargs="*")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if args.command=="prepare":
        prepare()
    elif args.command=="job":
        d,f,s=args.identity
        job=d,int(f),int(s)
        assert job in jobs()
        try:
            result=fit_job(job,HERE/"runs"/name(job))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result=dict(status="failed",error=f"{type(exc).__name__}: {exc}")
        save(HERE/"runs"/name(job)/"child.json",result)
        return int(result["status"]!="ok")
    else:
        check_freeze()
        if args.workers<1:
            p.error("workers must be positive")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending={pool.submit(launch,job):job for job in jobs()}
            statuses=[]
            for future in as_completed(pending):
                status=future.result();statuses.append(status)
                print(name(pending[future]),status,flush=True)
        print(f"Finished {len(statuses)} cells; failures={sum(s!='ok' for s in statuses)}",flush=True)
        return int(any(s!="ok" for s in statuses))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
