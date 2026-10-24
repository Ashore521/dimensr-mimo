"""Frozen contextual-transfer development study; no external holdouts are read."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import MIMOSymbolicRegressor, SearchConfig
from dimensr_mimo.regressor import _serial_candidate
from dimensr_mimo.search import search, predict_candidate
from dimensr_mimo._vendor.units import ZERO
from contextual import legacy, shared, transfer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("contextual_v3_split",ROOT/"experiments/real_blind_v3/runner.py")
v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3)
DATASETS = ("concrete_slump","parkinsons","air_quality")
METHODS = ("independent20","independent25","semantic25","whole25","context25","joint25")
BASE_SECONDS, EXTRA_SECONDS, CONTROL_SECONDS = 20.,5.,25.
THREAD_ENV = {k:"1" for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","BLIS_NUM_THREADS")}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def jobs():
    return [(d,f,s) for d in DATASETS for f in range(3) for s in range(5)]


def name(job):
    return f"{job[0]}_f{job[1]}_s{job[2]}"


def check_freeze():
    frozen = json.loads((HERE/"FREEZE.json").read_text(encoding="utf-8"))
    for path,digest in frozen["files"].items():
        if sha(ROOT/path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return frozen


def prepare():
    if (HERE/"FREEZE.json").exists():
        raise FileExistsError("This study is already frozen; create a separate experiment for changes.")
    splits = json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    for d in DATASETS:
        for f in range(3):
            assert v3.prepare_split(d,f)["metadata"]["fold_sha256"] == splits[f"{d}_f{f}"]["fold_sha256"]
    paths = list((ROOT/"src/dimensr_mimo").rglob("*.py"))+list(HERE.glob("*.py"))
    paths += [HERE/"PROTOCOL.md",ROOT/"experiments/semantic_transfer_v1/transfer.py",
        ROOT/"experiments/shared_private_v1/hybrid.py",ROOT/"experiments/budget_capacity_v1/curves.py",
        ROOT/"experiments/budget_capacity_v1/SPLITS.json",ROOT/"experiments/real_blind_v3/runner.py",
        ROOT/"experiments/real_blind_v1/data/manifest.json"]
    paths += [ROOT/f"experiments/real_blind_v1/data/{d}.npz" for d in DATASETS]
    paths += list((ROOT/"tests").glob("test_contextual*.py"))
    save(HERE/"FREEZE.json",dict(files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)},
        jobs=[name(j) for j in jobs()],methods=METHODS,created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_note="New experiment files are pinned by SHA256; base commit is not a claim they are committed.",
        python=platform.python_version(),platform=platform.platform(),
        packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","sympy","scikit-learn","threadpoolctl")}))
    print("Frozen 45 paired cells, 270 evaluations, 3825 nominal unique search CPU seconds.",flush=True)


def select_joint(pool, split, bounds, targets, cfg):
    """Use the released joint selector after fitting all experimental arms."""
    archive = []
    for c in pool:
        if not shared.enclosed(c,bounds,targets,cfg.interval_max_boxes):
            continue
        prediction = predict_candidate(c,split["X_val"])
        with np.errstate(all="ignore"):
            loss = float(np.mean(((prediction-split["Y_val"])/split["target_scale"])**2))
        if np.isfinite(prediction).all() and np.isfinite(loss):
            archive.append(dict(_serial_candidate(c),validation_loss=loss,origin="joint"))
    if not archive:
        raise ValueError("No legal finite joint validation candidate.")
    index = min(range(len(archive)),key=lambda i:(archive[i]["validation_loss"]+
        cfg.complexity_weight*archive[i]["complexity"],archive[i]["complexity"]))
    return archive,[index],shared.decode(archive[index])


def fit_job(job, out):
    check_freeze()
    d,fold,seed = job
    split = v3.prepare_split(d,fold)
    splits = json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    assert split["metadata"]["fold_sha256"] == splits[f"{d}_f{fold}"]["fold_sha256"]
    X,Y = split["X"],split["Y"]
    units,targets,coefficients = (ZERO,)*X.shape[1],(ZERO,)*Y.shape[1],(ZERO,)
    bounds = np.column_stack((X.min(axis=0),X.max(axis=0)))
    cfg = SearchConfig(**dict(v3.BASE_CONFIG,search_seconds=BASE_SECONDS,random_state=seed))
    cfg25 = replace(cfg,search_seconds=CONTROL_SECONDS)
    cpu,wall = time.process_time(),time.monotonic()
    result = dict(dataset=d,fold=fold,seed=seed,status="running",config=asdict(cfg),
        fold_sha256=split["metadata"]["fold_sha256"],freeze_sha256=sha(HERE/"FREEZE.json"))
    with threadpool_limits(limits=1):
        pools,counts = {},{}
        pools["independent20"],counts["independent20"] = legacy.independent(X,Y,bounds,units,targets,coefficients,cfg)
        pools["independent25"],counts["independent25"] = legacy.independent(X,Y,bounds,units,targets,coefficients,cfg25)
        pools["semantic25"],counts["semantic25"] = legacy.transfer(pools["independent20"],X,Y,bounds,
            targets,coefficients,cfg,seconds=EXTRA_SECONDS,rule="semantic")
        for label,internal in (("whole25",False),("context25",True)):
            pools[label],counts[label] = transfer(pools["independent20"],X,Y,bounds,targets,
                coefficients,cfg,seconds=EXTRA_SECONDS,internal=internal)
        pools["joint25"],_,_,counts["joint25"] = search(X,Y,bounds,units,targets,coefficients,cfg25)
        result["unique_search_cpu_seconds"] = sum(c["search_cpu_seconds"] for c in counts.values())
        result["fitting_cpu_seconds"] = time.process_time()-cpu
        archives,choices,models = {},{},{}
        for label in METHODS:
            selection = time.process_time()
            if label == "joint25":
                archives[label],indices,candidate = select_joint(pools[label],split,bounds,targets,cfg25)
                origins = ["joint"]*len(targets)
            else:
                archives[label] = shared.validate_archives(pools[label],split["X_val"],split["Y_val"],
                    split["target_scale"],bounds,targets,cfg.interval_max_boxes)
                indices = shared.select_indices(archives[label],"compact")
                candidate = shared.combine(archives[label],indices)
                origins = [archives[label][j][i]["origin"] for j,i in enumerate(indices)]
            model = shared.inference_model(candidate,cfg if label=="independent20" else cfg25,bounds,
                split["target_scale"],split["feature_names"],split["target_names"],units,targets,coefficients)
            file = label+".json"
            model.save(out/file)
            restored = MIMOSymbolicRegressor.load(out/file)
            np.testing.assert_array_equal(restored.predict(split["X_val"]),model.predict(split["X_val"]))
            connections = np.count_nonzero(model.coef_,axis=0)
            training = counts[label]["search_cpu_seconds"]
            if label in ("semantic25","whole25","context25"):
                training += counts["independent20"]["search_cpu_seconds"]
            choices[label] = dict(indices=indices,model=file,model_sha256=sha(out/file),
                complexity=model.model_["complexity"],shared_basis=int(np.count_nonzero(connections>1)),
                private_basis=int(np.count_nonzero(connections==1)),origins=origins,
                selected_transfer_heads=sum(o not in ("independent","joint") for o in origins),
                selected_internal_heads=origins.count("internal"),
                training_cpu_seconds=training,selection_cpu_seconds=time.process_time()-selection)
            models[label] = model
        save(out/"archives.json",archives)
        save(out/"CHOICES.json",choices)
        evaluated = {}
        observed = np.vstack((X,split["X_val"],split["X_eval"]))
        observed_bounds = np.column_stack((observed.min(axis=0),observed.max(axis=0)))
        for label,model in models.items():
            entry = dict(choices[label],status="ok")
            try:
                for part,suffix in (("train",""),("inner","_val"),("outer","_eval")):
                    entry[part] = v3.metrics(split["Y"+suffix],model.predict(split["X"+suffix]),split["target_scale"])
                entry["training_domain"] = model.audit_domain()["status"]
                entry["observed_domain"] = model.audit_domain(observed_bounds)["status"]
                entry["constant_outputs"] = np.all(model.coef_==0,axis=1).tolist()
            except ValueError as exc:
                entry.update(status="failed",error=str(exc))
            evaluated[label] = entry
    result.update(variants=evaluated,diagnostics=counts,archives_sha256=sha(out/"archives.json"),
        choices_sha256=sha(out/"CHOICES.json"),status="ok" if all(v["status"]=="ok" for v in evaluated.values()) else "failed",
        total_cpu_seconds=time.process_time()-cpu,total_wall_seconds=time.monotonic()-wall)
    return result


def launch(job):
    out = HERE/"runs"/name(job)
    if out.exists():
        if not (out/"result.json").exists():
            raise FileExistsError(f"Incomplete run preserved: {out}")
        r = json.loads((out/"result.json").read_text())
        if r["freeze_sha256"] != sha(HERE/"FREEZE.json"):
            raise ValueError("Existing result belongs to another freeze.")
        if r["status"] == "ok":
            for file,key in (("archives.json","archives_sha256"),("CHOICES.json","choices_sha256")):
                assert sha(out/file) == r[key]
            for v in r["variants"].values():
                assert sha(out/v["model"]) == v["model_sha256"]
        return r["status"]
    out.mkdir(parents=True)
    save(out/"started.json",dict(job=job,freeze_sha256=sha(HERE/"FREEZE.json"),time_utc=datetime.now(timezone.utc).isoformat()))
    with (out/"process.log").open("x",encoding="utf-8") as log:
        try:
            child = subprocess.run([sys.executable,str(Path(__file__).resolve()),"job",*map(str,job)],
                stdout=log,stderr=subprocess.STDOUT,timeout=300,
                env=dict(os.environ,**THREAD_ENV,PYTHONHASHSEED="0",PYTHONUTF8="1"))
            r = json.loads((out/"child.json").read_text()) if (out/"child.json").exists() else dict(status="failed",error="No child result")
            if child.returncode:
                r.update(status="failed",returncode=child.returncode)
        except subprocess.TimeoutExpired:
            r = dict(status="timeout",error="300-second wall timeout")
        except OSError as exc:
            r = dict(status="failed",error=str(exc))
    r.update(dataset=job[0],fold=job[1],seed=job[2],freeze_sha256=sha(HERE/"FREEZE.json"))
    save(out/"result.json",r)
    return r["status"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("prepare","batch","job"))
    p.add_argument("identity",nargs="*")
    p.add_argument("--workers",type=int,default=4)
    args = p.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "job":
        d,f,s = args.identity
        job = d,int(f),int(s)
        assert job in jobs()
        try:
            result = fit_job(job,HERE/"runs"/name(job))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result = dict(status="failed",error=f"{type(exc).__name__}: {exc}")
        save(HERE/"runs"/name(job)/"child.json",result)
        return int(result["status"] != "ok")
    else:
        check_freeze()
        if args.workers not in range(1,5):
            p.error("workers must be between 1 and 4")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(launch,job):job for job in jobs()}
            statuses = []
            for future in as_completed(pending):
                status = future.result();statuses.append(status)
                print(name(pending[future]),status,flush=True)
        print(f"Finished {len(statuses)} cells; failures={sum(s!='ok' for s in statuses)}",flush=True)
        return int(any(s != "ok" for s in statuses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
