"""Audit complete paired records and apply the prespecified advancement gate."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import statistics

from run import DATASETS, HERE, ROOT, jobs, name, sha, check_freeze

METHODS = ("joint","independent_penalty","independent_compact","hybrid_penalty","hybrid_compact")
PAIRS = (("hybrid_compact","independent_compact"),("hybrid_penalty","independent_penalty"),
         ("hybrid_compact","hybrid_penalty"),("independent_compact","independent_penalty"),
         ("joint","independent_compact"),("independent_penalty","independent_compact"),
         ("hybrid_penalty","independent_compact"))


def stats(values):
    return dict(mean=statistics.mean(values),sd=statistics.pstdev(values),min=min(values),max=max(values))


def aggregate(rows, expected):
    good = [r for r in rows if r["status"]=="ok"]
    out = dict(expected=expected,successful=len(good),complete=len(good)==expected)
    if out["complete"]:
        fields = set.intersection(*(set(r) for r in good))-{"fold","seed"}
        out["metrics"] = {k:stats([r[k] for r in good]) for k in sorted(fields) if isinstance(good[0][k],(int,float))}
    return out


def gate(groups):
    candidate,control,joint = (groups[m] for m in ("hybrid_compact","independent_compact","joint"))
    out = dict(eligible=False,complete=all(g["complete"] for g in (candidate,control,joint)))
    if not out["complete"]:
        return out
    def at_most(a,b):
        return a <= b + 1e-12*max(1.,abs(b))
    cm,bm,jm = (g["metrics"] for g in (candidate,control,joint))
    ce,be,ck,bk = cm["outer_nrmse"]["mean"],bm["outer_nrmse"]["mean"],cm["complexity"]["mean"],bm["complexity"]["mean"]
    guards = []
    for d in DATASETS:
        c,b = candidate["datasets"][d]["metrics"],control["datasets"][d]["metrics"]
        for key in sorted(k for k in b if k.startswith("outer_nrmse_")):
            old,new = b[key]["mean"],c[key]["mean"]
            guards.append(dict(dataset=d,output=int(key.rsplit("_",1)[1]),baseline=old,candidate=new,
                               allowed_increase=max(.01,.02*old),passed=at_most(new-old,max(.01,.02*old))))
    accuracy = at_most(ce,.99*be) and at_most(ck,1.1*bk)
    compression = at_most(ck,.9*bk) and at_most(ce,1.005*be)
    no_worse_joint = at_most(ce,jm["outer_nrmse"]["mean"])
    out.update(accuracy_route=accuracy,compression_route=compression,output_guards=guards,
        no_worse_joint=no_worse_joint,relative_error_change_pct=100*(ce/be-1) if be else None,
        relative_complexity_change_pct=100*(ck/bk-1) if bk else None,
        eligible=bool((accuracy or compression) and no_worse_joint and all(g["passed"] for g in guards)))
    return out


def pair(rows,left,right):
    maps = [{(r["dataset"],r["fold"],r["seed"]):r for r in rows if r["method"]==m} for m in (left,right)]
    out = []
    for d,f,s in product(DATASETS,range(3),range(5)):
        a,b = (g.get((d,f,s)) for g in maps)
        row = dict(dataset=d,fold=f,seed=s,left=left,right=right,status="incomplete")
        if a and b and a["status"]==b["status"]=="ok":
            row["status"]="ok"
            for k in sorted(a.keys()&b.keys()):
                if k=="complexity" or k.startswith(("train_","inner_","outer_")):
                    row["delta_"+k]=a[k]-b[k]
        out.append(row)
    return out


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def analyze(out):
    frozen=check_freeze()
    expected={name(j) for j in jobs()}
    paths={p.parent.name:p for p in (HERE/"runs").glob("*/result.json")}
    if expected!=set(paths):
        raise ValueError(f"Incomplete grid: missing={sorted(expected-set(paths))}, extra={sorted(set(paths)-expected)}")
    splits=json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    rows,records,digests=[],[],{}
    for job in jobs():
        path=paths[name(job)];r=json.loads(path.read_text())
        assert tuple(r[k] for k in ("dataset","arm","fold","seed"))==job
        assert r["freeze_sha256"]==sha(HERE/"FREEZE.json")
        digests[path.relative_to(HERE).as_posix()]=sha(path)
        records.append(r)
        labels=("joint",) if r["arm"]=="joint" else (r["arm"]+"_penalty",r["arm"]+"_compact")
        if r["status"]=="ok":
            assert r["fold_sha256"]==splits[f'{r["dataset"]}_f{r["fold"]}']["fold_sha256"]
            assert sha(path.parent/"archives.json")==r["archives_sha256"]
            assert sha(path.parent/"CHOICES.json")==r["choices_sha256"]
            choices=json.loads((path.parent/"CHOICES.json").read_text())
        for label in labels:
            row={k:r[k] for k in ("dataset","fold","seed")}
            row.update(method=label,status=r["status"])
            if r["status"]=="ok":
                v=r["variants"][label]
                assert all(v[k]==value for k,value in choices[label].items())
                assert sha(path.parent/v["model"])==v["model_sha256"]
                row["status"]=v["status"]
                for field in ("complexity","shared_basis","private_basis"):
                    row[field]=v[field]
                for field in ("fit_cpu_seconds","search_cpu_seconds","overshoot_seconds","selection_cpu_seconds","score_cpu_seconds","total_cpu_seconds","total_wall_seconds"):
                    row[field]=r[field]
                for part in ("train","inner","outer"):
                    row[part+"_nrmse"]=v[part]["mean_nrmse"]
                    row[part+"_worst_nrmse"]=v[part]["worst_nrmse"]
                    for metric in ("nrmse","r2"):
                        for j,value in enumerate(v[part][metric]):
                            row[f"{part}_{metric}_{j}"]=value
                for j,value in enumerate(v["connections_per_output"]):
                    row[f"connections_{j}"]=value
            rows.append(row)
    groups={}
    for method in METHODS:
        part=[r for r in rows if r["method"]==method]
        groups[method]=dict(aggregate(part,45),datasets={d:aggregate([r for r in part if r["dataset"]==d],15) for d in DATASETS})
    pairs=[r for left,right in PAIRS for r in pair(rows,left,right)]
    paired=[]
    for left,right in PAIRS:
        for d in (*DATASETS,"all"):
            part=[r for r in pairs if r["left"]==left and r["right"]==right and (d=="all" or r["dataset"]==d)]
            row=dict(left=left,right=right,dataset=d,**aggregate(part,45 if d=="all" else 15))
            if row["complete"]:
                row["lower_error_cells"]=sum(r["delta_outer_nrmse"]<0 for r in part)
            paired.append(row)
    summary=dict(source_commit=frozen["source_commit"],freeze_sha256=sha(HERE/"FREEZE.json"),
        training_runs=135,successful_training_runs=sum(r["status"]=="ok" for r in records),
        evaluations=len(rows),successful_evaluations=sum(r["status"]=="ok" for r in rows),
        result_sha256=digests,methods=groups,paired=paired,advancement_gate=gate(groups),
        analysis_source_sha256=sha(__file__))
    out.mkdir(exist_ok=False)
    write_csv(out/"scores.csv",rows);write_csv(out/"paired.csv",pairs)
    lines=["# Shared/private development results","","All means require all 15 fold/seed cells per dataset; lower NRMSE is better.","",
        "| Dataset | Method | Train | Inner | Outer | K | Shared bases | Private bases | Search CPU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d,method in product((*DATASETS,"all"),METHODS):
        group=groups[method] if d=="all" else groups[method]["datasets"][d]
        values=" | ".join(f'{group["metrics"][k]["mean"]:.6f}' for k in ("train_nrmse","inner_nrmse","outer_nrmse","complexity","shared_basis","private_basis","search_cpu_seconds")) if group["complete"] else "incomplete | | | | | |"
        lines.append(f"| {d} | {method} | {values} |")
    (out/"TABLES.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    summary["table_sha256"]={p.name:sha(p) for p in out.iterdir()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f'{summary["successful_training_runs"]}/135 training runs; {summary["successful_evaluations"]}/225 scores; advance={summary["advancement_gate"]["eligible"]}')


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"analysis")
    analyze(parser.parse_args().output)
