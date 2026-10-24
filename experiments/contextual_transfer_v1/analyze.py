"""Verify the complete frozen grid and report both advancement and attribution."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import statistics

from dimensr_mimo.regressor import _serial_candidate
from context_study import DATASETS, METHODS, HERE, ROOT, jobs, name, sha, check_freeze, shared

PAIRS = (("context25","independent25"),("context25","whole25"),
    ("context25","semantic25"),("context25","joint25"),("context25","independent20"))


def aggregate(rows, expected):
    good = [r for r in rows if r["status"]=="ok"]
    out = dict(expected=expected,successful=len(good),complete=len(good)==expected)
    if out["complete"]:
        fields = set.intersection(*(set(r) for r in good)) - {"fold","seed"}
        out["metrics"] = {}
        for k in sorted(fields):
            if all(isinstance(r[k],(int,float)) for r in good):
                v = [r[k] for r in good]
                out["metrics"][k] = dict(mean=statistics.mean(v),sd=statistics.pstdev(v),min=min(v),max=max(v))
    return out


def gate(groups, control="independent25"):
    candidate,baseline,base = (groups[m] for m in ("context25",control,"independent20"))
    out = dict(control=control,eligible=False,complete=all(g["complete"] for g in (candidate,baseline,base)))
    if not out["complete"]:
        return out
    def at_most(a,b):
        return a <= b+1e-12*max(1.,abs(b))
    cm,bm = candidate["metrics"],baseline["metrics"]
    ce,be,ck,bk = cm["outer_nrmse"]["mean"],bm["outer_nrmse"]["mean"],cm["complexity"]["mean"],bm["complexity"]["mean"]
    guards = []
    for d in DATASETS:
        c,b = candidate["datasets"][d]["metrics"],baseline["datasets"][d]["metrics"]
        for key in sorted(k for k in b if k.startswith("outer_nrmse_")):
            old,new = b[key]["mean"],c[key]["mean"]
            guards.append(dict(dataset=d,output=int(key.rsplit("_",1)[1]),baseline=old,candidate=new,
                allowed_increase=max(.01,.02*old),passed=at_most(new-old,max(.01,.02*old))))
    accuracy = at_most(ce,.99*be) and at_most(ck,1.1*bk)
    compression = at_most(ck,.9*bk) and at_most(ce,1.005*be)
    no_worse_base = at_most(ce,base["metrics"]["outer_nrmse"]["mean"])
    out.update(accuracy_route=accuracy,compression_route=compression,output_guards=guards,
        no_worse_base=no_worse_base,relative_error_change_pct=100*(ce/be-1) if be else None,
        relative_complexity_change_pct=100*(ck/bk-1) if bk else None,
        eligible=bool((accuracy or compression) and no_worse_base and all(g["passed"] for g in guards)))
    return out


def paired(rows, left, right):
    lookup = [{(r["dataset"],r["fold"],r["seed"]):r for r in rows if r["method"]==m} for m in (left,right)]
    result = []
    for key in jobs():
        a,b = (p.get(key) for p in lookup)
        row = dict(dataset=key[0],fold=key[1],seed=key[2],left=left,right=right,status="incomplete")
        if a and b and a["status"]==b["status"]=="ok":
            row["status"] = "ok"
            for k in sorted(a.keys()&b.keys()):
                if k=="complexity" or k.startswith(("train_nrmse","inner_nrmse","outer_nrmse")):
                    row["delta_"+k] = a[k]-b[k]
        result.append(row)
    return result


def write_csv(path, rows):
    with path.open("x",encoding="utf-8",newline="") as stream:
        w = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def analyze(out):
    frozen = check_freeze()
    paths = {p.parent.name:p for p in (HERE/"runs").glob("*/result.json")}
    expected = {name(j) for j in jobs()}
    if set(paths) != expected:
        raise ValueError(f"Incomplete grid: missing={sorted(expected-set(paths))}, extra={sorted(set(paths)-expected)}")
    splits = json.loads((ROOT/"experiments/budget_capacity_v1/SPLITS.json").read_text())
    rows,records,digests = [],[],{}
    for job in jobs():
        path = paths[name(job)];r = json.loads(path.read_text())
        assert tuple(r[k] for k in ("dataset","fold","seed")) == job
        assert r["freeze_sha256"] == sha(HERE/"FREEZE.json")
        records.append(r)
        digests[path.relative_to(HERE).as_posix()] = sha(path)
        if r["status"] == "ok":
            assert r["fold_sha256"] == splits[f"{job[0]}_f{job[1]}"]["fold_sha256"]
            for file,key in (("archives.json","archives_sha256"),("CHOICES.json","choices_sha256")):
                assert sha(path.parent/file) == r[key]
            choices = json.loads((path.parent/"CHOICES.json").read_text())
            archives = json.loads((path.parent/"archives.json").read_text())
            assert set(choices) == set(archives) == set(METHODS)
            for m in ("semantic25","whole25","context25"):
                for base,extended in zip(archives["independent20"],archives[m]):
                    assert extended[:len(base)] == base
        for method in METHODS:
            row = dict(dataset=job[0],fold=job[1],seed=job[2],method=method,status=r["status"])
            if r["status"] == "ok":
                v = r["variants"][method]
                assert all(v[k]==value for k,value in choices[method].items())
                assert sha(path.parent/v["model"]) == v["model_sha256"]
                if method == "joint25":
                    a = archives[method]
                    selected = min(range(len(a)),key=lambda i:(a[i]["validation_loss"]+
                        r["config"]["complexity_weight"]*a[i]["complexity"],a[i]["complexity"]))
                    assert v["indices"] == [selected]
                    c = shared.decode(a[selected])
                else:
                    assert v["indices"] == shared.select_indices(archives[method],"compact")
                    c = shared.combine(archives[method],v["indices"])
                assert _serial_candidate(c) == json.loads((path.parent/v["model"]).read_text())["model"]
                row["status"] = v["status"]
                for k in ("complexity","shared_basis","private_basis","selected_transfer_heads",
                    "selected_internal_heads","training_cpu_seconds","selection_cpu_seconds"):
                    row[k] = v[k]
                if v["status"] == "ok":
                    assert v["training_domain"] == "enclosed"
                    row["observed_domain"] = v["observed_domain"]
                    row["constant_output_count"] = sum(v["constant_outputs"])
                    for part in ("train","inner","outer"):
                        row[part+"_nrmse"] = v[part]["mean_nrmse"]
                        for metric in ("nrmse","r2"):
                            for j,value in enumerate(v[part][metric]):
                                row[f"{part}_{metric}_{j}"] = value
            rows.append(row)
    groups = {m:dict(aggregate([r for r in rows if r["method"]==m],45),
        datasets={d:aggregate([r for r in rows if r["method"]==m and r["dataset"]==d],15) for d in DATASETS}) for m in METHODS}
    pair_rows = [r for a,b in PAIRS for r in paired(rows,a,b)]
    pairs = []
    for a,b in PAIRS:
        for d in (*DATASETS,"all"):
            part = [r for r in pair_rows if r["left"]==a and r["right"]==b and (d=="all" or r["dataset"]==d)]
            entry = dict(left=a,right=b,dataset=d,**aggregate(part,45 if d=="all" else 15))
            if entry["complete"]:
                entry["lower_error_cells"] = sum(r["delta_outer_nrmse"]<0 for r in part)
            pairs.append(entry)
    summary = dict(freeze_sha256=sha(HERE/"FREEZE.json"),source_commit=frozen["source_commit"],
        cells=45,successful_cells=sum(r["status"]=="ok" for r in records),evaluations=len(rows),
        successful_evaluations=sum(r["status"]=="ok" for r in rows),result_sha256=digests,
        methods=groups,paired=pairs,advancement_gate=gate(groups),internal_attribution_gate=gate(groups,"whole25"))
    out.mkdir(exist_ok=False)
    write_csv(out/"scores.csv",rows);write_csv(out/"paired.csv",pair_rows)
    lines = ["# Contextual-transfer development results","","Descriptive means; all cells required. Lower NRMSE is better.","",
        "| Dataset | Method | Train | Inner | Outer | K | Shared bases | Internal heads | Training CPU |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for d,m in product((*DATASETS,"all"),METHODS):
        g = groups[m] if d=="all" else groups[m]["datasets"][d]
        values = " | ".join(f'{g["metrics"][k]["mean"]:.6f}' for k in
            ("train_nrmse","inner_nrmse","outer_nrmse","complexity","shared_basis","selected_internal_heads","training_cpu_seconds")) if g["complete"] else "incomplete"
        lines.append(f"| {d} | {m} | {values} |")
    (out/"TABLES.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    summary["table_sha256"] = {p.name:sha(p) for p in sorted(out.iterdir())}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("advancement_gate","internal_attribution_gate")},indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=HERE/"analysis")
    analyze(parser.parse_args().output)
