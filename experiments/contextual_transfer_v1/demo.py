"""Seeded structural checks on new synthetic rows; not blind discovery evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import SearchConfig, MIMOSymbolicRegressor
from dimensr_mimo.features import Feature, compose, evaluate_features
from dimensr_mimo.search import make_candidate
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo._vendor.units import ZERO
from contextual import transfer, shared


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    cfg = SearchConfig(max_terms=2,head_ridge=0.,random_state=1)
    bounds = np.array([[-1.,1.]]*3)
    X = np.random.default_rng(42).uniform(-1,1,(400,3))
    x0,x1,x2 = (Feature(Node(f"x{i}")) for i in range(3))
    donor = compose("add",x0,x1)
    report = dict(note="Hand-seeded incomplete archives; held-out synthetic rows verify mechanics, not discovery.",cases={})
    with threadpool_limits(limits=1):
        for case in ("internal","partial","none"):
            bases = (compose("sin",x0),)
            Y = np.column_stack((np.sin(X[:,0]+X[:,1]),X[:,0]+X[:,1]))
            second = (donor,)
            if case == "partial":
                Y[:,0] += X[:,2]**2
                bases += (compose("square",x2),)
            elif case == "none":
                Y = np.column_stack((np.sin(X[:,0]),X[:,2]**2))
                second = (compose("square",x2),)
            scales = shared.scales_of(Y[:200])
            pools = [[make_candidate(fs,evaluate_features(fs,X[:200]),Y[:200,j:j+1],scales[j:j+1],
                (ZERO,),(ZERO,))] for j,fs in enumerate((bases,second))]
            selected = {}
            for label,internal in (("whole",False),("context",True)):
                result,counts = transfer(pools,X[:200],Y[:200],bounds,(ZERO,)*2,(ZERO,),cfg,seconds=1.,internal=internal)
                archives = shared.validate_archives(result,X[200:300],Y[200:300],scales,bounds,(ZERO,)*2,32)
                indices = shared.select_indices(archives,"compact")
                model = shared.inference_model(shared.combine(archives,indices),cfg,bounds,scales,
                    ["x0","x1","x2"],["y0","y1"],(ZERO,)*3,(ZERO,)*2,(ZERO,))
                path = out/f"{case}_{label}.json"
                model.save(path)
                restored = MIMOSymbolicRegressor.load(path)
                np.testing.assert_array_equal(restored.predict(X[300:]),model.predict(X[300:]))
                selected[label] = dict(model=path.name,indices=indices,complexity=model.model_["complexity"],
                    test_nrmse=np.sqrt(np.mean(((model.predict(X[300:])-Y[300:])/scales)**2,axis=0)).tolist(),
                    domain=model.audit_domain()["status"],diagnostics=counts)
            report["cases"][case] = selected
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({case:{m:r["test_nrmse"] for m,r in arms.items()} for case,arms in report["cases"].items()},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/contextual-demo"))
    run(p.parse_args().output)
