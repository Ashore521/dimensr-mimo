"""Fit the optional semantic-transfer prototype on a known synthetic system."""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import SearchConfig, MIMOSymbolicRegressor
from dimensr_mimo._vendor.units import ZERO
from transfer import independent, transfer, shared


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seconds",type=float,default=10.)
    p.add_argument("--output",type=Path,default=Path("outputs/semantic-transfer-demo.json"))
    args=p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:
        p.error("seconds must be positive and finite")
    if args.output.exists():
        p.error("Output exists; choose a new filename.")
    rng=np.random.default_rng(42)
    X=rng.uniform(-1,1,(320,3))
    Y=np.column_stack((2*np.sin(X[:,0])+X[:,1]**2,-np.sin(X[:,0])+3*X[:,2]))
    cfg=SearchConfig(max_features=192,generations=10000,proposals_per_generation=128,
        max_terms=8,polish_max_nfev=10,head_ridge=.001,search_seconds=.8*args.seconds,random_state=7)
    bounds=np.array([[-1.,1.]]*3);units=(ZERO,)*3;targets=(ZERO,)*2;coefficients=(ZERO,)
    with threadpool_limits(limits=1):
        pools,base=independent(X[:240],Y[:240],bounds,units,targets,coefficients,cfg)
        pools,extra=transfer(pools,X[:240],Y[:240],bounds,targets,coefficients,cfg,
            seconds=.2*args.seconds,rule="semantic")
        scales=shared.scales_of(Y[:240])
        archives=shared.validate_archives(pools,X[240:],Y[240:],scales,bounds,targets,32)
        indices=shared.select_indices(archives,"compact")
        model=shared.inference_model(shared.combine(archives,indices),replace(cfg,search_seconds=args.seconds),
            bounds,scales,["x0","x1","x2"],["y0","y1"],units,targets,coefficients)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        model.save(args.output)
        restored=MIMOSymbolicRegressor.load(args.output)
        np.testing.assert_array_equal(restored.predict(X[240:]),model.predict(X[240:]))
        error=np.sqrt(np.mean(((model.predict(X[240:])-Y[240:])/scales)**2,axis=0))
    print("Synthetic validation NRMSE:",error.tolist())
    print("Actual K:",model.model_["complexity"])
    print("Training CPU:",base["search_cpu_seconds"]+extra["search_cpu_seconds"])
    print("Accepted training transfers:",extra["accepted"],"saved/reloaded:",args.output)


if __name__=="__main__":
    main()
