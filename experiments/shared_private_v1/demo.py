"""Run the experimental shared/private prototype on a known synthetic system."""
import argparse
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import SearchConfig, MIMOSymbolicRegressor
from dimensr_mimo._vendor.units import ZERO
from hybrid import search_hybrid, validate_archives, select_indices, combine, inference_model, scales_of


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds",type=float,default=10.)
    parser.add_argument("--output",type=Path,default=Path("outputs/shared_private_demo.json"))
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new path to preserve the previous model.")
    rng=np.random.default_rng(42)
    X=rng.uniform(-1,1,(320,3))
    Y=np.column_stack((2*np.sin(X[:,0])+X[:,1]**2,-np.sin(X[:,0])+3*X[:,2]))
    train,val=slice(0,240),slice(240,None)
    bounds=np.array([[-1.,1.]]*3)
    cfg=SearchConfig(max_terms=8,max_features=192,generations=10000,search_seconds=args.seconds,
                     proposals_per_generation=128,polish_max_nfev=10,head_ridge=.001,random_state=7)
    cfg.validate()
    with threadpool_limits(limits=1):
        pools,counts=search_hybrid(X[train],Y[train],bounds,(ZERO,)*3,(ZERO,)*2,(ZERO,),cfg)
        scales=scales_of(Y[train])
        archives=validate_archives(pools,X[val],Y[val],scales,bounds,(ZERO,)*2,32)
        indices=select_indices(archives,"compact")
        model=inference_model(combine(archives,indices),cfg,bounds,scales,["x0","x1","x2"],["y0","y1"],
                              (ZERO,)*3,(ZERO,)*2,(ZERO,))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        model.save(args.output)
        restored=MIMOSymbolicRegressor.load(args.output)
        np.testing.assert_array_equal(restored.predict(X[val]),model.predict(X[val]))
        error=np.sqrt(np.mean(((model.predict(X[val])-Y[val])/scales)**2,axis=0))
    print("Synthetic validation NRMSE:",error.tolist())
    print("Actual K:",model.model_["complexity"],"training CPU:",counts["search_cpu_seconds"])
    print("Saved and reloaded:",args.output)


if __name__=="__main__":
    main()
