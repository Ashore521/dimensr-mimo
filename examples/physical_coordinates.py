"""Train-only affine coordinates and an audited original-variable delivery.

Run with one numerical-library thread. This is a known synthetic API example,
not independent evidence of performance on unseen physical systems.
"""
from pathlib import Path
import numpy as np
import torch
from dimensr_mimo import AffineCoordinates,PhysicalModel,DimenSRMIMORegressor,SearchConfig

L=[1,0,0,0,0,0,0]
ZERO=[0]*7
BOUNDS=[[8.,12.],[90.,110.]]
SEARCH_LIMITS=dict(nrmse=[1e-4]*2,shared_graph_size=24,independent_parameters=6,
                   expanded_total_nodes=30,maximum_output_depth=6)
# Freeze a separate limit for the delivered graph, including fixed coordinates.
PHYSICAL_LIMITS=dict(nrmse=[1e-4]*2,shared_graph_size=48,independent_parameters=14,
                     expanded_total_nodes=64,maximum_output_depth=10)


def data():
    rng=np.random.default_rng(2601)
    X=rng.uniform(np.asarray(BOUNDS)[:,0],np.asarray(BOUNDS)[:,1],(144,2))
    Y=np.column_stack((2*X[:,0]+3*X[:,1]+5,X[:,0]-X[:,1]+2))
    return X[:96],Y[:96],X[96:],Y[96:]


def configuration():
    return SearchConfig(random_state=601,population_size=24,epochs=4,gp_generations=2,
        selection_generations=4,max_evaluations=1200,min_candidates=30,max_nodes=15,max_depth=6,
        hidden_size=32,constant_restarts=1,constant_max_nfev=25,polish_candidates=3,
        polish_max_nfev=50,rademacher_draws=16,perturbations=8,wall_seconds=52,
        operators=('add','sub','mul','div','square','sin','cos','sqrt','log'),
        initialization='dimensional',final_selection='precision',acceptance=SEARCH_LIMITS)


def fit_example(folder,callback=None):
    torch.set_num_threads(1)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'physical.json').exists():raise RuntimeError('Use a new output directory.')
    X,Y,V,Z=data()
    coordinates=AffineCoordinates.from_training(X,Y)
    model=DimenSRMIMORegressor(configuration(),[L,L],[L,L],[ZERO,L],['length0','length1'])
    model.fit(coordinates.inputs(X),coordinates.outputs(Y),X_val=coordinates.inputs(V),
        Y_val=coordinates.outputs(Z),input_bounds=coordinates.input_bounds(BOUNDS),callback=callback)
    physical=PhysicalModel.from_estimator(model,coordinates,BOUNDS,PHYSICAL_LIMITS,
        training_data=(X,Y),validation_data=(V,Z))
    model.save(folder/'model.json');physical.save(folder/'physical.json')
    loaded=PhysicalModel.load(folder/'physical.json')
    for A in (X,V):
        np.testing.assert_array_equal(physical.predict(A),coordinates.restore_outputs(model.predict(coordinates.inputs(A))))
        np.testing.assert_array_equal(physical.predict(A),loaded.predict(A))
    return model,physical


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/physical-example')
    model,physical=fit_example(parser.parse_args().output)
    print(physical.formulas)
    print(physical.acceptance)
    print(physical.structure)
