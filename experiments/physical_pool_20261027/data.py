"""Frozen historical pools and physical arrays; test access requires a new seal."""
from pathlib import Path
import numpy as np
from dimensr_mimo import AffineCoordinates
from experiments.compound_search_20261027 import cases as synthetic
from experiments.slump_physical_20261026 import data as slump
from examples import physical_coordinates as affine
from experiments.native_api_20261026.run import read,write,sha

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def jobs():
    result=[]
    for case in synthetic.CASES:
        for seed in synthetic.seeds(case):
            directory='mutation_units_20261027' if case in ('polynomial_n0','polynomial_n1') else 'compound_search_20261027'
            source=f'experiments/{directory}/runs/{case}_s{seed}_compound/model.json'
            result.append(dict(case=case,seed=seed,method='pool',kind='synthetic',source=source,seconds=60))
    for seed in (801,802,803):
        for method in ('legacy','dimensional'):
            source=f'experiments/reference_budget_20261027/runs/slump_s{seed}_{method}/model.json'
            result.append(dict(case='slump_'+method,seed=seed,method='pool',kind='slump',source=source,seconds=60))
    result.append(dict(case='affine_lengths',seed=2701,method='pool',kind='affine',
        source='experiments/native_api_20261026/runs/affine_lengths_physical/model.json',seconds=60))
    return result


def name(job):return f"{job['case']}_s{job['seed']}_{job['method']}"


def spec(job):
    if job['kind']=='synthetic':
        d=synthetic.design(job['case'])
        return dict(bounds=d['bounds'],coordinates=AffineCoordinates.identity(len(d['features']),len(d['targets'])).to_dict(),
            limits=d['acceptance'],clean_limits=d['clean_test_nrmse'])
    if job['kind']=='slump':
        m=read(ROOT/'experiments/slump_physical_20261026/DATA_MANIFEST.json')
        return dict(bounds=m['physical_bounds'],coordinates=m['coordinates'],
            limits=read(ROOT/'experiments/reference_budget_20261027/FREEZE.json')['physical_limits'],clean_limits=None)
    X,Y,_,_=affine.data()
    return dict(bounds=affine.BOUNDS,coordinates=AffineCoordinates.from_training(X,Y).to_dict(),
        limits=affine.PHYSICAL_LIMITS,clean_limits=[1e-4]*2)


def arrays(job,split):
    if split not in ('train','validation','test'):raise ValueError('Unknown split.')
    if split=='test' and not (HERE/'ALL_MODELS_FROZEN.json').exists():
        raise RuntimeError('Freeze all derived models before test access.')
    if job['kind']=='synthetic':return synthetic.generate(job['case'],split,job['seed'])
    if job['kind']=='slump':return slump.load('val' if split=='validation' else split)
    if split!='test':
        X,Y,V,Z=affine.data();return (X,Y) if split=='train' else (V,Z)
    rng=np.random.default_rng(2701);b=np.asarray(affine.BOUNDS)
    X=rng.uniform(b[:,0],b[:,1],(192,2))
    return X,clean(job,X)


def clean(job,X):
    if job['kind']=='synthetic':return synthetic.latent(job['case'],X)
    if job['kind']=='affine':return np.column_stack((2*X[:,0]+3*X[:,1]+5,X[:,0]-X[:,1]+2))
    return None
