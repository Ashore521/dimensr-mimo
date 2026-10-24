"""Shared paths and frozen implementations for the external evaluation."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATASETS=('tetouan','water','video')
SYMBOLIC=('independent20','independent25','random25','semantic25','joint25')
METHODS=SYMBOLIC+('mean','ridge','extra_trees')
SEALED=ROOT/'outputs/external_transfer_v1_sealed'
THREAD_ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

transfer=load_module('external_transfer_impl','experiments/semantic_transfer_v1/transfer.py')
shared=transfer.shared
v3=load_module('external_v3_helpers','experiments/real_blind_v3/runner.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def jobs():
    return [(d,s) for d in DATASETS for s in range(5)]

def baseline_predict(model,X):
    return model['estimator'].predict(X)*model['scale']+model['offset']

def name(job):
    return f'{job[0]}_s{job[1]}'

def check_freeze():
    frozen=json.loads((HERE/'FREEZE.json').read_text())
    for file,digest in frozen['files'].items():
        if sha(ROOT/file)!=digest:raise ValueError(f'Frozen file changed: {file}')
    return frozen
