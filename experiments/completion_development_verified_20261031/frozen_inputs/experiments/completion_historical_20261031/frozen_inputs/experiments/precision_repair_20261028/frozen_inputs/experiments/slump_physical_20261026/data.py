"""Reconstruct public historical row splits, convert units, seal holdout arrays."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np
from dimensr_mimo import AffineCoordinates

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CACHE=ROOT/'outputs/slump_physical_20261026'
HISTORICAL=ROOT.parent/'experiments/dimensr_historical_v3/splits/legacy_concrete_slump_None.json'
URL='https://archive.ics.uci.edu/static/public/182/concrete+slump+test.zip'
BOUNDS=[[0.,500.]]*3+[[100.,300.],[0.,50.],[500.,1300.],[400.,1100.]]
FEATURES=['Cement','Slag','Fly ash','Water','SP','Coarse Aggr.','Fine Aggr.']
TARGETS=['SLUMP','FLOW','Compressive Strength (28-day)']
DENSITY=[-3,1,0,0,0,0,0];L=[1,0,0,0,0,0,0];PRESSURE=[-1,1,-2,0,0,0,0]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(('\n'.join(v)+'\n').encode('utf-8')).hexdigest()


def prepare():
    if (HERE/'DATA_MANIFEST.json').exists():raise RuntimeError('Keep frozen preparation.')
    old=read(HISTORICAL);meta=old['dataset_manifest'];CACHE.mkdir(parents=True,exist_ok=True)
    archive=CACHE/'source.zip'
    if not archive.exists():
        with urllib.request.urlopen(URL,timeout=30) as response:archive.write_bytes(response.read())
    if sha(archive)!=meta['zip_sha256']:raise ValueError('Public archive differs from historical manifest.')
    with zipfile.ZipFile(archive) as z:content=z.read('slump_test.data')
    if hashlib.sha256(content).hexdigest()!=meta['member_sha256']:raise ValueError('Public member differs from historical manifest.')
    a=np.genfromtxt(io.BytesIO(content),delimiter=',',skip_header=1)
    if a.shape!=(103,11) or not np.isfinite(a).all() or len(np.unique(a[:,0]))!=103:raise ValueError('Unexpected public data shape, values or IDs.')
    ids=[f'slump_test.data:line:{i+2}' for i in range(len(a))];lookup={v:i for i,v in enumerate(ids)}
    positions={k:[lookup[v] for v in old['row_ids'][k]] for k in ('train','val')}
    assigned=set(positions['train'])|set(positions['val']);positions['test']=[i for i in range(len(a)) if i not in assigned]
    if len(assigned)!=82 or len(positions['test'])!=21:raise ValueError('Historical split overlap or missing rows.')
    # Preserve historical held-out order as well as membership using the old random split.
    order=np.random.default_rng(20261023).permutation(len(a))
    expected={k:meta['split_id_sha256'][k] for k in ('train','val','test')}
    if order[:61].tolist()!=positions['train'] or order[61:82].tolist()!=positions['val']:
        raise ValueError('Historical seeded permutation differs from the saved split.')
    positions['test']=order[82:].tolist()
    for k,rows in positions.items():
        if digest([ids[i] for i in rows])!=expected[k]:raise ValueError('Historical row ID hash differs: '+k)
    X=a[:,1:8];Y=a[:,8:]*np.array([.01,.01,1e6])
    coordinates=AffineCoordinates.from_training(X[positions['train']],Y[positions['train']])
    pairs=[]
    for left,right in (('train','val'),('train','test'),('val','test')):
        lefts={tuple(X[i]) for i in positions[left]};duplicates=sum(tuple(X[i]) in lefts for i in positions[right])
        pairs.append(dict(left=left,right=right,exact_input_duplicates=duplicates))
    for group,parts in (('development',('train','val')),('sealed',('test',))):
        values={name+'_'+part:array[positions[part]] for part in parts for name,array in (('X',X),('Y',Y))}
        np.savez(CACHE/(group+'.npz'),**values)
    # Do not print target statistics or any model/evaluation metric here.
    write(HERE/'DATA_MANIFEST.json',dict(source_url=meta['source_url'],download_url=URL,doi=meta['doi'],license=meta['license'],attribution=meta['attribution'],
        zip_sha256=sha(archive),member_sha256=meta['member_sha256'],historical_split_sha256=sha(HISTORICAL),
        row_ids={k:[ids[i] for i in rows] for k,rows in positions.items()},row_id_sha256=expected,
        rows={k:len(rows) for k,rows in positions.items()},feature_names=FEATURES,target_names=TARGETS,
        feature_units=[DENSITY]*7,target_units=[L,L,PRESSURE],target_si_multipliers=[.01,.01,1e6],
        coordinates=coordinates.to_dict(),physical_bounds=BOUNDS,exact_input_duplicates=pairs,
        files={g+'.npz':sha(CACHE/(g+'.npz')) for g in ('development','sealed')},
        claims='Historical row-level regression; laboratory batch independence unknown; no new blind validation.'))
    print('Prepared 61 training, 21 validation and 21 sealed historical rows; SI units recorded.')


def load(part):
    if part not in ('train','val','test'):raise ValueError('Unknown split.')
    if part=='test' and not (HERE/'ALL_MODELS_FROZEN.json').exists():raise RuntimeError('Holdout access requires all models frozen.')
    manifest=read(HERE/'DATA_MANIFEST.json');filename='sealed.npz' if part=='test' else 'development.npz'
    path=CACHE/filename
    if sha(path)!=manifest['files'][filename]:raise ValueError('Data cache drift.')
    with np.load(path,allow_pickle=False) as f:return f['X_'+part].copy(),f['Y_'+part].copy()


if __name__=='__main__':prepare()
