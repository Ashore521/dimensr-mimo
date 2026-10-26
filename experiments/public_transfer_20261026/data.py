"""Fixed public-data preparation; fitting has no test-cache access."""
import csv
import hashlib
import io
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT/'outputs/public_transfer_20261026'
CASES=('energy','bike_daily')
SEEDS=(401,402,403)
SOURCES={
    'energy':dict(id=242,url='https://archive.ics.uci.edu/static/public/242/energy%2Befficiency.zip',
                  member='ENB2012_data.xlsx',doi='10.24432/C51307',archive='uci242.zip'),
    'bike_daily':dict(id=275,url='https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip',
                     member='day.csv',doi='10.24432/C5W894',archive='uci275.zip')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def raw(name):
    source=SOURCES[name]
    with zipfile.ZipFile(CACHE/'raw'/source['archive']) as z:payload=z.read(source['member'])
    if name=='energy':
        ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        with zipfile.ZipFile(io.BytesIO(payload)) as book:
            strings=[''.join(e.itertext()) for e in ET.fromstring(book.read('xl/sharedStrings.xml')).findall('m:si',ns)]
            sheet=ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
        rows=[]
        for row in sheet.findall('.//m:sheetData/m:row',ns):
            values=[]
            for c in row.findall('m:c',ns):
                v=c.find('m:v',ns)
                if v is not None:values.append(strings[int(v.text)] if c.get('t')=='s' else float(v.text))
            if values:rows.append(values)
        assert rows[0]==['X'+str(i) for i in range(1,9)]+['Y1','Y2']
        array=np.asarray(rows[1:],float);assert array.shape==(768,10)
        X,Y=array[:,:8],array[:,8:];names=rows[0][:8];targets=['Heating Load','Cooling Load']
        _,groups=np.unique(X[:,:5],axis=0,return_inverse=True);assert len(np.unique(groups))==12
        order=np.random.default_rng(20261027).permutation(12)
        ids={k:np.flatnonzero(np.isin(groups,g)) for k,g in zip(('train','validation','test'),(order[:7],order[7:9],order[9:]))}
        metadata=dict(grouping='exact first five input features',group_ids=groups.tolist(),groups_by_split={k:np.unique(groups[v]).tolist() for k,v in ids.items()})
    else:
        rows=sorted(csv.DictReader(io.StringIO(payload.decode('utf-8'))),key=lambda r:r['dteday'])
        assert len(rows)==731 and len({r['dteday'] for r in rows})==731
        names=['season','yr','mnth','holiday','weekday','workingday','weathersit','temp','atemp','hum','windspeed']
        targets=['casual','registered']
        X=np.asarray([[float(r[n]) for n in names] for r in rows]);Y=np.asarray([[float(r[n]) for n in targets] for r in rows])
        assert all(float(r['cnt'])==float(r['casual'])+float(r['registered']) for r in rows)
        ids=dict(train=np.arange(438),validation=np.arange(438,584),test=np.arange(584,731))
        metadata=dict(grouping='chronological daily blocks',dates_by_split={k:[rows[v[0]]['dteday'],rows[v[-1]]['dteday']] for k,v in ids.items()})
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    assert len(set().union(*[set(v) for v in ids.values()]))==len(X)==sum(len(v) for v in ids.values())
    return X,Y,ids,dict(source,member_sha256=hashlib.sha256(payload).hexdigest(),feature_names=names,target_names=targets,license='CC BY 4.0',**metadata)


def prepare():
    if (HERE/'DATA_MANIFEST.json').exists():raise RuntimeError('Dataset already prepared; do not overwrite.')
    CACHE.mkdir(parents=True,exist_ok=True)
    records={}
    for name in CASES:
        X,Y,ids,meta=raw(name);fit=ids['train']
        mean=X[fit].mean(axis=0);scale=X[fit].std(axis=0);scale=np.where(scale>0,scale,1.)
        target_scale=Y[fit].std(axis=0);assert np.all(target_scale>1e-12)
        transformed=(X-mean)/scale
        development=CACHE/(name+'_development.npz');sealed=CACHE/(name+'_sealed.npz')
        if development.exists() or sealed.exists():raise RuntimeError('Data caches already exist.')
        np.savez_compressed(development,X=transformed[fit],Y=Y[fit],X_val=transformed[ids['validation']],Y_val=Y[ids['validation']])
        np.savez_compressed(sealed,X=transformed[ids['test']],Y=Y[ids['test']])
        meta.update(archive_sha256=sha(CACHE/'raw'/meta['archive']),rows=len(X),inputs=X.shape[1],outputs=2,
            row_ids={k:v.tolist() for k,v in ids.items()},row_counts={k:len(v) for k,v in ids.items()},
            input_mean=mean.tolist(),input_scale=scale.tolist(),target_scale=target_scale.tolist(),
            bounds=np.column_stack((transformed[fit].min(axis=0),transformed[fit].max(axis=0))).tolist(),
            development=development.relative_to(ROOT).as_posix(),development_sha256=sha(development),
            sealed=sealed.relative_to(ROOT).as_posix(),sealed_sha256=sha(sealed))
        records[name]=meta
    write(HERE/'DATA_MANIFEST.json',dict(prepared_unix=time.time(),test_used_for_scaling_selection=False,datasets=records))
    print(json.dumps({name:{k:r[k] for k in ('rows','inputs','outputs','row_counts')} for name,r in records.items()},indent=2))


def manifest():return read(HERE/'DATA_MANIFEST.json')['datasets']


def generate(name,split,seed):
    if name not in CASES or seed not in SEEDS or split not in ('train','validation','test'):raise ValueError('Outside fixed matrix.')
    meta=manifest()[name];test=split=='test'
    if test:
        freeze=HERE/'ALL_MODELS_FROZEN.json'
        if not freeze.exists():raise RuntimeError('All models must be frozen before test access.')
    path=ROOT/meta['sealed' if test else 'development']
    if sha(path)!=meta['sealed_sha256' if test else 'development_sha256']:raise RuntimeError('Dataset cache hash mismatch.')
    with np.load(path,allow_pickle=False) as a:
        return (a['X'].copy(),a['Y'].copy()) if split!='validation' else (a['X_val'].copy(),a['Y_val'].copy())


def training_data(name,split,seed):
    if split not in ('train','validation'):raise ValueError('Training cannot access test cache.')
    return generate(name,split,seed)


def design(name):
    meta=manifest()[name];zero=[0]*7
    return dict(bounds=meta['bounds'],features=[zero]*meta['inputs'],targets=[zero]*2,constants=[zero])


if __name__=='__main__':prepare()
