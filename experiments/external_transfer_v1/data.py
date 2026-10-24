"""Prepare fixed external splits without printing held-out target statistics."""
import csv
from datetime import datetime
import hashlib
import io
import json
from urllib.request import urlopen
import zipfile

import numpy as np
from common import HERE,ROOT,SEALED,DATASETS,sha

SEED=20261025
RAW=ROOT/'outputs/external_transfer_v1_raw'
SOURCES={
 'tetouan':dict(id=849,slug='power+consumption+of+tetouan+city',doi='10.24432/C5B034',
    member='Tetuan City power consumption.csv',zip_sha256='3c4bf684161180937043a9fb65701a83d44b740b0d42f0492b6e7aec57ddbbfe',
    member_sha256='ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40'),
 'water':dict(id=106,slug='water+treatment+plant',doi='10.24432/C5FS4C',member='water-treatment.data',
    zip_sha256='44e8cc2b294ce8d2d2c7c6a9bbb33dde463591c47add23f85e83ae67a94a0f31',
    member_sha256='af2e632b370646ea9763b11aed54b30423f0c3b8a7fb1b7d23e1c922098a7a8a'),
 'video':dict(id=335,slug='online+video+characteristics+and+transcoding+time+dataset',doi='10.24432/C58C9K',
    member='transcoding_mesurment.tsv',zip_sha256='ad44644b8aca29ad06e29dfe03700e53ff95de007faf7b79eacb77f322629ee9',
    member_sha256='243aa3a554444a615b90e6ddde0560992acaed94ffb871e51d1b6ced3fde0981'),
}
CODECS=('flv','h264','mpeg4','vp8')
VIDEO_NUMERIC=('duration','width','height','bitrate','framerate','i','p','b','frames',
               'i_size','p_size','b_size','size','o_bitrate','o_framerate','o_width','o_height')

def write_once(path,content):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=content:raise ValueError(f'Refusing to overwrite: {path}')
    else:
        with path.open('xb') as stream:stream.write(content)

def npz_once(path,**arrays):
    stream=io.BytesIO();np.savez_compressed(stream,**arrays);write_once(path,stream.getvalue())

def source_bytes(key):
    source=SOURCES[key];path=RAW/(key+'.zip')
    if not path.exists():
        url=f"https://archive.ics.uci.edu/static/public/{source['id']}/{source['slug']}.zip"
        with urlopen(url,timeout=45) as response:content=response.read(25_000_001)
        if len(content)>25_000_000 or hashlib.sha256(content).hexdigest()!=source['zip_sha256']:
            raise ValueError(f'Unexpected download: {key}')
        write_once(path,content)
    if sha(path)!=source['zip_sha256']:raise ValueError(f'Source changed: {key}')
    with zipfile.ZipFile(path) as archive:content=archive.read(source['member'])
    if hashlib.sha256(content).hexdigest()!=source['member_sha256']:raise ValueError('Member changed')
    return content.decode('utf-8-sig')

def parse(key,content):
    rows=[];seen=set();counts=dict(source_rows=0,duplicate_times=0,missing_or_nonfinite=0)
    if key=='water':
        features=['Q-E','ZN-E','PH-E','DBO-E','DQO-E','SS-E','SSV-E','SED-E','COND-E']
        targets=['DBO-S','DQO-S','SS-S'];reader=csv.reader(io.StringIO(content))
    else:
        reader=csv.DictReader(io.StringIO(content),delimiter='\t' if key=='video' else ',')
        reader.fieldnames=[' '.join(x.split()) for x in reader.fieldnames]
        features=list(VIDEO_NUMERIC)+[f'{k}={c}' for k in ('codec','o_codec') for c in CODECS] if key=='video' else [
            'Temperature','Humidity','Wind Speed','general diffuse flows','diffuse flows',
            'sin_hour','cos_hour','sin_weekday','cos_weekday']
        targets=['umem','utime'] if key=='video' else [f'Zone {j} Power Consumption' for j in (1,2,3)]
    for index,row in enumerate(reader):
        if not row:continue
        counts['source_rows']+=1
        if key=='water':
            if len(row)!=39:raise ValueError('Water schema changed')
            stamp=datetime.strptime(row[0],'D-%d/%m/%y');group=stamp.date().toordinal()
            raw_x=row[1:10];raw_y=[row[i] for i in (24,25,26)]
        elif key=='tetouan':
            stamp=datetime.strptime(row['DateTime'],'%m/%d/%Y %H:%M');group=stamp.date().toordinal()
            hour=2*np.pi*(stamp.hour+stamp.minute/60)/24;week=2*np.pi*stamp.weekday()/7
            raw_x=[row[k] for k in features[:5]]+[np.sin(hour),np.cos(hour),np.sin(week),np.cos(week)]
            raw_y=[row[k] for k in targets]
        else:
            group=row['id'];stamp=None
            if not group or any(row[k] not in CODECS for k in ('codec','o_codec')):raise ValueError('Unknown video ID/codec')
            raw_x=[row[k] for k in VIDEO_NUMERIC]+[float(row[k]==c) for k in ('codec','o_codec') for c in CODECS]
            raw_y=[row[k] for k in targets]
        if stamp is not None:
            if stamp in seen:counts['duplicate_times']+=1;continue
            seen.add(stamp)
        try:x,y=np.asarray(raw_x,dtype=float),np.asarray(raw_y,dtype=float)
        except ValueError:counts['missing_or_nonfinite']+=1;continue
        if not np.isfinite(np.concatenate((x,y))).all():counts['missing_or_nonfinite']+=1;continue
        rows.append((stamp,index,group,x,y))
    if key!='video':rows.sort(key=lambda r:r[0])
    return dict(X=np.array([r[3] for r in rows]),Y=np.array([r[4] for r in rows]),
        ids=np.array([r[1] for r in rows]),groups=np.array([r[2] for r in rows]),
        feature_names=features,target_names=targets,counts=counts)

def partition(key,groups):
    unique=np.unique(groups)
    if key=='video':unique=np.random.default_rng(SEED).permutation(unique)
    a,b=int(.6*len(unique)),int(.8*len(unique))
    pieces=[unique[:a],unique[a:b],unique[b:]]
    if key!='video':
        gap=1 if key=='tetouan' else 7
        pieces[0]=pieces[0][pieces[0]<pieces[1][0]-gap]
        pieces[1]=pieces[1][pieces[1]<pieces[2][0]-gap]
    return {k:np.flatnonzero(np.isin(groups,p)) for k,p in zip(('train','val','test'),pieces)}

def prepare_one(key):
    data=parse(key,source_bytes(key));indices=partition(key,data['groups'])
    available=len(indices['train'])
    if available>1200:indices['train']=np.sort(np.random.default_rng(SEED).choice(indices['train'],1200,replace=False))
    assert all(len(v)>=5 for v in indices.values())
    for a,b in (('train','val'),('train','test'),('val','test')):
        assert not set(data['groups'][indices[a]]) & set(data['groups'][indices[b]])
    X,Y=data['X'][indices['train']],data['Y'][indices['train']]
    mean,scale=X.mean(axis=0),X.std(axis=0);scale=np.where(scale>0,scale,1.)
    y_mean,y_scale=Y.mean(axis=0),Y.std(axis=0)
    if np.any(y_scale<=0):raise ValueError('Constant fitting target')
    arrays={}
    for part,index in indices.items():
        arrays[part]=dict(X=(data['X'][index]-mean)/scale,Y=data['Y'][index],
            ids=data['ids'][index],groups=data['groups'][index])
    development=HERE/'data'/f'{key}.npz';test=SEALED/f'{key}.npz'
    npz_once(development,**{f'{k}_{p}':v for p in ('train','val') for k,v in arrays[p].items()})
    npz_once(test,**arrays['test'])
    info=dict(source=SOURCES[key],license='CC BY 4.0',**data['counts'],eligible_rows=len(data['X']),
        available_fitting_rows=available,feature_names=data['feature_names'],target_names=data['target_names'],
        input_mean=mean.tolist(),input_scale=scale.tolist(),target_mean=y_mean.tolist(),target_scale=y_scale.tolist(),
        bounds=np.column_stack((arrays['train']['X'].min(axis=0),arrays['train']['X'].max(axis=0))).tolist(),
        rows={p:len(a['X']) for p,a in arrays.items()},groups={p:len(np.unique(a['groups'])) for p,a in arrays.items()},
        split_id_sha256={p:hashlib.sha256(a['ids'].tobytes()).hexdigest() for p,a in arrays.items()},
        development_sha256=sha(development),sealed_sha256=sha(test),split_seed=SEED,
        gap_excluded_rows=len(data['X'])-sum(len(v) for v in partition(key,data['groups']).values()))
    return info

def main():
    metadata={key:prepare_one(key) for key in DATASETS}
    write_once(HERE/'data/manifest.json',(json.dumps(metadata,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({k:{f:v[f] for f in ('source_rows','eligible_rows','rows','groups','gap_excluded_rows')} for k,v in metadata.items()},indent=2))

if __name__=='__main__':main()
