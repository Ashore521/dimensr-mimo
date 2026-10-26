"""Supplemental delivery audit; does not override the failed mechanism gate."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import PhysicalModel,audit
from experiments.slump_physical_20261026 import data,evaluate
from experiments.native_api_20261026.verify import raw,require

if (HERE/'ARTIFACT_VERIFICATION.json').exists():raise RuntimeError('Retain the existing supplemental record.')
f,_=evaluate.guard();rows=[]
for job in f['jobs']:
    folder=HERE/'runs'/evaluate.study.name(job)
    try:
        r=data.read(folder/'result.json');b=data.read(folder/'delivery.json');ms=evaluate.models(folder,r);single=job['method']=='independent'
        system,values,units,roles,original,constants,coords=evaluate.reconstruct(ms,single)
        require(system.to_dict()==b['system'] and np.array_equal(values,b['constants']),'Physical graph differs.')
        require(audit.complexity(system)==b['physical_structure'] and roles==b['parameter_roles'],'Metadata differs.')
        if not single:
            physical=PhysicalModel.load(folder/'physical.json')
            require(physical.to_dict()==data.read(folder/'physical.json'),'Native physical roundtrip differs.')
            require(physical.acceptance==b['native_physical_acceptance'],'Acceptance differs.')
        for split in ('train','val','test'):
            X,Y=data.load(split);pred=raw(b['system'],b['constants'],X)
            np.testing.assert_array_equal(pred,evaluate.predict(ms,X,coords,single))
            if not single:np.testing.assert_array_equal(pred,physical.predict(X))
        rows.append(dict(job=job,artifact_verified=True,physical_status=None if single else physical.acceptance['status']))
    except Exception as e:rows.append(dict(job=job,artifact_verified=False,error=repr(e)))
data.write(HERE/'ARTIFACT_VERIFICATION.json',dict(status='passed' if all(r['artifact_verified'] for r in rows) else 'failed',
    expected=9,verified=sum(r['artifact_verified'] for r in rows),rows=rows,
    scope='Supplemental prediction, native physical reload and metadata audit only; original technical verification remains 3/9 due to missing joint NSGA-II evaluations.'))
print('Artifact delivery checked:',sum(r['artifact_verified'] for r in rows),'/ 9')
