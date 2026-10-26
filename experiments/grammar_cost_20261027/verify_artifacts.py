"""Supplemental delivery checks; the failed full mechanism gate is preserved."""
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import PhysicalModel,audit
from experiments.grammar_cost_20261027 import study
from experiments.slump_physical_20261026 import evaluate
evaluate.HERE=HERE;evaluate.study.HERE=HERE
from experiments.slump_physical_20261026 import data
from experiments.native_api_20261026.verify import raw,require

if (HERE/'ARTIFACT_VERIFICATION.json').exists():raise RuntimeError('Keep the supplemental record.')
f,_=evaluate.guard();rows=[]
for job in f['jobs']:
    folder=HERE/'runs'/evaluate.study.name(job)
    try:
        r=data.read(folder/'result.json');b=data.read(folder/'delivery.json');ms=evaluate.models(folder,r)
        system,values,units,roles,original,constants,coords=evaluate.reconstruct(ms,False)
        require(system.to_dict()==b['system'] and np.array_equal(values,b['constants']),'Physical graph differs.')
        require(audit.complexity(system)==b['physical_structure'] and roles==b['parameter_roles'],'Metadata differs.')
        physical=PhysicalModel.load(folder/'physical.json')
        require(physical.to_dict()==data.read(folder/'physical.json'),'Physical reload differs.')
        require(physical.acceptance==b['native_physical_acceptance'],'Acceptance differs.')
        for split in ('train','val','test'):
            X,Y=data.load(split);pred=raw(b['system'],b['constants'],X)
            np.testing.assert_array_equal(pred,evaluate.predict(ms,X,coords,False))
            np.testing.assert_array_equal(pred,physical.predict(X))
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        rows.append(dict(job=job,artifact_verified=True,physical_status=physical.acceptance['status'],
            initialization_seconds=events[0]['elapsed_seconds'],
            neural_proposals=sum(e.get('proposals',0) for e in events if e['phase']=='neural'),
            neural_valid=sum(e.get('neural_valid',0) for e in events),
            gp_new=sum(e['new_evaluations'] for e in events if e['phase']=='gp'),
            nsga2_new=[e['new_evaluations'] for e in events if e['phase']=='nsga2']))
    except Exception as e:rows.append(dict(job=job,artifact_verified=False,error=repr(e)))
data.write(HERE/'ARTIFACT_VERIFICATION.json',dict(status='passed' if all(r['artifact_verified'] for r in rows) else 'failed',
    expected=6,verified=sum(r['artifact_verified'] for r in rows),rows=rows,
    scope='Physical prediction/reload/metadata only; complete technical verification has its own unchanged mechanism gate.'))
print('Physical artifacts:',sum(r['artifact_verified'] for r in rows),'/ 6')
