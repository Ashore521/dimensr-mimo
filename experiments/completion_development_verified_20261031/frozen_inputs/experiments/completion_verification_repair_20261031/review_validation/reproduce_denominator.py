"""Constructed control-comparator test; never calls formal verification/run."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import verify_sealed as repair
from test_verify import model,save

job=dict(case='fixture',seed=1,method='base',seconds=60)
name=lambda j:j['case']+'_s'+str(j['seed'])+'_'+j['method']
fixture=HERE/'denominator_fixture'
fixture.mkdir(exist_ok=True)
study=SimpleNamespace(HERE=fixture/'current',PRIOR=fixture/'prior',name=name,EXPECTED=1)
score=dict(job=job,status='complete',clean_nrmse=[.01,.02],observed_nrmse=[.01,.02],mean_clean_nrmse=.015,both_accepted=True,
           structure=dict(shared_graph_size=4,independent_parameters=2,expanded_total_nodes=5,maximum_output_depth=2))
old=deepcopy(score);old['job']['method']='repair'
for filename in ('model.json','result.json','delivery.json','pool.json'):
    value=model() if filename=='model.json' else {'unchanged':True}
    save(study.HERE/'runs'/name(job)/filename,value)
    save(study.PRIOR/'runs'/name(old['job'])/filename,value)
result={}
for fault,extra in [('duplicate_unrelated',dict(job=dict(case='extra',seed=9,method='repair'))),('invalid_extra',{'invalid':True})]:
    rows=[old,extra,deepcopy(extra)] if fault=='duplicate_unrelated' else [old,extra]
    save(study.PRIOR/'SCORES.json',{'rows':rows})
    outcomes={}
    for label,cls in [('original',repair.original.Verifier),('repair',repair.SealedVerifier)]:
        verifier=cls.__new__(cls);verifier.study=study
        try:outcomes[label]={'accepted':verifier.control_identity(job,score)}
        except Exception as error:outcomes[label]={'accepted':False,'error':repr(error)}
    result[fault]=outcomes
print(json.dumps(result,indent=2))
