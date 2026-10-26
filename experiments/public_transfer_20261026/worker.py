"""Fixed public-task search and candidate export; development data only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,regressor
from dimensr_mimo._single import DimenSRRegressor
from dimensr_mimo.graph import System
from experiments.public_transfer_20261026 import data as cases
from experiments.second_round_20261026.cases import previous
from experiments.second_round_20261026.worker import selection_record
from experiments.guarded_selection_20261026 import selector
from experiments.shared_discovery_20261026.seeds import typed_seeds

base_worker=previous('worker',cases=cases)
base_worker.generate=cases.training_data
read,write,sha=cases.read,cases.write,cases.sha


def prediction_hashes(model,X,V):
    return {name:hashlib.sha256(model.predict(A).astype('<f8').tobytes()).hexdigest() for name,A in [('train',X),('validation',V)]}


def run(folder,job,config):
    if job['method'] not in ('original','candidate','independent') or (folder/'result.json').exists():
        raise RuntimeError('Invalid arm or existing result.')
    choice=read(HERE/'ALGORITHM.json');start=time.monotonic()
    reused=job['method']=='candidate' and not choice['trig_seeds']
    result=dict(job=job,status='failed',models=[],diagnostics=[],prediction_hashes=[],audits=[],search_reused=reused)
    try:
        if reused:
            source_folder=folder.parent/f"{job['case']}_s{job['seed']}_original"
            result['source_search']=source_folder.name
            if read(source_folder/'resources.json')['status']!='complete':
                result['status']='source_search_failed';write(folder/'result.json',result);return
            for name,expected in read(source_folder/'CELL_FROZEN.json')['files'].items():
                if sha(source_folder/name)!=expected:raise RuntimeError('Source checkpoint drift.')
            source=read(source_folder/'model.json')
            result.update(models=['model.json'],fit_wall_seconds=0.,fit_cpu_seconds=0.,source_model_sha256=sha(source_folder/'model.json'))
            now=time.monotonic()
            (folder/'events.jsonl').write_text(json.dumps(dict(phase='post_selection',output=None,worker_elapsed=now-start,monotonic=now,new_evaluations=0))+'\n',encoding='utf-8')
        else:
            old=regressor.typed_seeds
            if job['method']=='candidate' and choice['trig_seeds']:regressor.typed_seeds=typed_seeds
            try:base_worker.run(folder,job,config)
            finally:regressor.typed_seeds=old
            result=read(folder/'result.json')
            if result['status']!='complete':return
            result['search_reused']=False
            source=read(folder/'model.json') if job['method']!='independent' else None
        X,_=cases.training_data(job['case'],'train',job['seed']);V,_=cases.training_data(job['case'],'validation',job['seed'])
        if job['method']=='candidate':
            if not reused:write(folder/'searched_model.json',source)
            data=selector.derive(source,'guarded');write(folder/'model.json',data);write(folder/'selection.json',data['post_selection'])
            graph=System.from_dict(data['best']['system'])
            result['diagnostics']=[data['diagnostics']]
            result['prediction_hashes']=[{name:hashlib.sha256(graph.evaluate(A,data['best']['constants']).astype('<f8').tobytes()).hexdigest() for name,A in [('train',X),('validation',V)]}]
            result['audits']=[DimenSRMIMORegressor.load(folder/'model.json').audit_domain(input_bounds=cases.design(job['case'])['bounds'])]
        cls=DimenSRRegressor if job['method']=='independent' else DimenSRMIMORegressor
        result['selections']=[];hashes=[];audits=[]
        for filename in result['models']:
            model=cls.load(folder/filename);data=read(folder/filename)
            record=selection_record(data)
            if job['method']!='independent':record['rule']=data.get('post_selection',selector.choose(data['candidates'],'original'))
            result['selections'].append(dict(model=filename,**record))
            hashes.append(prediction_hashes(model,X,V));audits.append(model.audit_domain(input_bounds=cases.design(job['case'])['bounds']))
        result.update(reload_predictions_equal=hashes==result['prediction_hashes'],reload_audits_equal=audits==result['audits'],worker_total_seconds=time.monotonic()-start)
        if not result['reload_predictions_equal'] or not result['reload_audits_equal']:raise RuntimeError('Reload failed.')
        result['status']='complete'
    except Exception as e:
        result.update(status='failed',error=repr(e));(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    write(folder/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('job');p.add_argument('config')
    a=p.parse_args();run(Path(a.folder),read(Path(a.job)),read(Path(a.config)))
