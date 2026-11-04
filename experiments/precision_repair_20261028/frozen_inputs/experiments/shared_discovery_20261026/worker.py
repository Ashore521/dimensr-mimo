"""One search intervention, then the same frozen guarded selector in both arms."""
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
from dimensr_mimo import DimenSRMIMORegressor
from dimensr_mimo.graph import System
from experiments.guarded_selection_20261026 import selector
from experiments.second_round_20261026.worker import selection_record
from experiments.shared_discovery_20261026 import cases,trace

base_worker=cases.previous('worker',cases=cases)
base_worker.generate=cases.training_data
write=base_worker.write
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def run(folder,job,config):
    if job['method'] not in ('control','trig_seeds') or (folder/'result.json').exists():
        raise RuntimeError('Invalid arm or existing result.')
    start=time.monotonic()
    restore=trace.install(folder/'search_trace.jsonl',job['method']=='trig_seeds')
    try:
        base_worker.run(folder,job,config)
    finally:
        restore()
    result=read(folder/'result.json')
    if result['status']!='complete':return
    try:
        source=read(folder/'model.json')
        write(folder/'searched_model.json',source)
        data=selector.derive(source,'guarded')
        write(folder/'model.json',data)
        write(folder/'selection.json',data['post_selection'])
        write(folder/'discovery.json',trace.summarize(folder/'search_trace.jsonl',data))
        model=DimenSRMIMORegressor.load(folder/'model.json')
        graph=System.from_dict(data['best']['system'])
        hashes,loaded={},{}
        for split in ('train','validation'):
            X,_=cases.training_data(job['case'],split,job['seed'])
            hashes[split]=hashlib.sha256(graph.evaluate(X,data['best']['constants']).astype('<f8').tobytes()).hexdigest()
            loaded[split]=hashlib.sha256(model.predict(X).astype('<f8').tobytes()).hexdigest()
        audit=model.audit_domain(input_bounds=cases.design(job['case'])['bounds'])
        result.update(search_reused=False,selection_reused=True,diagnostics=[data['diagnostics']],
            prediction_hashes=[hashes],audits=[audit],reload_predictions_equal=hashes==loaded,
            reload_audits_equal=audit==model.audit_domain(input_bounds=cases.design(job['case'])['bounds']),
            selections=[dict(model='model.json',**selection_record(data),rule=data['post_selection'])],
            worker_total_seconds=time.monotonic()-start)
        if not result['reload_predictions_equal'] or not result['reload_audits_equal']:
            raise RuntimeError('Reload check failed.')
    except Exception as error:
        result.update(status='failed_post_selection',error=repr(error))
        (folder/'traceback_post_selection.txt').write_text(traceback.format_exc(),encoding='utf-8')
    write(folder/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('job');p.add_argument('config')
    a=p.parse_args();run(Path(a.folder),read(Path(a.job)),read(Path(a.config)))
