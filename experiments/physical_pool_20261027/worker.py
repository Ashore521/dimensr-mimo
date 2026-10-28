"""Derive physical models without training, refitting or held-out labels."""
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,AffineCoordinates,select_physical
from dimensr_mimo.selection import digest
from dimensr_mimo.audit import data_digest
from experiments.physical_pool_20261027 import data
read,write,sha=data.read,data.write,data.sha


def forbidden(*args,**kwargs):raise RuntimeError('This frozen-pool study forbids fit.')


def run(folder,job,config):
    torch.set_num_threads(1);start=time.monotonic()
    result=dict(job=job,status='failed',test_data_used=False,searches=0,coefficients_refitted=False)
    DimenSRMIMORegressor.fit=forbidden
    try:
        source=ROOT/job['source'];source_hash=sha(source)
        model=DimenSRMIMORegressor.load(source);before=digest(model.to_dict())
        spec=data.spec(job);coords=AffineCoordinates(**spec['coordinates'])
        train,validation=data.arrays(job,'train'),data.arrays(job,'validation')
        kwargs=dict(coordinates=coords,input_bounds=spec['bounds'],physical_limits=spec['limits'],
            training_data=train,validation_data=validation)
        control=PhysicalModel.from_estimator(model,**kwargs);control.save(folder/'control.json')
        selected,report=select_physical(model,**kwargs);write(folder/'selection.json',report)
        if selected is not None:selected.save(folder/'selected.json')
        assert before==digest(model.to_dict()) and source_hash==sha(source)
        result.update(status='complete' if selected is not None else 'no_valid_model',source_sha256=source_hash,
            candidate_count=report['candidate_count'],selected_index=report['candidate_index'],
            search_fingerprint=model.search_fingerprint_,source_model_unchanged=True,
            data_sha256=dict(training=data_digest(*train),validation=data_digest(*validation)),
            control_acceptance=control.acceptance,selected_acceptance=selected.acceptance if selected is not None else None)
    except Exception as error:
        result['error']=repr(error);(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result['worker_wall_seconds']=time.monotonic()-start
    write(folder/'result.json',result)


if __name__=='__main__':run(Path(sys.argv[1]),read(sys.argv[2]),read(sys.argv[3]))
