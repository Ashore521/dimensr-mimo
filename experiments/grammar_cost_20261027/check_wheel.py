"""Isolated-wheel loading, old-model migration and physical inference."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
wheel=Path(sys.argv[1]).resolve();sys.path.insert(0,str(wheel))


class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='experiments' or fullname.startswith('experiments.'):raise ImportError('Wheel must be independent of experiment helpers.')


sys.meta_path.insert(0,NoExperiments())
import numpy as np
import dimensr_mimo
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,SearchConfig,seeds,structure
assert dimensr_mimo.__version__=='1.2.1' and str(wheel) in dimensr_mimo.__file__
assert all(str(wheel) in m.__file__ for m in (seeds,structure))
SearchConfig(wall_seconds=52,time_allocation='phased').validate()
old=ROOT/'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json'
model=DimenSRMIMORegressor.load(old);saved=json.loads(old.read_text(encoding='utf-8'))
assert model.best_.to_dict()==saved['best'] and model.post_selection_['status']==saved['post_selection']['status']
temp=ROOT/'outputs/grammar-wheel-check';temp.mkdir(exist_ok=True);model.save(temp/'migrated.json')
reloaded=DimenSRMIMORegressor.load(temp/'migrated.json')
assert reloaded.to_dict()==model.to_dict()
physical=PhysicalModel.load(ROOT/'experiments/native_api_20261026/runs/affine_lengths_physical/physical.json')
np.testing.assert_allclose(physical.predict([[10.,100.]]),[[325.,-88.]],rtol=0,atol=1e-11)
record=dict(status='passed',wheel=wheel.name,wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    version=dimensr_mimo.__version__,experiment_imports_blocked=True,old_model_choice_and_status_preserved=True,
    old_model_resave_reload_pass=True,physical_prediction_pass=True)
(HERE/'PACKAGING.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))
