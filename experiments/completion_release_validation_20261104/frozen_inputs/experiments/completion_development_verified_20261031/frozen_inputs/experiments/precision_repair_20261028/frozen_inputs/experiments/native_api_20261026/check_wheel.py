"""Run with python -I; imports must resolve inside the wheel without experiments."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import zipfile

wheel=Path(sys.argv[1]).resolve();root=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(wheel))


class NoExperimentImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='experiments' or fullname.startswith('experiments.'):
            raise ImportError('Published package must not import experiment helpers.')


sys.meta_path.insert(0,NoExperimentImports())
import numpy as np
import dimensr_mimo
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,seeds,structure
assert str(wheel) in dimensr_mimo.__file__,dimensr_mimo.__file__
assert all(str(wheel) in m.__file__ for m in (seeds,structure))
assert dimensr_mimo.__version__=='1.1.0'
loaded=[]
for folder in sorted((root/'runs').iterdir()):
    model=DimenSRMIMORegressor.load(folder/'model.json')
    assert model.to_dict()==json.loads((folder/'model.json').read_text(encoding='utf-8'))
    loaded.append(folder.name)
physical=PhysicalModel.load(root/'runs/affine_lengths_physical/physical.json')
np.testing.assert_allclose(physical.predict([[10.,100.]]),[[325.,-88.]],atol=1e-11,rtol=0)
with zipfile.ZipFile(wheel) as archive:
    names=archive.namelist()
    assert not any(n.startswith('experiments/') for n in names)
result=dict(status='passed',wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
    package_version=dimensr_mimo.__version__,experiment_imports_blocked=True,loaded_models=loaded,
    physical_prediction_checked=True,package_files=len(names))
print(json.dumps(result,indent=2))
(root/'PACKAGING.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
