"""Check standalone 1.5.0 delivery and replay a frozen physical selection."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
wheel = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(wheel))


class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments' or fullname.startswith('experiments.'):
            raise ImportError('Wheel must be independent of experiment helpers.')


sys.meta_path.insert(0, NoExperiments())
import numpy as np
import dimensr_mimo
from dimensr_mimo import AffineCoordinates, DimenSRMIMORegressor, PhysicalModel, select_physical
from dimensr_mimo import physical_selection

assert dimensr_mimo.__version__ == '1.5.0'
assert all(str(wheel) in module.__file__ for module in (dimensr_mimo, physical_selection))
temp = ROOT / 'outputs/physical-wheel-check'
temp.mkdir(exist_ok=True)
checked = 0
for path in sorted((HERE / 'runs').glob('*/selected.json')):
    physical = PhysicalModel.load(path)
    physical.save(temp / 'physical.json')
    assert PhysicalModel.load(temp / 'physical.json').to_dict() == physical.to_dict()
    checked += 1
assert checked == 24

old = ROOT / 'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json'
model = DimenSRMIMORegressor.load(old)
saved = json.loads(old.read_text(encoding='utf-8'))
assert model.best_.to_dict() == saved['best']
assert model.post_selection_['status'] == saved['post_selection']['status']
model.save(temp / 'migrated.json')
assert DimenSRMIMORegressor.load(temp / 'migrated.json').to_dict() == model.to_dict()

spec = json.loads((HERE / 'FREEZE.json').read_text(encoding='utf-8'))['metadata']['affine_lengths_s2701_pool']['spec']
model = DimenSRMIMORegressor.load(ROOT / 'experiments/native_api_20261026/runs/affine_lengths_physical/model.json')
rng = np.random.default_rng(2601)
bounds = np.asarray([[8., 12.], [90., 110.]])
X = rng.uniform(bounds[:, 0], bounds[:, 1], (144, 2))
Y = np.column_stack((2*X[:, 0] + 3*X[:, 1] + 5, X[:, 0] - X[:, 1] + 2))
selected, report = select_physical(model, coordinates=AffineCoordinates(**spec['coordinates']),
    input_bounds=spec['bounds'], physical_limits=spec['limits'],
    training_data=(X[:96], Y[:96]), validation_data=(X[96:], Y[96:]))
folder = HERE / 'runs/affine_lengths_s2701_pool'
assert report == json.loads((folder / 'selection.json').read_text(encoding='utf-8'))
assert selected.to_dict() == PhysicalModel.load(folder / 'selected.json').to_dict()
np.testing.assert_allclose(selected.predict([[10., 100.]]), [[325., -88.]], rtol=0, atol=1e-11)
record = dict(status='passed', version=dimensr_mimo.__version__, wheel=wheel.name,
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), experiment_imports_blocked=True,
    physical_models_reloaded=checked, old_model_choice_and_status_preserved=True,
    old_model_resave_reload_pass=True, affine_selection_exact_replay=True, physical_prediction_pass=True,
    check_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE / 'PACKAGING.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
