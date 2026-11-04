"""Verify standalone 1.6.0 models and structural proposal generation."""
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
from dimensr_mimo import DimenSRMIMORegressor, PhysicalModel, sharing

assert dimensr_mimo.__version__ == '1.6.0'
assert all(str(wheel) in module.__file__ for module in (dimensr_mimo, sharing))
temp = ROOT/'outputs/matched-wheel-check'
temp.mkdir(exist_ok=True)
checked = 0
for path in sorted((HERE/'runs').glob('*/model.json')):
    model = DimenSRMIMORegressor.load(path)
    model.save(temp/'model.json')
    assert DimenSRMIMORegressor.load(temp/'model.json').to_dict() == model.to_dict()
    physical = PhysicalModel.load(path.with_name('physical.json'))
    physical.save(temp/'physical.json')
    assert PhysicalModel.load(temp/'physical.json').to_dict() == physical.to_dict()
    bounds = np.asarray(physical.input_bounds)
    X = np.random.default_rng(1600).uniform(bounds[:, 0], bounds[:, 1], (12, len(bounds)))
    np.testing.assert_array_equal(model.predict(X), physical.predict(X))
    assert model.config_.sharing_strategy == path.parent.name.rsplit('_', 1)[1]
    checked += 1
assert checked == 34
control = DimenSRMIMORegressor.load(HERE/'runs/damped_n0_s701_random/model.json')
proposals = sharing.matched_proposals(control.best_.system, control.best_.constants)
assert proposals and all(graph.size < control.best_.system.size for graph, _ in proposals)
old = DimenSRMIMORegressor.load(ROOT/'experiments/native_api_20261026/runs/dimensional_frequency_normalized/model.json')
assert old.config_.sharing_strategy == 'random'
old.save(temp/'old.json')
assert DimenSRMIMORegressor.load(temp/'old.json').to_dict() == old.to_dict()
record = dict(status='passed', version=dimensr_mimo.__version__, wheel=wheel.name,
    wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(), experiment_imports_blocked=True,
    searched_and_physical_models_reloaded=checked, physical_prediction_exact=True,
    matched_proposals=len(proposals), old_model_default_and_reload_pass=True,
    check_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'PACKAGING.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
