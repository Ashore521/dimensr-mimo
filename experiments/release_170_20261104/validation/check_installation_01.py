"""Check an isolated installed package against sealed native/physical models."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--installed', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
installed = args.installed.resolve()
root = args.root.resolve()
sys.path.insert(0, str(installed))


class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'experiments' or fullname.startswith('experiments.'):
            raise ImportError('Installed package must be independent of experiment helpers.')


sys.meta_path.insert(0, NoExperiments())
import numpy as np
import dimensr_mimo
from dimensr_mimo import AffineCoordinates, DimenSRMIMORegressor, PhysicalModel, SearchConfig
from dimensr_mimo import completion, repair, sharing

assert dimensr_mimo.__version__ == '1.7.0'
for module in (dimensr_mimo, completion, repair, sharing):
    assert Path(module.__file__).resolve().is_relative_to(installed)
defaults = SearchConfig()
assert defaults.initialization == 'legacy' and defaults.sharing_strategy == 'random'
assert defaults.precision_repair is False and defaults.compositional_completion is False
manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
assert manifest['version'] == '1.7.0' and manifest['models']
args.output.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checked = []
for row in manifest['models']:
    path = (root/row['native']).resolve()
    physical_path = (root/row['physical']).resolve()
    assert path.is_relative_to(root) and physical_path.is_relative_to(root)
    assert sha(path) == row['native_sha256'] and sha(physical_path) == row['physical_sha256']
    model = DimenSRMIMORegressor.load(path)
    physical = PhysicalModel.load(physical_path)
    assert physical.coordinates == AffineCoordinates.identity(model.n_features_in_, model.n_outputs_)
    model.save(args.output/'native_roundtrip.json')
    physical.save(args.output/'physical_roundtrip.json')
    assert DimenSRMIMORegressor.load(args.output/'native_roundtrip.json').to_dict() == model.to_dict()
    assert PhysicalModel.load(args.output/'physical_roundtrip.json').to_dict() == physical.to_dict()
    bounds = np.asarray(physical.input_bounds)
    fractions = np.linspace(.1, .9, 7)[:, None]
    X = bounds[:, 0]+fractions*(bounds[:, 1]-bounds[:, 0])
    np.testing.assert_array_equal(model.predict(X), physical.predict(X))
    checked.append(dict(native=row['native'], native_sha256=row['native_sha256'],
                        physical_sha256=row['physical_sha256'],
                        initialization=model.config_.initialization,
                        sharing_strategy=model.config_.sharing_strategy,
                        precision_repair=model.config_.precision_repair,
                        compositional_completion=model.config_.compositional_completion))

legacy_checked = []
for row in manifest['legacy_native']:
    path = (root/row['native']).resolve()
    assert path.is_relative_to(root) and sha(path) == row['native_sha256']
    model = DimenSRMIMORegressor.load(path)
    model.save(args.output/'legacy_roundtrip.json')
    assert DimenSRMIMORegressor.load(args.output/'legacy_roundtrip.json').to_dict() == model.to_dict()
    assert model.config_.precision_repair is False and model.config_.compositional_completion is False
    legacy_checked.append(row)

for relative, expected in manifest['source_files'].items():
    assert relative.startswith('src/')
    assert sha(installed/relative[4:]) == expected

record = dict(status='passed', version=dimensr_mimo.__version__,
              installed_package_path=str(Path(dimensr_mimo.__file__).resolve()),
              checked_model_count=len(checked), checked_models=checked,
              legacy_native_checked=legacy_checked,
              experiment_imports_blocked=True, default_configuration_passed=True,
              installed_source_count=len(manifest['source_files']),
              native_physical_fixed_point_predictions_exact=True,
              manifest_sha256=sha(args.manifest), script_sha256=sha(Path(__file__)))
(args.output/'INSTALLATION_CHECK.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: value for key, value in record.items() if key != 'checked_models'}))
