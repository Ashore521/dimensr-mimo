"""New seeds of four previously inspected development tasks; no blind benchmark claim."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = ('shared_parameter', 'scales', 'partial', 'boundary')
SEEDS = (101, 102, 103, 104, 105)


def previous(name, **dependencies):
    """Load frozen helpers with explicit imports, without changing their files."""
    spec = importlib.util.spec_from_file_location('_round_one_' + name, HERE.parent / 'first_round' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    saved = {key: sys.modules.get(key) for key in dependencies}
    sys.modules.update(dependencies)
    try:
        spec.loader.exec_module(module)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


_cases = previous('cases')
design = _cases.design


def generate(name, split, seed):
    if name not in CASES or seed not in SEEDS:
        raise ValueError('Outside the frozen second-round matrix.')
    return _cases.generate(name, split, seed)


def training_data(name, split, seed):
    if split not in ('train', 'validation'):
        raise ValueError('Training worker cannot request test data.')
    return generate(name, split, seed)
