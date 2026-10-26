"""Fresh splits of known development tasks, with a training-only entry point."""
from experiments.second_round_20261026.cases import previous

CASES = ('shared_parameter', 'scales', 'partial', 'boundary')
SEEDS = (201, 202, 203, 204, 205)
_source = previous('cases')
design = _source.design


def generate(name, split, seed):
    if name not in CASES or seed not in SEEDS:
        raise ValueError('Outside the preregistered matrix.')
    return _source.generate(name, split, seed)


def training_data(name, split, seed):
    if split not in ('train', 'validation'):
        raise ValueError('Training and selection cannot access test data.')
    return generate(name, split, seed)
