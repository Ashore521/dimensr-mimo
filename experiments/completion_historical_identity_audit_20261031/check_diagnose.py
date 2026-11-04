"""Constructed checks for difference classification; no experiment data."""
from diagnose import differences, timing_path

assert differences({'a': [1., 2.]}, {'a': [1., 2.]}) == []
assert differences({'a': [1., 2.]}, {'a': [1., 3.]}) == [
    dict(path='/a/1', kind='value', current=2., original=3.)]
assert differences({'a': True}, {'a': 1})[0]['kind'] == 'type'
assert differences({'a': []}, {'a': [1]})[0]['kind'] == 'length'
assert differences({'extra': 1}, {})[0]['kind'] == 'missing'
for key in ('epoch_allowance_seconds', 'neural_until_seconds', 'gp_until_seconds'):
    assert timing_path('/history/1/'+key)
    assert not timing_path('/config/'+key)
    assert not timing_path('/history/1/model/'+key)
for path in ('/history/1/evaluations', '/history/1/errors/0', '/best/constants/0', '/structure/shared_graph_size'):
    assert not timing_path(path)
print('Constructed classification checks passed; no data, prediction or fitting called.')
