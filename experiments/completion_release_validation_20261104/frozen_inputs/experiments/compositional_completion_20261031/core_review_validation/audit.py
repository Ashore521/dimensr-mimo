"""Independent bounded source/receipt and hand-built expression audit; no search."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE = OUT.parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from dimensr_mimo import SearchConfig, completion
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import ZERO, unit, sub, scale
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.structure import operator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


lock = json.loads((STAGE / 'BASE_SOURCE_LOCK.json').read_text(encoding='utf-8'))
implementation = json.loads((STAGE / 'core_validation/IMPLEMENTATION_CHECKS.json').read_text(encoding='utf-8'))
changed = []
for name, original in lock['source_files'].items():
    assert sha(ROOT / lock['source_archive'] / name) == original, name
    current = sha(ROOT / name)
    assert current == implementation['source_files'][name]['current_sha256'], name
    if current != original:
        changed.append(name)
assert set(changed) == {'src/dimensr_mimo/config.py', 'src/dimensr_mimo/sharing.py',
                        'src/dimensr_mimo/acceptance.py', 'src/dimensr_mimo/regressor.py'}
for name, expected in implementation['new_files'].items():
    assert sha(ROOT / name) == expected, name
for name, expected in implementation['logs'].items():
    assert sha(STAGE / 'core_validation' / name) == expected, name
assert sha(STAGE / 'PLAN.md') == '3d561857fee731eba08cdf3c0fac757fb16a494bd6032d8420779f0ddfe270a5'
assert sha(STAGE / 'DESIGN_REVIEW.md') == implementation['design_review_sha256']
assert sha(ROOT / 'src/dimensr_mimo/completion.py') == '5210be43a043ce38527da8a675b35efc28d7f7fd08cb8488beeb1be708782318'

L = unit([1, 0, 0, 0, 0, 0, 0])
inverse = sub(ZERO, L)
X = np.array([[.3, .5], [.7, .9], [1.1, 1.3], [1.3, 1.7]])
variable = Node('x0', units=L)
z = Node('x1', units=L)
rows = []
for token in ('log', 'sqrt'):
    inner = (operator('square', operator('mul', Node('c1', units=inverse), z))
             if token == 'log' else operator('square', z))
    constants = np.array([1.2]) if token == 'log' else np.array([])
    unary = operator(token, inner)
    for form in ('direct', 'multiply', 'divide'):
        bare = unary if form == 'direct' else operator('mul' if form == 'multiply' else 'div', variable, unary)
        targets = (bare.units, bare.units)
        cfg = SearchConfig(operators=('add', 'mul', 'div', 'square', 'log', 'sqrt'), max_nodes=15, max_depth=6)
        g = SystemGrammar((L, L), targets, (ZERO, inverse), cfg).plain[0]
        y = bare.evaluate(X, constants)
        actual, reason = completion._inverse_target(token, form, None if form == 'direct' else X[:, 0], y)
        assert reason is None
        np.testing.assert_allclose(actual, inner.evaluate(X, constants), rtol=3e-15, atol=3e-15)
        for amplitude in (False, True):
            branch = Node('c0') if form == 'direct' else operator('mul', Node('c0'), variable)
            tree = bare if not amplitude else operator('mul' if form != 'divide' else 'div', branch, unary)
            values = constants if not amplitude else np.r_[1.7, constants]
            tree, values = tree.canonical(values)
            g.trace(tree)
            np.testing.assert_allclose(tree.evaluate(X, values), y * (1.7 if amplitude else 1.), rtol=3e-15, atol=3e-15)
            rows.append(dict(token=token, form=form, amplitude=amplitude, nodes=tree.size,
                             depth=tree.depth, units=[str(v) for v in tree.units]))

# Absorption without a dimensionless constant is legal when an existing
# inverse-unit scale can itself carry the dimensionless weight.
g = SystemGrammar((L,), (ZERO, ZERO), (inverse,), SearchConfig(operators=('mul', 'square'))).plain[0]
tree = operator('square', operator('mul', Node('c0', units=inverse), Node('x0', units=L)))
weighted, values = completion._weighted(tree, np.array([2.]), .25, {inverse: Node('c0', units=inverse)}, g)
g.trace(weighted)
np.testing.assert_allclose(weighted.evaluate(X[:, :1], values), .25 * tree.evaluate(X[:, :1], [2.]))

snapshot = {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted((ROOT / 'src/dimensr_mimo').rglob('*.py'))}
receipt = dict(status='passed_bounded_constructive_and_integrity_checks', source_count=len(snapshot),
               archived_base_sources_verified=len(lock['source_files']), changed_existing_sources=changed,
               source_sha256=snapshot, implementation_logs_verified=list(implementation['logs']),
               handbuilt_form_checks=rows, existing_inverse_scale_absorption=True,
               initializer_executed=False, fitter_executed=False, search_executed=False)
(OUT / 'audit_checks.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(dict(status=receipt['status'], archived_sources=len(lock['source_files']),
                     handbuilt_forms=len(rows), current_source_count=len(snapshot))))
