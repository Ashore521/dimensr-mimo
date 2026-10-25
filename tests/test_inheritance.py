"""Inheritance and additional invariants that can fail despite a successful fit."""
from pathlib import Path
import hashlib
import json
from dataclasses import replace
import numpy as np
import pytest
from dimensr_mimo import SearchConfig, DimenSRMIMORegressor
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.structure import operator
from dimensr_mimo._single.units import ZERO, unit
from dimensr_mimo.graph import Builder
from dimensr_mimo.evaluation import Evaluator
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.policy import Policy


def test_upstream_files_have_original_hashes():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'docs/UPSTREAM.json').read_text(encoding='utf-8-sig'))
    for name, sha in manifest['sha256'].items():
        assert hashlib.sha256((root / 'src/dimensr_mimo/_single' / name).read_bytes()).hexdigest() == sha


def test_unresolved_is_rejected_without_rewrite():
    x, c = Node('x0'), Node('c0')
    denom = operator('add', operator('sub', operator('square', x), x), c)
    t = operator('div', c, denom)
    b = Builder(); b.roots = [b.tree(t, [1., 1.])]
    s, values = b.finish()
    assert s.audit(values, [[0, 1]], max_boxes=1)[0]['status'] == 'unresolved'
    assert s.audit(values, [[0, 1]], max_boxes=32)[0]['status'] == 'enclosed'
    # Force the proposed constants (zero fit iterations is not required): the
    # acceptance mechanism itself is covered by a no-constant dependency case.
    t2 = operator('div', x, operator('add', operator('sub', x, x), x))
    b = Builder(); b.roots = [b.tree(t2)]
    s2, v2 = b.finish()
    X = np.linspace(.1, 1, 20)[:, None]
    assert Evaluator(X, np.ones((20, 1)), SearchConfig(interval_max_boxes=1), input_bounds=[[.1, 1]])(s2, v2) is None


def test_reference_generation_and_policy_replay_have_finite_gradients():
    cfg = SearchConfig(max_nodes=9, max_depth=5, max_references=16)
    g = SystemGrammar((ZERO,), (ZERO, ZERO), (ZERO,), cfg)
    policy = Policy(g.n_tokens, seed=4, context_size=g.context_size)
    rng = np.random.default_rng(4)
    systems = [g.generate(rng, policy) for _ in range(40)]
    assert any(any(t >= g.plain[0].index['r0'] and t < g.plain[0].index['add']
                   for trace in tr.traces for t in trace.tokens) for _, _, tr in systems)
    lp, en = policy.log_probs([tr for _, _, tr in systems])
    assert np.isfinite(lp.detach().numpy()).all() and np.isfinite(en.detach().numpy()).all()
    # Replay the generated trace, not a reconstructed display tree.
    update = policy.update([tr for _, _, tr in systems], list(np.linspace(0, 1, 40)))
    assert update['gradient_norm'] > 0


def test_different_units_and_output_scales_fit_jointly():
    mass = unit([0, 1, 0, 0, 0, 0, 0]); speed = unit([1, 0, -1, 0, 0, 0, 0])
    p = unit([1, 1, -1, 0, 0, 0, 0]); energy = unit([2, 1, -2, 0, 0, 0, 0])
    a, b, c = Node('x0', units=mass), Node('x1', units=speed), Node('c0')
    first = operator('mul', c, operator('mul', a, b))
    second = operator('mul', c, operator('mul', a, operator('square', b)))
    build = Builder(); build.roots = [build.tree(first), build.tree(second)]
    s, warm = build.finish()
    X = np.random.default_rng(0).uniform(.1, 1, (50, 2))
    Y = np.column_stack((.01 * X[:, 0] * X[:, 1], 100 * X[:, 0] * X[:, 1] ** 2))
    candidate = Evaluator(X, Y, SearchConfig())(s, warm)
    np.testing.assert_allclose(candidate.constants, [.01, 100], rtol=1e-10)
    assert [s.nodes[r].units for r in s.roots] == [p, energy]
