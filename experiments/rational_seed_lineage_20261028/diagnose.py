"""Observe two frozen training-only initializer calls, without changing their code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN = HERE.parent/'protected_closure_20261028/frozen_inputs'
MODEL = HERE.parent/'search_lineage_20261028/runs/rational_n0_s702_random/model.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def worker():
    sys.path[:0] = [str(FROZEN/'src'), str(FROZEN)]
    import importlib.metadata
    import numpy as np
    from dimensr_mimo import SearchConfig, structure
    from dimensr_mimo._single.units import unit
    from dimensr_mimo.generation import SystemGrammar
    from experiments.noise_families_20261026 import cases

    assert Path(structure.__file__).resolve().is_relative_to(FROZEN)
    assert Path(cases.__file__).resolve().is_relative_to(FROZEN)
    historical = json.loads(MODEL.read_text(encoding='utf-8'))
    cfg = SearchConfig(**historical['config'])
    grammar = SystemGrammar(*(tuple(map(unit, historical[k])) for k in
        ('feature_units', 'target_units', 'constant_units')), cfg)
    X, Y = cases.training_data('rational_n0', 'train', 702)
    files = list((FROZEN/'src').rglob('*.py')) + [Path(cases.__file__), MODEL,
        Path(__file__), HERE/'PLAN.md', HERE/'PRE_RUN_REVIEW.md', HERE/'BUDGET.json']
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    data_hash = hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest()
    write(HERE/'FREEZE.json', dict(files=hashes, case='rational_n0', seed=702,
        training_shapes=[list(X.shape),list(Y.shape)], training_dtype=str(X.dtype),
        training_sha256=data_hash, expected_calls=2, limit=12, compound=True,
        nonlinear_fits=0, validation_labels=False, test_labels=False,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch')}))

    def feature(item):
        tree, constants, values = item
        return dict(tree=tree.to_dict(), constants=constants.tolist(),
            vector_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def raw(tree, constants):
        coefficients = iter(constants)
        def visit(node):
            token = node['token']
            children = node['children']
            if not children:
                return X[:,int(token[1:])] if token.startswith('x') else next(coefficients)
            values = [visit(c) for c in children]
            return dict(add=np.add, sub=np.subtract, mul=np.multiply, div=np.divide,
                square=np.square, sqrt=np.sqrt, sin=np.sin, cos=np.cos,
                exp=np.exp, log=np.log, neg=np.negative)[token](*values)
        with np.errstate(all='ignore'):
            return np.broadcast_to(visit(tree), (len(X),))

    checks = []
    for j, g in enumerate(grammar.plain):
        attempts, denominators, captured = [], [], {}
        stream = (HERE/f'output_{j}_trace.jsonl').open('x',encoding='utf-8')
        def emit(kind, record):
            stream.write(json.dumps(dict(kind=kind,record=record),allow_nan=False)+'\n')
            stream.flush()
        def trace(frame, event, arg):
            if frame.f_code.co_filename != structure.__file__:
                return None
            name, local = frame.f_code.co_name, frame.f_locals
            if name not in ('admit','fit_terms','structured_seeds'):
                return None
            if name=='fit_terms' and event=='call' and local.get('denominator') is not None:
                denominators.append(feature(local['denominator']))
                emit('denominator_call',denominators[-1])
            if name=='admit' and event=='return':
                constants = local.get('c',local.get('values'))
                attempts.append(dict(tree=local['tree'].to_dict(),
                    constants=None if constants is None else np.asarray(constants).tolist(),
                    admitted=arg is not None, return_line=frame.f_lineno))
                emit('admit_return',attempts[-1])
            if name=='structured_seeds' and event=='return':
                captured.update(features=list(local['features']), proposals=dict(local['proposals']),
                    eligible_denominators=list(local.get('denominators',[])))
            return trace
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            seeds, counts = structure.structured_seeds(g, X, Y[:,j], 12, compound=True)
        finally:
            sys.settrace(previous)
            stream.close()
        proposals = captured['proposals']
        ordered = sorted(proposals, key=lambda t:(proposals[t][0],t.size,repr(t)))
        write(HERE/f'output_{j}_raw.json',dict(output=j,counts=counts,
            features=[feature(f) for f in captured['features']],
            eligible_denominators=[feature(f) for f in captured['eligible_denominators']],
            denominator_calls=denominators,proposals=[dict(tree=t.to_dict(),constants=proposals[t][1].tolist(),
                score=proposals[t][0]) for t in ordered],
            returned_seeds=[dict(tree=t.to_dict(),constants=c.tolist()) for t,c in seeds]))
        rows, recomputed = [], []
        for rank, tree in enumerate(ordered):
            score, c = proposals[tree]
            prediction = raw(tree.to_dict(), c)
            mse = float(np.mean((prediction-Y[:,j])**2))
            checked_score = mse/max(float(np.var(Y[:,j])),1e-12)+1e-8*tree.size
            np.testing.assert_allclose(checked_score, score, rtol=1e-10, atol=1e-12)
            recomputed.append((checked_score,tree.size,repr(tree),tree))
            rows.append(dict(rank=rank, tree=tree.to_dict(), constants=c.tolist(),
                score=score, recomputed_score=checked_score, training_nrmse=float(np.sqrt(mse)/Y[:,j].std()),
                retained=rank<12))
        assert [t for t,c in seeds]==ordered[:12]
        for tree,c in seeds:
            np.testing.assert_array_equal(c,proposals[tree][1])
        independent = [r[3] for r in sorted(recomputed,key=lambda r:r[:3])]
        order_match = independent==ordered
        seeds_match = independent[:12]==[t for t,c in seeds]
        assert counts==historical['diagnostics']['initialization_diagnostics'][j]
        write(HERE/f'output_{j}.json',dict(output=j,counts=counts,proposals=rows,retained_count=len(seeds),
            independent_order_as_original_indices=[ordered.index(t) for t in independent],
            independent_full_order_match=order_match,independent_seed_order_match=seeds_match))
        checks.append(dict(output=j,verified=order_match and seeds_match,proposals=len(rows),historical_counts_match=True,
            independent_full_order_match=order_match,independent_seed_order_match=seeds_match,
            ranking_status='matched' if order_match and seeds_match else 'ranking_unresolved'))
        print('output',j,checks[-1]['ranking_status'],len(rows),'proposals',flush=True)
    assert all(sha(ROOT/rel)==digest for rel,digest in hashes.items())
    write(HERE/'VERIFICATION.json',dict(status='passed' if all(c['verified'] for c in checks) else 'incomplete',rows=checks,completed_calls=len(checks),
        freeze_sha256=sha(HERE/'FREEZE.json'),outputs={p.name:sha(p) for p in HERE.glob('output_*.*')},
        nonlinear_fits=0,validation_labels=False,test_labels=False))


def main():
    assert not any((HERE/name).exists() for name in ('BUDGET.json','FREEZE.json','EXECUTION.json'))
    start = time.time()
    write(HERE/'BUDGET.json',dict(start_unix=start,limit_seconds=60,initializer_calls=2,numerical_threads=1))
    env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    reason = None
    with (HERE/'worker.log').open('x',encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],cwd=ROOT,
            env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=max(.01,start+60-time.time()))
        except subprocess.TimeoutExpired:
            process.kill();code=process.wait();reason='global_deadline'
    ok = (HERE/'VERIFICATION.json').exists() and json.loads((HERE/'VERIFICATION.json').read_text())['status']=='passed'
    write(HERE/'EXECUTION.json',dict(returncode=code,termination_reason=reason,
        wall_seconds=time.time()-start,verification_passed=ok))
    if code or reason or not ok:
        raise SystemExit(1)


if __name__=='__main__':
    worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main()
