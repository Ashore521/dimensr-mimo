"""Observe four frozen compound calls without modifying their source or proposals."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent/'new_task_validation_20261030'
FROZEN = SOURCE/'frozen_inputs'
CASES = ('logarithmic_gain_n0', 'algebraic_norm_n0')
CALLS = [(case, j) for case in CASES for j in range(2)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(diagnostic_value(data), stream, indent=2, allow_nan=False)
        stream.write('\n')


def diagnostic_value(value):
    if isinstance(value, dict):
        return {key:diagnostic_value(item) for key,item in value.items()}
    if isinstance(value, (tuple, list)):
        return [diagnostic_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf'}
    return value


def raw(tree, constants, X):
    """Independent strict tree interpretation, including invalid intermediates."""
    import numpy as np
    def count(node):
        return sum(count(child) for child in node['children']) if node['children'] else int(node['token'].startswith('c'))
    if len(constants) != count(tree):
        raise ValueError('Incorrect number of constants.')
    coefficients = iter(constants)
    invalid = np.zeros(len(X), dtype=bool)
    def visit(node):
        nonlocal invalid
        token, children = node['token'], node['children']
        if not children:
            value = X[:,int(token[1:])] if token.startswith('x') else next(coefficients)
        else:
            values = [visit(child) for child in children]
            value = dict(add=np.add, sub=np.subtract, mul=np.multiply, div=np.divide,
                square=np.square, sqrt=np.sqrt, sin=np.sin, cos=np.cos,
                exp=np.exp, log=np.log, neg=np.negative)[token](*values)
        invalid |= ~np.isfinite(value)
        return value
    with np.errstate(all='ignore'):
        result = np.broadcast_to(visit(tree), (len(X),)).astype(float, copy=True)
    sentinel = object()
    assert next(coefficients, sentinel) is sentinel
    result[invalid] = np.nan
    return result


def call_receipts():
    rows = []
    for case,j in CALLS:
        tag = f'{case}_o{j}'
        row = dict(case=case,output=j,verified=False,initializer_returned=False,status='not_started')
        failure, result = HERE/f'output_{tag}_FAILURE.json', HERE/f'output_{tag}.json'
        try:
            if failure.exists():
                saved = json.loads(failure.read_text(encoding='utf-8'))
                assert saved['case']==case and saved['output']==j
                assert type(saved['initializer_returned']) is bool and isinstance(saved['error'], str)
                row.update(status='failed',error=saved['error'],initializer_returned=saved['initializer_returned'])
            elif result.exists():
                saved = json.loads(result.read_text(encoding='utf-8'))
                assert saved['case']==case and saved['output']==j
                assert saved['raw_sha256']==sha(HERE/f'output_{tag}_raw.json')
                assert saved['trace_sha256']==sha(HERE/f'output_{tag}_trace.jsonl')
                passed = all(saved[k] is True for k in ('historical_counts_match','independent_full_order_match','independent_seed_order_match'))
                row.update(status='passed' if passed else 'ranking_unresolved',verified=passed,initializer_returned=True)
            elif (HERE/f'output_{tag}_STARTED.json').exists():
                row.update(status='interrupted_or_failed')
        except Exception as error:
            row.update(status='invalid_receipt',error=repr(error))
        rows.append(row)
    return rows


def save_execution(code, reason, wall_seconds):
    error, verified = None, False
    try:
        receipt = json.loads((HERE/'VERIFICATION.json').read_text(encoding='utf-8'))
        verified = receipt['status']=='passed' and receipt['expected_calls']==4
    except Exception as problem:
        error = repr(problem)
    rows = call_receipts()
    ok = verified and all(row['verified'] for row in rows)
    report = dict(returncode=code,termination_reason=reason,wall_seconds=wall_seconds,
        verification_passed=ok,verification_receipt_error=error,expected_calls=4,
        verified_calls=sum(row['verified'] for row in rows),
        initializer_completed_calls=sum(row['initializer_returned'] for row in rows),calls=rows)
    write(HERE/'EXECUTION.json',report)
    return ok


def worker():
    sys.path[:0] = [str(FROZEN/'src'), str(FROZEN)]
    import importlib.metadata
    import numpy as np
    from dimensr_mimo import SearchConfig, structure
    from dimensr_mimo.audit import data_digest
    from dimensr_mimo._single.units import unit
    from dimensr_mimo.generation import SystemGrammar
    case_path = FROZEN/'experiments/new_task_validation_20261030/cases.py'
    spec = importlib.util.spec_from_file_location('_frozen_lineage_cases', case_path)
    cases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cases)

    assert Path(structure.__file__).resolve().is_relative_to(FROZEN)
    assert Path(cases.__file__).resolve().is_relative_to(FROZEN)
    original = json.loads((SOURCE/'FREEZE.json').read_text())
    seal = json.loads((SOURCE/'ALL_MODELS_FROZEN.json').read_text())
    assert seal['freeze_sha256'] == sha(SOURCE/'FREEZE.json')
    sources = list((FROZEN/'src').rglob('*.py'))
    assert len(sources) == 40
    for path in sources+[case_path]:
        assert sha(path) == original['files'][path.relative_to(FROZEN).as_posix()]
    models, arrays = {}, {}
    model_paths = [SOURCE/'runs'/f'{case}_s9131_base/model.json' for case in CASES]
    for case, path in zip(CASES, model_paths):
        assert sha(path) == seal['files'][path.relative_to(SOURCE).as_posix()]
        models[case] = json.loads(path.read_text(encoding='utf-8'))
        arrays[case] = cases.training_data(case, 'train', 9131)
        assert data_digest(*arrays[case]) == models[case]['diagnostics']['data_sha256']['training']
    files = sources + [case_path, *model_paths, SOURCE/'FREEZE.json', SOURCE/'ALL_MODELS_FROZEN.json',
        Path(__file__), HERE/'PLAN.md', HERE/'PRE_RUN_REVIEW.md', HERE/'BUDGET.json',
        HERE/'test_observer.py', HERE/'TEST_VALIDATION.json', HERE/'PRE_RUN_CHECKS.json']
    files += list(HERE.glob('validation*.log'))
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write(HERE/'FREEZE.json', dict(files=hashes, calls=CALLS, seed=9131,
        training_shapes={case:[list(a.shape) for a in pair] for case,pair in arrays.items()},
        training_sha256={case:data_digest(*pair) for case,pair in arrays.items()},
        expected_calls=4, limit=12, compound=True, implicit=False,
        nonlinear_fits=0, validation_labels=False, test_labels=False,
        versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','torch')}))
    for name, digest in hashes.items():
        destination = HERE/'frozen_inputs'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copyfile(ROOT/name, destination)
        assert sha(destination) == digest

    source_lines = Path(structure.__file__).read_text().splitlines()
    anchors = (
        'if term.size > grammar.max_nodes or term.depth > grammar.max_depth:',
        'if tree.size > grammar.max_nodes or tree.depth > grammar.max_depth or not np.isfinite(error):')
    boundary_lines = {}
    for anchor in anchors:
        matches = [i+1 for i,line in enumerate(source_lines) if line.strip() == anchor]
        assert len(matches) == 1, anchor
        boundary_lines[matches[0]] = 'term' if anchor.startswith('if term.') else 'tree'

    def feature(item):
        tree, constants, values = item
        return dict(tree=tree.to_dict(), constants=constants.tolist(),
            vector_sha256=hashlib.sha256(values.tobytes()).hexdigest())

    def observe(case, j, state):
        historical = models[case]
        cfg = SearchConfig(**historical['config'])
        grammar = SystemGrammar(*(tuple(map(unit, historical[k])) for k in
            ('feature_units', 'target_units', 'constant_units')), cfg)
        g = grammar.plain[j]
        X, Y = arrays[case]
        tag = f'{case}_o{j}'
        attempts, denominators, captured = [], [], {}
        boundary_counts = dict(attempted=0, structure_rejected=0, nonfinite_error=0)
        stream = (HERE/f'output_{tag}_trace.jsonl').open('x',encoding='utf-8')
        def emit(kind, record):
            stream.write(json.dumps(diagnostic_value(dict(kind=kind,record=record)),allow_nan=False)+'\n')
            stream.flush()
        def trace(frame, event, arg):
            if frame.f_code.co_filename != structure.__file__:
                return None
            name, local = frame.f_code.co_name, frame.f_locals
            if name not in ('admit','fit_terms','structured_seeds'):
                return None
            if event == 'line' and frame.f_lineno in boundary_lines:
                symbol = boundary_lines[frame.f_lineno]
                tree = local[symbol]
                too_large = tree.size > g.max_nodes or tree.depth > g.max_depth
                error = local.get('error') if symbol == 'tree' else None
                finite_error = None if error is None else bool(np.isfinite(error))
                constants = local['c'] if symbol == 'tree' else np.r_[local['weight'], local['c']]
                boundary_counts['attempted'] += 1
                boundary_counts['structure_rejected'] += int(too_large)
                boundary_counts['nonfinite_error'] += int(finite_error is False)
                emit('proposal_boundary', dict(tree=tree.to_dict(), constants=np.asarray(constants).tolist(),
                    size=tree.size, depth=tree.depth, structure_rejected=too_large,
                    error=None if error is None or not finite_error else float(error),
                    finite_error=finite_error, source_line=frame.f_lineno))
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
        state['initializer_returned'] = True
        proposals = captured['proposals']
        ordered = sorted(proposals, key=lambda t:(proposals[t][0],t.size,repr(t)))
        write(HERE/f'output_{tag}_raw.json',dict(case=case,output=j,counts=counts,boundary_counts=boundary_counts,
            features=[feature(f) for f in captured['features']],
            eligible_denominators=[feature(f) for f in captured['eligible_denominators']],
            denominator_calls=denominators,proposals=[dict(tree=t.to_dict(),constants=proposals[t][1].tolist(),
                score=proposals[t][0]) for t in ordered],
            returned_seeds=[dict(tree=t.to_dict(),constants=c.tolist()) for t,c in seeds]))
        rows, recomputed = [], []
        for rank, tree in enumerate(ordered):
            score, c = proposals[tree]
            prediction = raw(tree.to_dict(), c, X)
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
        assert boundary_counts['attempted'] > 0
        assert 0 <= boundary_counts['structure_rejected'] <= boundary_counts['attempted']
        assert len(attempts) >= len(captured['features'])
        write(HERE/f'output_{tag}.json',dict(case=case,output=j,counts=counts,boundary_counts=boundary_counts,
            proposals=rows,retained_count=len(seeds),
            historical_counts_match=True,raw_sha256=sha(HERE/f'output_{tag}_raw.json'),
            trace_sha256=sha(HERE/f'output_{tag}_trace.jsonl'),
            independent_order_as_original_indices=[ordered.index(t) for t in independent],
            independent_full_order_match=order_match,independent_seed_order_match=seeds_match))
        print(tag,'matched' if order_match and seeds_match else 'ranking_unresolved',len(rows),'proposals',flush=True)

    for case,j in CALLS:
        tag = f'{case}_o{j}'
        state = dict(initializer_returned=False)
        write(HERE/f'output_{tag}_STARTED.json',dict(case=case,output=j,started_unix=time.time()))
        try:
            observe(case,j,state)
        except Exception as error:
            write(HERE/f'output_{tag}_FAILURE.json',dict(case=case,output=j,error=repr(error),**state))
            break
    checks = call_receipts()
    input_error = None
    try:
        assert all(sha(ROOT/rel)==digest for rel,digest in hashes.items())
        assert all(sha(HERE/'frozen_inputs'/rel)==digest for rel,digest in hashes.items())
    except Exception as error:
        input_error = repr(error)
    write(HERE/'VERIFICATION.json',dict(status='passed' if input_error is None and all(c['verified'] for c in checks) else 'incomplete',
        rows=checks,completed_calls=sum(c['initializer_returned'] for c in checks),verified_calls=sum(c['verified'] for c in checks),
        expected_calls=4,input_integrity_error=input_error,
        freeze_sha256=sha(HERE/'FREEZE.json'),outputs={p.name:sha(p) for p in HERE.glob('output_*.*')},
        nonlinear_fits=0,validation_labels=False,test_labels=False))


def main():
    assert not any((HERE/name).exists() for name in ('BUDGET.json','FREEZE.json','EXECUTION.json','worker.log'))
    assert (HERE/'PRE_RUN_REVIEW.md').is_file()
    start = time.time()
    write(HERE/'BUDGET.json',dict(start_unix=start,limit_seconds=60,initializer_calls=4,numerical_threads=1))
    env = dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    reason = None
    with (HERE/'worker.log').open('x',encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker'],cwd=ROOT,
            env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=max(.01,start+60-time.time()))
        except subprocess.TimeoutExpired:
            process.kill();code=process.wait();reason='global_deadline'
    ok = save_execution(code,reason,time.time()-start)
    if code or reason or not ok:
        raise SystemExit(1)


if __name__=='__main__':
    worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main()
