"""Refuse reruns before starting the observed initializer or writing a budget."""
import importlib.util
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('_composition_observer', Path(__file__).with_name('diagnose.py'))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@pytest.mark.parametrize('name', ['BUDGET.json', 'FREEZE.json', 'EXECUTION.json', 'worker.log'])
def test_existing_execution_is_preserved(tmp_path, monkeypatch, name):
    (tmp_path / name).write_bytes(b'preserved failure evidence')
    monkeypatch.setattr(observer, 'HERE', tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('An existing run must never start a process.')
    monkeypatch.setattr(observer.subprocess, 'Popen', forbidden)
    with pytest.raises(AssertionError):
        observer.main()
    assert {p.name for p in tmp_path.iterdir()} == {name}
    assert (tmp_path / name).read_bytes() == b'preserved failure evidence'


def test_missing_review_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(observer, 'HERE', tmp_path)
    with pytest.raises(AssertionError):
        observer.main()
    assert not list(tmp_path.iterdir())


def test_receipt_write_is_exclusive(tmp_path):
    path = tmp_path / 'evidence.json'
    observer.write(path, {'status': 'failed'})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        observer.write(path, {'status': 'passed'})
    assert path.read_bytes() == original


def node(token, *children):
    return dict(token=token,children=list(children))


def test_raw_preserves_invalid_intermediates():
    x = node('x0')
    tree = node('div',node('c0'),node('div',node('c0'),node('sub',x,x)))
    assert np.isnan(observer.raw(tree,[1.,1.],np.array([[.4],[.8]]))).all()


@pytest.mark.parametrize('constants', [[1.], [1.,2.,99.]])
def test_raw_rejects_wrong_constant_count(constants):
    tree = node('add',node('c0'),node('c0'))
    with pytest.raises(ValueError, match='number of constants'):
        observer.raw(tree,constants,np.zeros((2,1)))


def test_raw_consumes_constants_in_tree_order():
    tree = node('sub',node('mul',node('c0'),node('x0')),node('mul',node('c0'),node('x1')))
    np.testing.assert_array_equal(observer.raw(tree,[2.,3.],np.array([[1.,2.],[3.,4.]])),[-4.,-6.])


@pytest.mark.parametrize('size,depth', [(8,4),(16,4),(8,7)])
def test_actual_trace_handles_nonfinite_boundary(size,depth):
    parsed = ast.parse(Path(observer.__file__).read_text(encoding='utf-8'))
    functions = {n.name:n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef)}
    stream=io.StringIO()
    ns=dict(np=np,json=json,diagnostic_value=observer.diagnostic_value,stream=stream,
        structure=SimpleNamespace(__file__='frozen.py'),g=SimpleNamespace(max_nodes=15,max_depth=6),
        boundary_lines={308:'tree'},boundary_counts=dict(attempted=0,structure_rejected=0,nonfinite_error=0),
        attempts=[],denominators=[],captured={})
    exec(compile(ast.Module(body=[functions['emit'],functions['trace']],type_ignores=[]),'trace_fixture','exec'),ns)
    tree=SimpleNamespace(size=size,depth=depth,to_dict=lambda:node('c0'))
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename='frozen.py',co_name='fit_terms'),
        f_lineno=308,f_locals=dict(tree=tree,c=np.array([np.inf,np.nan,-np.inf]),error=np.inf))
    ns['trace'](frame,'line',None)
    result=json.loads(stream.getvalue())['record']
    assert result['constants']==[{'nonfinite':'+inf'},{'nonfinite':'nan'},{'nonfinite':'-inf'}]
    assert result['finite_error'] is False and result['error'] is None
    assert result['structure_rejected']==(size>15 or depth>6)
    assert ns['boundary_counts']['nonfinite_error']==1


@pytest.mark.parametrize('content', [None,'{','[]'])
def test_bad_verification_still_records_execution(tmp_path,monkeypatch,content):
    monkeypatch.setattr(observer,'HERE',tmp_path)
    if content is not None:(tmp_path/'VERIFICATION.json').write_text(content)
    assert observer.save_execution(1,None,.1) is False
    report=json.loads((tmp_path/'EXECUTION.json').read_text())
    assert report['expected_calls']==4 and report['verified_calls']==0
    assert len(report['calls'])==4 and all(r['status']=='not_started' for r in report['calls'])
    assert report['verification_receipt_error']


def test_semantic_failure_preserves_denominator(tmp_path,monkeypatch):
    monkeypatch.setattr(observer,'HERE',tmp_path)
    case,j=observer.CALLS[0];tag=f'{case}_o{j}'
    observer.write(tmp_path/f'output_{tag}_FAILURE.json',dict(case=case,output=j,
        initializer_returned=True,error='score/semantic mismatch'))
    observer.write(tmp_path/'VERIFICATION.json',dict(status='incomplete',expected_calls=4))
    assert observer.save_execution(0,None,.1) is False
    r=json.loads((tmp_path/'EXECUTION.json').read_text())
    assert r['initializer_completed_calls']==1 and r['verified_calls']==0
    assert [row['status'] for row in r['calls']]==['failed','not_started','not_started','not_started']


def test_partial_raw_does_not_prove_completion(tmp_path,monkeypatch):
    monkeypatch.setattr(observer,'HERE',tmp_path)
    case,j=observer.CALLS[0];tag=f'{case}_o{j}'
    observer.write(tmp_path/f'output_{tag}_STARTED.json',dict(case=case,output=j))
    (tmp_path/f'output_{tag}_raw.json').write_text('{')
    assert observer.save_execution(-9,'global_deadline',60.) is False
    r=json.loads((tmp_path/'EXECUTION.json').read_text())
    assert r['calls'][0]['status']=='interrupted_or_failed'
    assert r['initializer_completed_calls']==r['verified_calls']==0
    assert (tmp_path/f'output_{tag}_raw.json').read_text()=='{'


@pytest.mark.parametrize('timeout', [False,True])
def test_supervisor_worker_failure_and_timeout(tmp_path,monkeypatch,timeout):
    monkeypatch.setattr(observer,'HERE',tmp_path)
    (tmp_path/'PRE_RUN_REVIEW.md').write_text('fixture only')
    class Process:
        killed=False
        def wait(self,timeout=None):
            if timeout is not None and should_timeout:
                raise observer.subprocess.TimeoutExpired('fixture',timeout)
            return -9 if self.killed else 1
        def kill(self):
            self.killed=True
    should_timeout=timeout;process=Process()
    monkeypatch.setattr(observer.subprocess,'Popen',lambda *a,**k:process)
    with pytest.raises(SystemExit):observer.main()
    report=json.loads((tmp_path/'EXECUTION.json').read_text())
    assert report['termination_reason']==('global_deadline' if timeout else None)
    assert report['expected_calls']==4 and not report['verification_passed']
    assert process.killed is timeout
