"""Shared execution plumbing for the two fixed completion development batches."""
from dataclasses import asdict
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import psutil
import torch

from dimensr_mimo import DimenSRMIMORegressor, PhysicalModel, SearchConfig
from experiments.native_api_20261026.run import read, write, sha, arrays_hash
from experiments.native_api_20261026.verify import require
from experiments.precision_repair_20261028 import study as audited
from experiments.second_round_20261026.cases import previous
from experiments.typed_coverage_20261026 import delivery

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT/'experiments/compositional_completion_20261031'
HISTORICAL = ROOT/'experiments/completion_historical_20261031'
GATE_FILES = ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','POST_RUN_REVIEW.md')


def isolated(path,label):
    spec = importlib.util.spec_from_file_location(label,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(s,worker):
    s.ROOT,s.SHARED = ROOT,SHARED
    s.METHODS,s.LIMIT = ('base','completion'),2880
    s.PRIOR = ROOT/'experiments'/('precision_repair_20261028' if s.STAGE=='historical' else 'new_task_validation_20261030')
    s.EXPECTED = 34 if s.STAGE=='historical' else 32
    case_path = ROOT/'experiments'/('compound_search_20261027' if s.STAGE=='historical' else 'new_task_validation_20261030')/'cases.py'
    s.cases = isolated(case_path,'_completion_'+s.STAGE+'_cases')
    s.cases.HERE = s.HERE
    original_generate = s.cases.generate
    def generate(case,split,seed):
        if split=='test' and not (s.HERE/'ALL_MODELS_FROZEN.json').exists():
            raise RuntimeError('Test labels require the current all-model seal.')
        return original_generate(case,split,seed)
    s.cases.generate = generate
    s.supervisor = previous('study',cases=s.cases,worker=worker)
    s.supervisor.HERE,s.supervisor.ROOT = s.HERE,s.ROOT
    s.supervisor.budget = s.budget
    s.name,s.mechanism = audited.name,audited.mechanism


def verify_files(s,files,root=None):
    root = s.ROOT if root is None else root
    for name,expected in files.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(path)==expected,'Frozen file drift: '+name)


def check_candidate_lock(s):
    lock = read(s.SHARED/'CANDIDATE_LOCK.json')
    actual = {p.relative_to(s.ROOT).as_posix() for p in (s.ROOT/'src').rglob('*.py')}
    require(bool(actual) and set(lock['source_files'])==actual,'Candidate source manifest drift.')
    require(lock.get('expected_source_count',len(actual))==len(actual),'Candidate source count drift.')
    verify_files(s,lock['source_files'])


def check_stage_gate(s):
    if s.STAGE=='historical':
        return
    prior = s.HERE.parent/'completion_historical_20261031'
    receipt = read(prior/'POST_REVIEW_ACCEPTANCE.json')
    require(receipt['status']=='accepted','Historical independent review not accepted.')
    expected = {n:sha(prior/n) for n in GATE_FILES}
    expected['CANDIDATE_LOCK.json'] = sha(s.SHARED/'CANDIDATE_LOCK.json')
    require(receipt['sha256']==expected,'Historical review receipt chain drift.')
    summary,verification,frozen = (read(prior/n) for n in ('SUMMARY.json','VERIFICATION.json','FREEZE.json'))
    require(summary['technical_pass'] is True and summary['adoption_pass'] is True and verification['status']=='passed','Historical gates failed.')
    require(frozen['candidate_lock_sha256']==expected['CANDIDATE_LOCK.json'],'Different stage candidate lock.')
    require(frozen['config']==configuration(s),'Different stage configuration.')
    require(verification['freeze_sha256']==expected['FREEZE.json'] and verification['models_sha256']==expected['ALL_MODELS_FROZEN.json']
        and verification['scores_sha256']==expected['SCORES.json'],'Historical verification chain drift.')
    require(read(prior/'ALL_MODELS_FROZEN.json')['freeze_sha256']==expected['FREEZE.json'] and
        read(prior/'SCORES.json')['all_models_sha256']==expected['ALL_MODELS_FROZEN.json'],'Historical model/score chain drift.')


def matrix(s):
    return [dict(case=c,seed=seed,method=m,seconds=60) for c in s.cases.CASES for seed in s.cases.seeds(c)
            for m in (s.METHODS if seed%2 else s.METHODS[::-1])]


def configuration(s):
    prior = s.ROOT/'experiments/precision_repair_20261028'
    require(read(prior/'ALL_MODELS_FROZEN.json')['freeze_sha256']==sha(prior/'FREEZE.json'),'Original configuration seal drift.')
    return dict(read(prior/'FREEZE.json')['config'],precision_repair=False,compositional_completion=False)


def effective_config(s,job,config):
    require(job['method'] in s.METHODS,'Unknown completion study method.')
    return SearchConfig(**dict(config,random_state=job['seed'],initialization='compound',sharing_strategy='closure',
        precision_repair=job['method']=='base',compositional_completion=job['method']=='completion',
        final_selection='precision',acceptance=s.cases.design(job['case'])['acceptance']))


def preflight(s):
    s.check_candidate_lock();s.check_stage_gate()
    require(not (s.HERE/'PREFLIGHT.json').exists(),'Keep existing preflight.')
    jobs = s.matrix()
    require(len(jobs)==len({s.name(j) for j in jobs})==s.EXPECTED,'Wrong matrix.')
    for job in jobs:
        effective_config(s,job,s.configuration()).validate()
    rows = []
    original = read(s.PRIOR/'FREEZE.json')
    for case in s.cases.CASES:
        require(s.cases.design(case)==original['specs'][case],'Original case specification drift.')
        for seed in s.cases.seeds(case):
            X,Y = s.cases.training_data(case,'train',seed)
            V,Z = s.cases.training_data(case,'validation',seed)
            require(X.shape==Y.shape==(96,2) and V.shape==Z.shape==(48,2),'Wrong data shape.')
            require(all(np.isfinite(a).all() for a in (X,Y,V,Z)) and np.all(Y.std(axis=0)>0),'Invalid training data.')
            data_hash = arrays_hash(X,Y,V,Z)
            require(data_hash==original['data_sha256'][f'{case}_s{seed}'],'Original training/validation data drift.')
            rows.append(dict(case=case,seed=seed,data_sha256=data_hash))
    write(s.HERE/'PREFLIGHT.json',dict(status='passed',rows=rows,test_labels_accessed=False))


def freeze_inputs(s):
    prior_freeze = read(s.PRIOR/'FREEZE.json')
    files = [s.ROOT/n for n in prior_freeze['files']]
    # Original archives prove old bytes; live inputs separately freeze this execution.
    files += [s.PRIOR/'frozen_inputs'/n for n in prior_freeze['files']]
    files += [s.PRIOR/n for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','FROZEN_INPUT_ARCHIVE.json',
        'SCORES.json','VERIFICATION.json','SUMMARY.json','POST_RUN_REVIEW.md')]
    files += [s.PRIOR/n for n in read(s.PRIOR/'ALL_MODELS_FROZEN.json')['files']]
    files += list((s.ROOT/'src').rglob('*.py'))+list((s.ROOT/'tests').glob('*.py'))
    other = s.HERE.parent/('completion_development_20261031' if s.STAGE=='historical' else 'completion_historical_20261031')
    files += list(other.glob('*.py'))+[other/'PLAN.md',other/'PROTOCOL.md']
    for folder in (s.SHARED,s.HERE):
        files += list(folder.glob('*.py'))+list(folder.glob('*.md'))+list(folder.glob('*.json'))
        for logs in ('harness_validation','harness_review_validation','core_validation','core_review_validation'):
            files += [p for p in (folder/logs).rglob('*') if p.is_file() and p.suffix!='.pyc' and '__pycache__' not in p.parts]
    files += [s.HERE/n for n in ('PLAN.md','PROTOCOL.md','HARNESS_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json')]
    files += [s.SHARED/n for n in ('PLAN.md','DESIGN_REVIEW.md','BASE_SOURCE_LOCK.json','CANDIDATE_LOCK.json','CORE_REVIEW.md')]
    files += [s.ROOT/'pyproject.toml',s.ROOT/'docs/COMPOSITIONAL_COMPLETION.zh-CN.md']
    if s.STAGE=='development':
        prior = s.HERE.parent/'completion_historical_20261031'
        files += [prior/n for n in GATE_FILES+('POST_REVIEW_ACCEPTANCE.json',)]
    return sorted(set(p for p in files if not p.is_dir()))


def freeze(s):
    s.check_candidate_lock();s.check_stage_gate()
    require(not any((s.HERE/n).exists() for n in ('FREEZE.json','BUDGET.json')),'Already frozen.')
    require(read(s.HERE/'PREFLIGHT.json')['status']=='passed','Preflight failed.')
    s.verify_files(read(s.PRIOR/'FREEZE.json')['files'],s.PRIOR/'frozen_inputs')
    s.verify_files(read(s.PRIOR/'ALL_MODELS_FROZEN.json')['files'],s.PRIOR)
    files = s.freeze_inputs()
    for path in files:
        require(path.is_file(),'Missing freeze input: '+str(path))
    for path in (s.ROOT/'src').rglob('*.py'):
        dest = s.HERE/'source'/path.relative_to(s.ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest);files.append(dest)
    b = dict(start_unix=time.time(),limit_seconds=s.LIMIT,sequential_workers=1,threads=1,system_cap_seconds=60,
        fit_seconds=52,verification_reserve_seconds=600,versions={n:importlib.metadata.version(n)
            for n in ('numpy','scipy','torch','sympy','scikit-learn','psutil')})
    write(s.HERE/'BUDGET.json',b);files.append(s.HERE/'BUDGET.json')
    write(s.HERE/'FREEZE.json',dict(frozen_unix=time.time(),jobs=s.matrix(),config=s.configuration(),
        candidate_lock_sha256=sha(s.SHARED/'CANDIDATE_LOCK.json'),
        specs={c:s.cases.design(c) for c in s.cases.CASES},
        data_sha256={f'{c}_s{seed}':arrays_hash(*s.cases.training_data(c,'train',seed),*s.cases.training_data(c,'validation',seed))
            for c in s.cases.CASES for seed in s.cases.seeds(c)},files={p.relative_to(s.ROOT).as_posix():sha(p) for p in files},
        budget=b,test_labels_accessed=False,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=s.ROOT).decode().strip(),
        paired_test_protection=dict(relative=.1,absolute=1e-8)))


def train(s):
    s.check_candidate_lock();s.check_stage_gate()
    frozen = read(s.HERE/'FREEZE.json');s.verify_files(frozen['files'])
    require(frozen['jobs']==s.matrix(),'Frozen matrix drift.')
    if (s.HERE/'ALL_MODELS_FROZEN.json').exists():
        s.verify_files(read(s.HERE/'ALL_MODELS_FROZEN.json')['files'],s.HERE);return
    if not (s.HERE/'TRAINING_STARTED.json').exists():
        write(s.HERE/'TRAINING_STARTED.json',dict(start_unix=time.time()))
    for job in frozen['jobs']:
        s.check_candidate_lock();s.verify_files(frozen['files'])
        folder = s.HERE/'runs'/s.name(job)
        if (folder/'CELL_FROZEN.json').exists():
            s.verify_files(read(folder/'CELL_FROZEN.json')['files'],folder);continue
        require(not (folder/'job.json').exists(),'Unsealed cell requires incident review: '+s.name(job))
        if time.time()+job['seconds']>s.budget()-600:
            folder.mkdir(parents=True,exist_ok=True)
            write(folder/'job.json',job);write(folder/'config.json',frozen['config'])
            write(folder/'resources.json',dict(status='not_started_budget',job=job))
        else:
            s.supervisor.run_job(job,frozen['config'],'runs')
        write(folder/'CELL_FROZEN.json',dict(frozen_unix=time.time(),files={p.name:sha(p)
            for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'}))
    write(s.HERE/'ALL_MODELS_FROZEN.json',dict(frozen_unix=time.time(),test_labels_accessed=False,
        freeze_sha256=sha(s.HERE/'FREEZE.json'),files={p.relative_to(s.HERE).as_posix():sha(p)
            for p in (s.HERE/'runs').rglob('*') if p.is_file()}))


def worker_run(s,folder,job,config):
    s.check_candidate_lock();s.check_stage_gate()
    torch.set_num_threads(1);start=time.monotonic();cpu=time.process_time()
    result=dict(job=job,status='failed',models=[],diagnostics=[],prediction_hashes=[],audits=[],test_scores_computed=False)
    try:
        X,Y=s.cases.training_data(job['case'],'train',job['seed']);V,Z=s.cases.training_data(job['case'],'validation',job['seed'])
        spec=s.cases.design(job['case']);cfg=effective_config(s,job,config)
        def event(row):
            with (folder/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(row,output=None,monotonic=time.monotonic()),allow_nan=False)+'\n')
        model=DimenSRMIMORegressor(cfg,spec['features'],spec['targets'],spec['constants'],spec['feature_names']).fit(
            X,Y,X_val=V,Y_val=Z,input_bounds=spec['bounds'],callback=event)
        model.save(folder/'model.json')
        result.update(models=['model.json'],diagnostics=[model.diagnostics_],effective_config=asdict(cfg),
            search_fingerprint=model.search_fingerprint_,application_acceptance=model.post_selection_)
        physical=PhysicalModel.from_estimator(model,input_bounds=spec['bounds'],physical_limits=spec['acceptance'],
            training_data=(X,Y),validation_data=(V,Z));physical.save(folder/'physical.json')
        result['prediction_hashes']=[{n:arrays_hash(model.predict(A)) for n,A in [('train',X),('validation',V)]}]
        result['audits']=[model.audit_domain(spec['bounds'])]
        models=[DimenSRMIMORegressor.load(folder/'model.json')]
        system,constants=delivery.graph(models,False);bundle=delivery.record(system,constants,spec,X,Y,V,Z)
        np.testing.assert_array_equal(model.predict(X),system.evaluate(X,constants))
        bundle.update(formulas=system.formulas(constants,spec['feature_names']),sharing=system.sharing,domain_audits=result['audits'])
        write(folder/'delivery.json',bundle);write(folder/'pool.json',delivery.pool(read(folder/'model.json'),spec))
        result.update(status='complete',data_sha256=arrays_hash(X,Y,V,Z),development_accepted=bundle['development_accepted'])
    except Exception as error:
        result.update(status='failed',error=repr(error));(folder/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
    result.update(worker_wall_seconds=time.monotonic()-start,worker_cpu_seconds=time.process_time()-cpu)
    write(folder/'result.json',result)


def bounded(s):
    require(not any((s.HERE/n).exists() for n in ('BUDGET.json','supervisor.log','EXECUTION.json','PREFLIGHT.json',
        'FREEZE.json','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json','SCORES.json','SCORING_FAILURE.json','VERIFICATION.json')),
        'Existing execution requires incident inspection; no restart.')
    s.check_candidate_lock();s.check_stage_gate()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    start=time.time();deadline,reason=start+60,None
    with (s.HERE/'supervisor.log').open('x',encoding='utf-8') as log:
        process=subprocess.Popen([sys.executable,str(s.HERE/'study.py'),'run'],cwd=s.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            if (s.HERE/'BUDGET.json').exists():
                try:
                    b=read(s.HERE/'BUDGET.json');deadline=b['start_unix']+b['limit_seconds']
                except json.JSONDecodeError:
                    pass
            if time.time()>=deadline:
                reason='global_deadline'
                try:
                    for child in reversed(psutil.Process(process.pid).children(recursive=True)):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    process.kill()
                except psutil.NoSuchProcess:pass
                break
            time.sleep(.2)
        code=process.wait()
    try:verified=read(s.HERE/'VERIFICATION.json')['status']=='passed'
    except (OSError,ValueError,KeyError,TypeError):verified=False
    report=dict(returncode=code,termination_reason=reason,wall_seconds=time.time()-start,poll_seconds=.2,verification_passed=verified)
    write(s.HERE/'EXECUTION.json',report);print(json.dumps(report))
    if reason or code or not verified:raise SystemExit(1)
