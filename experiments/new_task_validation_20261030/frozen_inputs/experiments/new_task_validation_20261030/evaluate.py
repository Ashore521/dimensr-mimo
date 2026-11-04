"""Isolated reuse of audited raw/repair verification for the sealed new matrix."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.new_task_validation_20261030 import cases, study
from experiments.native_api_20261026.run import read, write, sha
from experiments.native_api_20261026.verify import require

# Each loaded module owns its globals; no historical module singleton is patched.
spec = importlib.util.spec_from_file_location('_new_task_repair_verifier',
    HERE.parent/'precision_repair_20261028/evaluate.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
legacy.HERE, legacy.cases, legacy.study = HERE, cases, study
base = legacy.base
base.HERE, base.cases, base.study = HERE, cases, study
_raw_error = base.error


def error(pred,true,scale):
    values = _raw_error(pred,true,scale)
    require(np.isfinite(values).all(),'Nonfinite normalized prediction errors.')
    return values


# The historical scorer catches cell exceptions before strict JSON serialization.
base.error = error


def guards():
    study.check_candidate_lock()
    return legacy.guards()


base.guards = guards
evaluate = base.evaluate
closure_evidence, paired_identity = legacy.closure_evidence, legacy.paired_identity


def complete(row):
    if row.get('status')!='complete':
        return False
    try:
        errors = np.asarray([row['clean_nrmse'],row['observed_nrmse']],dtype=float)
        structure = row['structure']
        sizes = [structure[k] for k in ('shared_graph_size','independent_parameters',
            'expanded_total_nodes','maximum_output_depth')]
        return bool(errors.shape==(2,2) and np.isfinite(errors).all() and np.all(errors>=0) and
            np.isfinite(row['mean_clean_nrmse']) and row['mean_clean_nrmse']>=0 and isinstance(row['both_accepted'],bool) and
            np.isfinite(sizes).all() and all(v>=0 for v in sizes))
    except (KeyError,TypeError,ValueError):
        return False


def summarize(scores, verified):
    jobs = study.matrix()
    keys = [(j['case'],j['seed'],j['method']) for j in jobs]
    index = {(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in scores}
    vindex = {(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in verified}
    denominator = len(scores)==len(verified)==len(index)==len(vindex)==32 and set(index)==set(vindex)==set(keys)

    def aggregate(expected):
        rows = [index[key] for key in expected if complete(index.get(key,{}))]
        errors = [float(np.mean(r['clean_nrmse'])) for r in rows]
        mean_error = float(np.mean(errors)) if len(rows)==len(expected) and rows else None
        mean_graph = float(np.mean([r['structure']['shared_graph_size'] for r in rows])) if len(rows)==len(expected) and rows else None
        return dict(expected=len(expected),complete=len(rows),failed_or_missing=len(expected)-len(rows),
            both_accepted=sum(bool(r['both_accepted']) for r in rows),
            mean_clean_nrmse=mean_error if mean_error is not None and np.isfinite(mean_error) else None,
            mean_shared_graph_size=mean_graph if mean_graph is not None and np.isfinite(mean_graph) else None)

    methods = {}
    for method in study.METHODS:
        expected = [key for key in keys if key[2]==method]
        methods[method] = aggregate(expected)
        methods[method]['statuses'] = dict(Counter(index.get(key,{}).get('status','missing') for key in expected))
        methods[method]['strata'] = {label:aggregate([key for key in expected if predicate(key[0])])
            for label,predicate in [(f,lambda c,f=f:cases.design(c)['family']==f) for f in cases.FAMILIES]+
                [('noise0',lambda c:c.endswith('_n0')),('noise1',lambda c:c.endswith('_n1'))]}

    pairs = []
    for case in cases.CASES:
        for seed in cases.seeds(case):
            a,b = (index.get((case,seed,m),{}) for m in ('repair','base'))
            pair = dict(case=case,seed=seed,available=complete(a) and complete(b))
            if pair['available']:
                pair.update(protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),
                    candidate_accepted=bool(a['both_accepted']),control_accepted=bool(b['both_accepted']))
                try:
                    candidate = read(HERE/'runs'/study.name(a['job'])/'model.json')
                    control = read(HERE/'runs'/study.name(b['job'])/'model.json')
                    pair.update(paired_identity(candidate,control))
                except Exception as error:
                    pair['identity_error'] = repr(error)
            pairs.append(pair)
    base_match = len(pairs)==16 and all(p['available'] and p.get('base_prefix_matches',False) and p.get('base_cache_matches',False) for p in pairs)
    skip_match = len(pairs)==16 and all(p['available'] and 'repair_skipped' in p and
        (not p['repair_skipped'] or p['skipped_pool_matches'] and p['skipped_choice_matches']) for p in pairs)
    fields = ('new_evaluations','valid','protected','validation_retained')
    totals = {k:sum(r.get('closure',{}).get(k,0) for r in verified) for k in fields}
    by_method = {m:{k:sum(r.get('closure',{}).get(k,0) for r in verified if r['job']['method']==m)
        for k in fields} for m in study.METHODS}
    closure_mechanism = all(v>0 for row in by_method.values() for v in row.values())
    repair = {k:sum(r.get('repair',{}).get(k,0) for r in verified) for k in
        ('triggered','new_evaluations','valid','protected','validation_retained','linear_subproblems','retained_roots','implicit_new_systems')}
    counts_complete = all(m['complete']==16 and m['mean_clean_nrmse'] is not None for m in methods.values())
    technical = denominator and counts_complete and all(r.get('verified',False) for r in verified) and closure_mechanism and base_match and skip_match
    protected = sum(sum(p.get('protected',[])) for p in pairs)
    preserved = all(p['available'] and (not p['control_accepted'] or p['candidate_accepted']) for p in pairs)
    gains = sum(p.get('candidate_accepted',False) and not p.get('control_accepted',False) for p in pairs if p['available'])
    losses = sum(p.get('control_accepted',False) and not p.get('candidate_accepted',False) for p in pairs if p['available'])
    a,b = methods['repair'],methods['base']
    candidate_all = a['both_accepted']==16 and a['complete']==16
    macro = a['mean_clean_nrmse'] is not None and b['mean_clean_nrmse'] is not None and a['mean_clean_nrmse']<=b['mean_clean_nrmse']
    graph = a['mean_shared_graph_size'] is not None and b['mean_shared_graph_size'] is not None and a['mean_shared_graph_size']<=b['mean_shared_graph_size']
    adoption = technical and candidate_all and protected==32 and preserved and macro and graph
    family_evidence = {}
    for family in cases.FAMILIES:
        family_rows = [r for r in verified if r['job']['method']=='repair' and cases.design(r['job']['case'])['family']==family]
        family_pairs = [p for p in pairs if cases.design(p['case'])['family']==family]
        family_evidence[family] = dict({k:sum(r.get('repair',{}).get(k,0) for r in family_rows) for k in repair},
            expected_systems=4,verified_systems=sum(r.get('verified',False) for r in family_rows),
            acceptance_gains=sum(p.get('candidate_accepted',False) and not p.get('control_accepted',False) for p in family_pairs if p['available']))
    discovery = dict(repair,acceptance_gains=gains,acceptance_losses=losses,net_acceptance_gain=gains-losses,
        actual_implicit_evaluation=all(repair[k]>0 for k in ('triggered','linear_subproblems','retained_roots','implicit_new_systems','valid')),
        validation_retained_discovery=repair['validation_retained']>0,
        acceptance_gain_observed=gains>0,by_family=family_evidence,
        families_with_validation_retained=sum(r['validation_retained']>0 for r in family_evidence.values()),
        families_with_acceptance_gain=sum(r['acceptance_gains']>0 for r in family_evidence.values()),
        claim_scope='Actual events on this fixed matrix; zero triggers or retained candidates do not establish repair discovery benefit across families.')
    return dict(methods=methods,pairs=pairs,expected_systems=32,expected_pairs=16,expected_outputs=32,
        full_denominator=denominator,protected_outputs=protected,technical_pass=bool(technical),
        closure=totals,closure_by_method=by_method,closure_mechanism_pass=closure_mechanism,
        repair=repair,repair_discovery_evidence=discovery,paired_base_identity_pass=base_match,
        untriggered_identity_pass=skip_match,candidate_all_accepted=candidate_all,
        control_acceptance_preserved=preserved,macro_clean_error_nonincrease=bool(macro),
        mean_graph_nonincrease=bool(graph),adoption_pass=bool(adoption),production_default_changed=False,
        claim_scope='Descriptive validation of four fixed synthetic output-pair families; not statistical significance, public benchmark or real engineering acceptance.')


def verify():
    if (HERE/'VERIFICATION.json').exists():
        raise RuntimeError('Keep existing verification.')
    jobs=study.matrix();frozen={};score_rows=[];deadline=None;preparation_error=None
    try:
        deadline=study.budget()
        frozen,sealed = guards()
        require(not (HERE/'SCORING_FAILURE.json').exists(),'Scoring phase failed; see SCORING_FAILURE.json.')
        scores = read(HERE/'SCORES.json')
        require(scores['all_models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Score seal drift.')
        require(sealed['frozen_unix']<=scores['scored_unix'],'Scoring preceded sealing.')
        require(frozen['jobs']==jobs,'Frozen matrix drift.')
        # Validate indexing before accepting even a partial score table.
        require(isinstance(scores['rows'],list),'Invalid score rows.')
        counts = Counter((r['job']['case'],r['job']['seed'],r['job']['method']) for r in scores['rows'])
        index = {(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in scores['rows']}
        score_rows=scores['rows']
    except Exception as error:
        preparation_error=repr(error)
    rows = []
    for job in jobs:
        try:
            require(preparation_error is None,'Verification prerequisites failed: '+str(preparation_error))
            require(time.time()<deadline,'Verification budget expired.')
            key = (job['case'],job['seed'],job['method'])
            score = index.get(key,{})
            require(counts[key]==1 and score.get('job')==job and complete(score),'Incomplete or invalid cell.')
            row = base.verify_cell(job,score,frozen)
            row['closure'],row['repair'] = closure_evidence(job,frozen)
            require(time.time()<deadline,'Verification exceeded the deadline.')
        except Exception as error:
            row = dict(job=job,verified=False,error=repr(error))
        rows.append(row)
        print(study.name(job),row.get('error','verified'),flush=True)
    summary = summarize(score_rows,rows)
    within = deadline is not None and time.time()<deadline
    if not within:
        summary.update(technical_pass=False,adoption_pass=False)
    def receipt_hash(name):
        try:
            return sha(HERE/name)
        except OSError:
            return None
    start=frozen.get('budget',{}).get('start_unix')
    write(HERE/'VERIFICATION.json',dict(status='passed' if summary['technical_pass'] else 'failed',rows=rows,
        expected_systems=32,verified_systems=sum(r['verified'] for r in rows),budget_met=within,verified_unix=time.time(),
        preparation_error=preparation_error,elapsed_seconds=None if start is None else time.time()-start,
        freeze_sha256=receipt_hash('FREEZE.json'),models_sha256=receipt_hash('ALL_MODELS_FROZEN.json'),
        scores_sha256=receipt_hash('SCORES.json')))
    write(HERE/'SUMMARY.json',summary)


def run():
    if any((HERE/name).exists() for name in ('SCORES.json','SCORING_FAILURE.json','VERIFICATION.json')):
        raise RuntimeError('Keep existing scoring or verification evidence.')
    try:
        evaluate()
    except Exception as error:
        write(HERE/'SCORING_FAILURE.json',dict(status='failed',phase='scoring',error=repr(error),failed_unix=time.time()))
    verify()


if __name__=='__main__':
    run()
