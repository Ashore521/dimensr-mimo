"""Score sealed graphs and verify closure provenance with independent predictions."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dimensr_mimo import audit
from dimensr_mimo.graph import System, in_bounds
from dimensr_mimo import SearchConfig
from dimensr_mimo.selection import FIELDS, digest
from experiments.native_api_20261026.run import read, write, sha
from experiments.native_api_20261026.verify import raw, require
from experiments.compound_search_20261027 import cases
from experiments.implicit_rational_20261028 import study

spec = importlib.util.spec_from_file_location('_closure_raw_verifier', HERE.parent/'compound_search_20261027/evaluate.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE, base.study = HERE, study
evaluate = base.evaluate


def closure_evidence(job, frozen):
    folder = HERE/'runs'/study.name(job)
    m = read(folder/'model.json'); cfg = SearchConfig(**m['config'])
    expected=SearchConfig(**dict(frozen['config'],random_state=job['seed'],initialization=job['method'],sharing_strategy='closure',
        final_selection='precision',acceptance=cases.design(job['case'])['acceptance']))
    require(digest(m['config'])==digest(asdict(expected))==digest(read(folder/'result.json')['effective_config']), 'Effective configuration drift.')
    resources=read(folder/'resources.json')
    require(resources['wall_seconds']<=job['seconds']+.5 and resources['returncode']==0, 'Resource bound failure.')
    require(cfg.sharing_strategy == 'closure' and cfg.initialization == job['method'], 'Ablation strategy drift.')
    require(m['diagnostics']['evaluations'] <= cfg.max_evaluations, 'Total evaluation cap exceeded.')
    record = m['diagnostics']['sharing_closure']; events = record['events']
    n = record['base_candidate_count']
    payload = [{k:c[k] for k in FIELDS} for c in m['candidates'][:n]]
    require(digest(payload) == record['base_candidate_sha256'], 'Base pool drift.')
    require(record['rounds'] <= 2 and record['new_evaluations'] <= 48, 'Closure resource cap exceeded.')
    require(all(sum(e['new_evaluation'] for e in events if e['round']==r)<=24 for r in range(2)), 'Closure round cap exceeded.')
    require(sum(e['new_evaluation'] for e in events) == record['new_evaluations'], 'New evaluation accounting drift.')
    require(record['base_evaluations']+record['new_evaluations'] == m['diagnostics']['evaluations'], 'Evaluation total drift.')
    X,Y = cases.training_data(job['case'], 'train', job['seed'])
    V,Z = cases.training_data(job['case'], 'validation', job['seed'])
    known = {digest(c) for c in payload}
    retained, valid = [], 0
    for e in events:
        require(digest(e['parent']) in known, 'Parent was not in the base/retained pool.')
        graph = System.from_dict(e['graph']); parent = System.from_dict(e['parent']['system'])
        require(graph.size < parent.size and in_bounds(graph, cfg), 'Invalid closure structure.')
        require(e['round'] in (0,1), 'Invalid round.')
        audit.units(graph, m['feature_units'], m['target_units'], m['constant_units'])
        np.testing.assert_allclose(base.error(raw(e['parent']['system'],e['parent']['constants'],X),Y,m['scales']),
            e['parent']['errors'], rtol=1e-9, atol=1e-11)
        child = e['child']; protected = False
        if child is not None:
            valid += 1
            require(child['system'] == e['graph'] and e['new_evaluation'], 'Child provenance mismatch.')
            np.testing.assert_allclose(base.error(raw(child['system'],child['constants'],X),Y,m['scales']),
                child['errors'], rtol=1e-9, atol=1e-11)
            protected = all(a<=b*1.1+1e-8 for a,b in zip(child['errors'],e['parent']['errors']))
            if protected:
                retained.append(child); known.add(digest(child))
        require(protected == e['protected'], 'Training parent guard drift.')
    require(record['valid']==valid and record['protected']==len(retained) and record['protection_rejections']==valid-len(retained), 'Protection accounting drift.')
    require(digest(payload+retained) == record['full_training_pool_sha256'], 'Full training pool drift.')
    acceptable, rejected = [], []
    for child in retained:
        pred = raw(child['system'], child['constants'], V)
        with np.errstate(all='ignore'):
            errors = np.sqrt(np.mean(((pred-Z)/m['scales'])**2, axis=0))
        ok = np.isfinite(pred).all() and np.isfinite(errors).all()
        if cfg.output_bounds is not None:
            lo,hi = cfg.output_bounds
            ok = ok and (lo is None or np.all(pred>=lo)) and (hi is None or np.all(pred<=hi))
        (acceptable if ok else rejected).append(child)
    require([{k:c[k] for k in FIELDS} for c in m['candidates'][n:]] == acceptable, 'Validation extension subset drift.')
    require(record['validation_retained_extensions']==len(acceptable), 'Retained extension count drift.')
    require([r['graph_sha256'] for r in record.get('validation_rejections',[])] == [digest(c['system']) for c in rejected], 'Validation rejection record drift.')
    post = m['post_selection']; baseline = payload[post['base_candidate_index']]
    require(digest(baseline)==post['base_choice_sha256'], 'Base choice hash drift.')
    require(all(a<=b for a,b in zip(m['best']['errors'],post['base_training_bounds'])), 'Final training protection failed.')
    require(all(a<=b for a,b in zip(m['validation_nrmse'],post['base_validation_bounds'])), 'Final validation protection failed.')
    return dict(new_evaluations=record['new_evaluations'],valid=valid,protected=len(retained),
        validation_retained=len(acceptable),base_search_fingerprint=record['base_search_fingerprint'],
        base_candidate_sha256=record['base_candidate_sha256'])


def initializer_evidence(job):
    folder=HERE/'runs'/study.name(job);model=read(folder/'model.json')
    history=next(r for r in model['history'] if r['phase']=='initialization')
    import json
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    event=next(r for r in events if r['phase']=='initialization')
    require(all(event[k]==v for k,v in history.items()), 'Initialization event drift.')
    if job['method']=='compound':
        require('implicit_seed_jobs' not in history, 'Control used implicit seeds.')
        return {}
    jobs,new,valid=(history[k] for k in ('implicit_seed_jobs','implicit_seed_new_evaluations','implicit_seed_valid'))
    require(0<=valid<=new<=jobs and new<=history['new_evaluations'], 'Initializer accounting drift.')
    stats=model['diagnostics']['initialization_diagnostics']
    for row in stats:
        require(row['implicit_denominators']<=64, 'Implicit feature budget exceeded.')
        require(all(0<=i<12 for i in row['implicit_retained_indices']), 'Invalid retained implicit position.')
    return dict(jobs=jobs,new_evaluations=new,valid=valid,
        linear_subproblems=sum(r['implicit_linear_subproblems'] for r in stats),
        retained_roots=sum(len(r['implicit_retained_indices']) for r in stats))


def summarize(scores, verified):
    index={(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in scores}
    methods={}
    for method in study.METHODS:
        rows=[r for r in scores if r['job']['method']==method]; complete=[r for r in rows if r['status']=='complete']
        methods[method]=dict(expected=len(rows),complete=len(complete),both_accepted=sum(r['both_accepted'] for r in complete),
            mean_clean_nrmse=float(np.mean([r['mean_clean_nrmse'] for r in complete])) if len(complete)==len(rows) else None,
            mean_shared_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in complete])) if len(complete)==len(rows) else None)
    pairs=[]
    for case in cases.CASES:
        for seed in cases.seeds(case):
            a,b=index[case,seed,'implicit'],index[case,seed,'compound']
            pair=dict(case=case,seed=seed,available=a['status']==b['status']=='complete')
            if pair['available']:
                new=read(HERE/'runs'/study.name(a['job'])/'model.json');old=read(HERE/'runs'/study.name(b['job'])/'model.json')
                pair.update(protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),
                    candidate_accepted=a['both_accepted'],control_accepted=b['both_accepted'],
                    search_fingerprint_matches_control=new['search_fingerprint']==old['search_fingerprint'])
            pairs.append(pair)
    technical=len(verified)==34 and all(r['verified'] for r in verified)
    protected=sum(sum(p.get('protected',[])) for p in pairs)
    totals={k:sum(r.get('closure',{}).get(k,0) for r in verified) for k in ('new_evaluations','valid','protected','validation_retained')}
    closure_by_method={method:{k:sum(r.get('closure',{}).get(k,0) for r in verified if r['job']['method']==method)
        for k in totals} for method in study.METHODS}
    closure_mechanism=all(v>0 for row in closure_by_method.values() for v in row.values())
    implicit_totals={k:sum(r.get('initializer',{}).get(k,0) for r in verified) for k in
        ('jobs','new_evaluations','valid','linear_subproblems','retained_roots')}
    implicit_mechanism=all(v>0 for v in implicit_totals.values())
    technical=technical and closure_mechanism and implicit_mechanism
    target=index['damped_n0',701,'implicit']
    compact=False
    if target['status']=='complete':
        m=read(HERE/'runs'/study.name(target['job'])/'model.json')
        compact=(target['both_accepted'] and target['structure']['shared_graph_size']<=20 and
            max(m['best']['errors']+m['validation_nrmse']+target['clean_nrmse'])<=1e-12)
    rational=index['rational_n0',702,'implicit'];rational_gain=False
    if rational['status']=='complete':
        model=read(HERE/'runs'/study.name(rational['job'])/'model.json')
        rational_gain=(rational['both_accepted'] and max(model['best']['errors']+model['validation_nrmse']+rational['clean_nrmse'])<=1e-4)
    acceptance_preserved=all(p['available'] and (not p['control_accepted'] or p['candidate_accepted']) for p in pairs)
    a,b=methods['implicit'],methods['compound']
    adoption=(technical and protected==34 and totals['protected']>0 and compact and rational_gain and acceptance_preserved and
        a['both_accepted']>=b['both_accepted'] and a['mean_clean_nrmse']<=b['mean_clean_nrmse'])
    return dict(methods=methods,pairs=pairs,expected_outputs=34,protected_outputs=protected,technical_pass=technical,
        closure=totals,closure_by_method=closure_by_method,closure_mechanism_pass=closure_mechanism,
        initializer=implicit_totals,implicit_mechanism_pass=implicit_mechanism,
        compact_damped_gain_preserved=bool(compact),rational_training_precision_gained=bool(rational_gain),
        control_acceptance_preserved=acceptance_preserved,adoption_pass=bool(adoption),production_default_changed=False,
        claim_scope='Historical development regression; no new-task validation.')


def verify():
    if (HERE/'VERIFICATION.json').exists():
        raise RuntimeError('Keep existing verification.')
    frozen,sealed=base.guards();scores=read(HERE/'SCORES.json')
    require(scores['all_models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Score seal drift.')
    require(sealed['frozen_unix']<=scores['scored_unix'],'Scoring preceded sealing.')
    require(len(scores['rows'])==len(frozen['jobs'])==34,'Incomplete score denominator.')
    rows=[]
    for job,score in zip(frozen['jobs'],scores['rows']):
        try:
            require(time.time()<study.budget(),'Verification budget expired.')
            require(score['job']==job and score['status']=='complete','Incomplete cell.')
            row=base.verify_cell(job,score,frozen);row['closure']=closure_evidence(job,frozen);row['initializer']=initializer_evidence(job)
            require(time.time()<study.budget(),'Verification exceeded the deadline.')
        except Exception as error:
            row=dict(job=job,verified=False,error=repr(error))
        rows.append(row);print(study.name(job),row.get('error','verified'),flush=True)
    summary=summarize(scores['rows'],rows);within=time.time()<study.budget()
    if not within:summary.update(technical_pass=False,adoption_pass=False)
    write(HERE/'VERIFICATION.json',dict(status='passed' if summary['technical_pass'] else 'failed',rows=rows,
        expected_systems=34,verified_systems=sum(r['verified'] for r in rows),budget_met=within,
        verified_unix=time.time(),elapsed_seconds=time.time()-frozen['budget']['start_unix'],
        freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json')))
    write(HERE/'SUMMARY.json',summary)


if __name__=='__main__':
    evaluate();verify()
