"""Score only sealed models; replay training-only repair and closure evidence."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dimensr_mimo import SearchConfig, audit
from dimensr_mimo.graph import System, in_bounds
from dimensr_mimo.selection import FIELDS, digest
from experiments.native_api_20261026.run import read, write, sha
from experiments.native_api_20261026.verify import raw, require
from experiments.compound_search_20261027 import cases
from experiments.precision_repair_20261028 import study

spec = importlib.util.spec_from_file_location('_repair_raw_verifier', HERE.parent/'compound_search_20261027/evaluate.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.HERE, base.study = HERE, study
evaluate = base.evaluate
_base_guards = base.guards


def guards():
    frozen, sealed = _base_guards()
    previous = read(HERE/'TRAINING_STARTED.json')['start_unix']
    for job in frozen['jobs']:
        folder = HERE/'runs'/study.name(job)
        path = folder/'CELL_FROZEN.json'
        cell = read(path)
        require(previous <= cell['frozen_unix'] <= sealed['frozen_unix'], 'Cell sealing chronology drift.')
        previous = cell['frozen_unix']
        require(sealed['files'].get(path.relative_to(HERE).as_posix()) == sha(path), 'Cell seal missing or changed in all-model seal.')
        study.verify_files(cell['files'], folder)
        for name, expected in cell['files'].items():
            require(sealed['files'].get((folder/name).relative_to(HERE).as_posix()) == expected, 'Cell artifact missing or changed in all-model seal.')
    return frozen, sealed


base.guards = guards


def training_payload(model):
    return [{k:c[k] for k in FIELDS} for c in model['candidates']]


def root_payload(graph, constants, output):
    """Extract one raw output, preserving aliases and its warm parameter values."""
    nodes, mapped, params, values = [], {}, {}, []
    def visit(index):
        if index not in mapped:
            node = graph['nodes'][index]
            children = [visit(i) for i in node['children']]
            parameter = node['parameter']
            if parameter is not None:
                if parameter not in params:
                    params[parameter] = len(values)
                    values.append(constants[parameter])
                parameter = params[parameter]
            nodes.append(dict(node,children=children,parameter=parameter))
            mapped[index] = len(nodes)-1
        return mapped[index]
    root = visit(graph['roots'][output])
    return dict(nodes=nodes,roots=[root]), values


def extension_evidence(model, X, Y, cfg, events, known, *, parent=None):
    """Interpret saved graphs and recompute the per-output training guard."""
    retained, valid = [], 0
    graphs = set()
    for event in events:
        require(digest(event['parent']) in known, 'Parent was not in the base/retained pool.')
        if parent is not None:
            require(event['parent']==parent, 'Repair parent is not the fixed training optimum.')
        graph = System.from_dict(event['graph'])
        require(in_bounds(graph, cfg), 'Invalid extension structure.')
        graph_hash = digest(event['graph'])
        require(graph_hash not in graphs, 'Repeated extension evaluation graph.')
        graphs.add(graph_hash)
        require(len(event['warm'])==graph.n_parameters and np.isfinite(event['warm']).all(), 'Invalid warm parameters.')
        audit.units(graph, model['feature_units'], model['target_units'], model['constant_units'])
        np.testing.assert_allclose(base.error(raw(event['parent']['system'],event['parent']['constants'],X),Y,model['scales']),
            event['parent']['errors'], rtol=1e-9, atol=1e-11)
        child = event['child']; protected = False
        if child is not None:
            valid += 1
            require(child['system']==event['graph'] and event['new_evaluation'], 'Child provenance mismatch.')
            np.testing.assert_allclose(base.error(raw(child['system'],child['constants'],X),Y,model['scales']),
                child['errors'], rtol=1e-9, atol=1e-11)
            protected = all(a<=b*1.1+1e-8 for a,b in zip(child['errors'],event['parent']['errors']))
            if protected:
                retained.append(child)
                known.add(digest(child))
        require(protected==event['protected'], 'Training parent guard drift.')
    return retained, valid


def accounting(record, retained, valid, limit):
    require(0<=record['new_evaluations']<=limit and
        sum(event['new_evaluation'] for event in record['events'])==record['new_evaluations'], 'New evaluation accounting drift.')
    require(record['valid']==valid and record['protected']==len(retained) and
        record['protection_rejections']==valid-len(retained), 'Protection accounting drift.')


def repair_evidence(model, payload, X, Y, cfg, known):
    report = model['diagnostics']['sharing_closure'].get('precision_repair')
    if not cfg.precision_repair:
        require(report is None, 'Base control used precision repair.')
        return [], {}
    require(report is not None, 'Missing precision-repair report.')
    eligible = sum(all(a<=b for a,b in zip(c['errors'],cfg.acceptance['nrmse'])) for c in payload)
    require(report['training_eligible']==eligible and report['triggered']==(eligible==0), 'Training-only trigger drift.')
    limit = min(12,cfg.population_size//len(Y.T))
    require(report['seed_limit']==limit, 'Repair seed limit drift.')
    stats = report['initialization_diagnostics']
    require(len(stats)<=len(Y.T), 'Too many repair output initializers.')
    for row in stats:
        require(0<=row.get('implicit_denominators',0)<=64, 'Implicit feature budget exceeded.')
        slots = row.get('implicit_retained_indices',[])
        require(len(slots)==len(set(slots)) and all(0<=i<min(limit,row.get('proposed_structures',0)) for i in slots), 'Invalid implicit retained position.')
    if eligible:
        require(not stats and not report['events'] and report['stop_reason']=='training_precision_met', 'Skipped repair performed work.')
        require(all(report[k]==0 for k in ('new_evaluations','valid','protected','protection_rejections',
            'cache_skips','nonimplicit_skips','structure_rejections')), 'Skipped repair changed counters.')
        retained, valid = [], 0
    else:
        parent = min(payload,key=lambda c:(float(np.mean(c['errors'])),c['complexity'],digest(c['system'])))
        require(report['parent']==parent, 'Repair training parent drift.')
        indices = [event['seed_index'] for event in report['events']]
        require(indices==sorted(set(indices)), 'Repair seed order drift.')
        for event in report['events']:
            require(len(stats)==len(Y.T), 'Repair event preceded all output initializers.')
            lengths = [min(limit,row['proposed_structures']) for row in stats]
            k = event['seed_index']
            require(0<=k<max(lengths), 'Repair system index exceeded seed pool.')
            implicit = [j for j,row in enumerate(stats) if k<lengths[j] and k in row['implicit_retained_indices']]
            require(bool(implicit) and event['implicit_outputs']==implicit, 'Event lacks an actual retained implicit root.')
            for j,length in enumerate(lengths):
                if k>=length:
                    require(root_payload(event['graph'],event['warm'],j)==root_payload(parent['system'],parent['constants'],j),
                        'Missing repair root was not copied from the fixed parent.')
        retained, valid = extension_evidence(model,X,Y,cfg,report['events'],known,parent=parent)
    accounting(report,retained,valid,12)
    require(report['stop_reason'] in ('training_precision_met','completed','deadline','global_budget','repair_budget'), 'Unknown repair stop reason.')
    return retained, dict(triggered=int(report['triggered']), training_eligible=eligible,
        new_evaluations=report['new_evaluations'], valid=valid, protected=len(retained),
        linear_subproblems=sum(row.get('implicit_linear_subproblems',0) for row in stats),
        retained_roots=sum(len(row.get('implicit_retained_indices',[])) for row in stats),
        implicit_new_systems=sum(bool(event['implicit_outputs']) and event['new_evaluation'] for event in report['events']))


def closure_evidence(job, frozen):
    folder = HERE/'runs'/study.name(job)
    model = read(folder/'model.json'); cfg = SearchConfig(**model['config'])
    expected = SearchConfig(**dict(frozen['config'],random_state=job['seed'],initialization='compound',
        sharing_strategy='closure',precision_repair=job['method']=='repair',final_selection='precision',
        acceptance=cases.design(job['case'])['acceptance']))
    require(digest(model['config'])==digest(asdict(expected))==digest(read(folder/'result.json')['effective_config']), 'Effective configuration drift.')
    resources = read(folder/'resources.json')
    require(resources['wall_seconds']<=job['seconds']+.5 and resources['returncode']==0, 'Resource bound failure.')
    require(model['diagnostics']['evaluations']<=cfg.max_evaluations, 'Total evaluation cap exceeded.')
    record = model['diagnostics']['sharing_closure']; events = record['events']
    n = record['base_candidate_count']; payload = training_payload(model)[:n]
    require(0<n<=len(model['candidates']) and digest(payload)==record['base_candidate_sha256'], 'Base pool drift.')
    X,Y = cases.training_data(job['case'],'train',job['seed'])
    V,Z = cases.training_data(job['case'],'validation',job['seed'])
    known = {digest(c) for c in payload}
    repair_retained, repair = repair_evidence(model,payload,X,Y,cfg,known)
    require(0<=record['rounds']<=2 and all(event['round'] in range(record['rounds']) for event in events), 'Closure round drift.')
    require(all(sum(event['new_evaluation'] for event in events if event['round']==r)<=24 for r in range(2)), 'Closure round cap exceeded.')
    all_events = record.get('precision_repair',{}).get('events',[])+events
    evaluated_graphs = [digest(event['graph']) for event in all_events]
    require(len(evaluated_graphs)==len(set(evaluated_graphs)), 'Extension stages reused a cached graph.')
    for event in events:
        require(System.from_dict(event['graph']).size<System.from_dict(event['parent']['system']).size, 'Closure did not reduce structure.')
    retained, valid = extension_evidence(model,X,Y,cfg,events,known)
    accounting(record,retained,valid,48)
    require(record['base_evaluations']+repair.get('new_evaluations',0)+record['new_evaluations']==model['diagnostics']['evaluations'], 'Evaluation total drift.')
    require(digest(payload+repair_retained+retained)==record['full_training_pool_sha256'], 'Full training pool drift.')
    acceptable, rejected, types = [], [], []
    for kind, children in (('repair',repair_retained),('closure',retained)):
        for child in children:
            pred = raw(child['system'],child['constants'],V)
            with np.errstate(all='ignore'):
                errors = np.sqrt(np.mean(((pred-Z)/model['scales'])**2,axis=0))
            ok = np.isfinite(pred).all() and np.isfinite(errors).all()
            if cfg.output_bounds is not None:
                lo,hi = cfg.output_bounds
                ok = ok and (lo is None or np.all(pred>=lo)) and (hi is None or np.all(pred<=hi))
            (acceptable if ok else rejected).append(child)
            if ok: types.append(kind)
    require(training_payload(model)[n:]==acceptable, 'Validation extension subset drift.')
    require(record['validation_retained_extensions']==len(acceptable), 'Retained extension count drift.')
    require([row['graph_sha256'] for row in record.get('validation_rejections',[])]==[digest(c['system']) for c in rejected], 'Validation rejection record drift.')
    post = model['post_selection']; i = post['base_candidate_index']
    require(0<=i<n and digest(payload[i])==post['base_choice_sha256'], 'Original base choice hash drift.')
    require(all(a<=b for a,b in zip(model['best']['errors'],post['base_training_bounds'])), 'Final training protection failed.')
    require(all(a<=b for a,b in zip(model['validation_nrmse'],post['base_validation_bounds'])), 'Final validation protection failed.')
    history = next(row for row in model['history'] if row['phase']=='sharing_closure')
    runtime = next(json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()
        if json.loads(line)['phase']=='sharing_closure')
    require(all(runtime[k]==v for k,v in history.items()), 'Closure runtime event drift.')
    require(history['new_evaluations']==record['new_evaluations']+repair.get('new_evaluations',0) and
        history['new_systems']==record['new_evaluations'] and history['protected_systems']==record['protected'], 'Closure history accounting drift.')
    require(history.get('repair_new_systems',0)==repair.get('new_evaluations',0) and
        history.get('repair_protected_systems',0)==repair.get('protected',0), 'Repair history accounting drift.')
    if repair: repair['validation_retained'] = types.count('repair')
    return dict(new_evaluations=record['new_evaluations'],valid=valid,protected=len(retained),
        validation_retained=types.count('closure'),base_candidate_count=n,base_evaluations=record['base_evaluations'],
        base_search_fingerprint=record['base_search_fingerprint'],base_candidate_sha256=record['base_candidate_sha256']), repair


def paired_identity(candidate, control):
    a,b = (m['diagnostics']['sharing_closure'] for m in (candidate,control))
    prefix = all(a[k]==b[k] for k in ('base_candidate_count','base_candidate_sha256','base_evaluations','base_search_fingerprint'))
    skipped = not a['precision_repair']['triggered']
    pool = candidate['candidates']==control['candidates'] and candidate['search_fingerprint']==control['search_fingerprint']
    selected = candidate['best']==control['best'] and candidate['validation_nrmse']==control['validation_nrmse']
    return dict(base_prefix_matches=prefix,base_cache_matches=a['base_search_fingerprint']==b['base_search_fingerprint'],
        repair_skipped=skipped,skipped_pool_matches=pool if skipped else None,skipped_choice_matches=selected if skipped else None)


def summarize(scores, verified):
    jobs = study.matrix(); keys = [(j['case'],j['seed'],j['method']) for j in jobs]
    index = {(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in scores}
    vindex = {(r['job']['case'],r['job']['seed'],r['job']['method']):r for r in verified}
    denominator = len(scores)==len(verified)==len(index)==len(vindex)==34 and set(index)==set(vindex)==set(keys)
    methods = {}
    for method in study.METHODS:
        expected = [key for key in keys if key[2]==method]
        complete = [index[key] for key in expected if index.get(key,{}).get('status')=='complete']
        methods[method] = dict(expected=len(expected),complete=len(complete),both_accepted=sum(r['both_accepted'] for r in complete),
            mean_clean_nrmse=float(np.mean([r['mean_clean_nrmse'] for r in complete])) if len(complete)==len(expected) else None,
            mean_shared_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in complete])) if len(complete)==len(expected) else None)
    pairs = []
    for case in cases.CASES:
        for seed in cases.seeds(case):
            a,b = (index.get((case,seed,method),dict(status='missing')) for method in ('repair','base'))
            pair = dict(case=case,seed=seed,available=a['status']==b['status']=='complete')
            if pair['available']:
                candidate = read(HERE/'runs'/study.name(a['job'])/'model.json')
                control = read(HERE/'runs'/study.name(b['job'])/'model.json')
                pair.update(paired_identity(candidate,control),protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),
                    candidate_accepted=a['both_accepted'],control_accepted=b['both_accepted'])
            pairs.append(pair)
    base_match = len(pairs)==17 and all(p['available'] and p['base_prefix_matches'] and p['base_cache_matches'] for p in pairs)
    skip_match = all(p['available'] and (not p['repair_skipped'] or p['skipped_pool_matches'] and p['skipped_choice_matches']) for p in pairs)
    fields = ('new_evaluations','valid','protected','validation_retained')
    totals = {k:sum(r.get('closure',{}).get(k,0) for r in verified) for k in fields}
    by_method = {m:{k:sum(r.get('closure',{}).get(k,0) for r in verified if r['job']['method']==m) for k in fields} for m in study.METHODS}
    closure_mechanism = all(v>0 for row in by_method.values() for v in row.values())
    repair = {k:sum(r.get('repair',{}).get(k,0) for r in verified) for k in
        ('triggered','new_evaluations','valid','protected','validation_retained','linear_subproblems','retained_roots','implicit_new_systems')}
    repair_mechanism = all(v>0 for k,v in repair.items() if k!='validation_retained')
    technical = denominator and all(r.get('verified',False) for r in verified) and closure_mechanism and repair_mechanism and base_match and skip_match
    protected = sum(sum(p.get('protected',[])) for p in pairs)
    def target_pass(case,seed,threshold,compact=False):
        row = index.get((case,seed,'repair'),dict(status='missing'))
        if row['status']!='complete': return False
        model = read(HERE/'runs'/study.name(row['job'])/'model.json')
        return bool(row['both_accepted'] and (not compact or row['structure']['shared_graph_size']<=20) and
            max(model['best']['errors']+model['validation_nrmse']+row['clean_nrmse'])<=threshold)
    compact = target_pass('damped_n0',701,1e-12,True)
    rational = target_pass('rational_n0',702,1e-4)
    preserved = all(p['available'] and (not p['control_accepted'] or p['candidate_accepted']) for p in pairs)
    a,b = methods['repair'],methods['base']
    adoption = (technical and protected==34 and compact and rational and preserved and
        a['both_accepted']>=b['both_accepted'] and a['mean_clean_nrmse']<=b['mean_clean_nrmse'])
    return dict(methods=methods,pairs=pairs,expected_systems=34,expected_outputs=34,full_denominator=denominator,
        protected_outputs=protected,technical_pass=bool(technical),closure=totals,closure_by_method=by_method,
        closure_mechanism_pass=closure_mechanism,repair=repair,repair_mechanism_pass=repair_mechanism,
        paired_base_identity_pass=base_match,untriggered_identity_pass=skip_match,compact_damped_gain_preserved=compact,
        rational_training_precision_gained=rational,control_acceptance_preserved=preserved,adoption_pass=bool(adoption),
        production_default_changed=False,claim_scope='Historical development regression; no new-task validation.')


def verify():
    if (HERE/'VERIFICATION.json').exists(): raise RuntimeError('Keep existing verification.')
    frozen,sealed = base.guards(); scores = read(HERE/'SCORES.json')
    require(scores['all_models_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'), 'Score seal drift.')
    require(sealed['frozen_unix']<=scores['scored_unix'], 'Scoring preceded sealing.')
    require(len(scores['rows'])==len(frozen['jobs'])==34, 'Incomplete score denominator.')
    rows = []
    for job,score in zip(frozen['jobs'],scores['rows']):
        try:
            require(time.time()<study.budget(), 'Verification budget expired.')
            require(score['job']==job and score['status']=='complete', 'Incomplete cell.')
            row = base.verify_cell(job,score,frozen)
            row['closure'],row['repair'] = closure_evidence(job,frozen)
            require(time.time()<study.budget(), 'Verification exceeded the deadline.')
        except Exception as error:
            row = dict(job=job,verified=False,error=repr(error))
        rows.append(row); print(study.name(job),row.get('error','verified'),flush=True)
    summary = summarize(scores['rows'],rows); within = time.time()<study.budget()
    if not within: summary.update(technical_pass=False,adoption_pass=False)
    write(HERE/'VERIFICATION.json',dict(status='passed' if summary['technical_pass'] else 'failed',rows=rows,
        expected_systems=34,verified_systems=sum(r['verified'] for r in rows),budget_met=within,verified_unix=time.time(),
        elapsed_seconds=time.time()-frozen['budget']['start_unix'],freeze_sha256=sha(HERE/'FREEZE.json'),
        models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json')))
    write(HERE/'SUMMARY.json',summary)


if __name__=='__main__':
    evaluate(); verify()
