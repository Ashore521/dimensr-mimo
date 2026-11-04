"""Audited raw replay plus explicit completion provenance and fixed-batch gates."""
from collections import Counter
from dataclasses import asdict
import json
import time
import numpy as np

from dimensr_mimo import SearchConfig, audit
from dimensr_mimo._single.expression import Node
from dimensr_mimo._single.units import unit
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import Builder, System
from dimensr_mimo.genetics import clone_subgraph
from dimensr_mimo.selection import digest
from experiments.native_api_20261026.run import read, write, sha
from experiments.native_api_20261026.verify import raw, require
from experiments.compositional_completion_20261031 import harness

COUNTS = ('new_evaluations','valid','protected','validation_retained')
COMPLETION_COUNTS = ('triggered','new_evaluations','valid','protected','validation_retained',
    'compositional_linear_subproblems','implicit_linear_subproblems','linear_subproblems',
    'compositional_returned_roots','implicit_returned_roots','compositional_event_roots',
    'implicit_event_roots','compositional_new_systems','compositional_valid_systems','compositional_protected_systems')
TIME_KEYS = {'constant_fit_seconds','evaluation_seconds','elapsed_seconds','cpu_seconds','phase_seconds',
    'worker_wall_seconds','worker_cpu_seconds','monotonic'}


def key(row):
    job = row['job']
    return job['case'],job['seed'],job['method']


def complete(row):
    try:
        errors = np.asarray([row['clean_nrmse'],row['observed_nrmse']],float)
        sizes = [row['structure'][n] for n in ('shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth')]
        return bool(row['status']=='complete' and errors.shape==(2,2) and np.isfinite(errors).all() and (errors>=0).all()
            and np.isfinite(row['mean_clean_nrmse']) and row['mean_clean_nrmse']>=0 and isinstance(row['both_accepted'],bool)
            and np.isfinite(sizes).all() and all(type(n) is int and n>=0 for n in sizes))
    except (KeyError,TypeError,ValueError):
        return False


def normalized(value):
    """Only declared timing/version/default and their derived source hash differ."""
    if isinstance(value,list):
        return [normalized(v) for v in value]
    if isinstance(value,dict):
        out = {k:normalized(v) for k,v in value.items() if k not in TIME_KEYS|{'package_version','schema_version','source_model_sha256'}}
        if 'config' in out:
            out['config'].setdefault('compositional_completion',False)
        if 'effective_config' in out:
            out['effective_config'].setdefault('compositional_completion',False)
        if 'job' in out:
            out['job'].pop('method',None)
        return out
    return value


class Verifier:
    def __init__(self,study):
        self.study = study
        self.legacy = harness.isolated(study.ROOT/'experiments/precision_repair_20261028/evaluate.py',
            '_completion_'+study.STAGE+'_raw_verifier')
        self.base = self.legacy.base
        for module in (self.legacy,self.base):
            module.HERE,module.cases,module.study = study.HERE,study.cases,study
        self.base.guards = self.guards
        original_error = self.base.error
        def finite_error(pred,true,scale):
            result = original_error(pred,true,scale)
            require(np.isfinite(result).all() and np.all(np.asarray(result)>=0),'Invalid normalized prediction error.')
            return result
        self.base.error = finite_error

    @property
    def here(self):
        return self.study.HERE

    def guards(self):
        s = self.study
        s.check_candidate_lock();s.check_stage_gate()
        frozen,sealed = self.legacy.guards()
        require(frozen['jobs']==s.matrix(),'Frozen matrix drift.')
        require(frozen['candidate_lock_sha256']==sha(s.SHARED/'CANDIDATE_LOCK.json'),'Frozen source lock drift.')
        return frozen,sealed

    def evaluate(self):
        return self.base.evaluate()

    def completion_evidence(self,model,payload,X,Y,cfg,known):
        record = model['diagnostics']['sharing_closure']
        report = record.get('compositional_completion')
        if not cfg.compositional_completion:
            require(report is None,'Control used compositional completion.')
            return [],{}
        require(not cfg.precision_repair and record.get('precision_repair') is None,'Both supplements enabled.')
        require(report['mode']=='compositional_completion_v1' and report['triggered'] is True,'Completion must use always mode.')
        parent = min(payload,key=lambda c:(float(np.mean(c['errors'])),c['complexity'],digest(c['system'])))
        require(report['parent']==parent,'Completion parent drift.')
        require(report['training_eligible']==sum(all(a<=b for a,b in zip(c['errors'],cfg.acceptance['nrmse'])) for c in payload),'Training eligibility drift.')
        limit = min(12,cfg.population_size//Y.shape[1])
        require(report['seed_limit']==limit,'Completion seed limit drift.')
        require(all(type(report[k]) is int and report[k]>=0 for k in ('new_evaluations','valid','protected',
            'protection_rejections','cache_skips','structure_rejections','nonimplicit_skips','missing_roots','linear_subproblems')),'Invalid completion counts.')
        require(report['stop_reason'] in ('completed','deadline','global_budget','completion_budget'),'Unknown completion stop reason.')
        stats = report['initialization_diagnostics']
        require(len(stats)<=Y.shape[1],'Too many initializers.')
        grammar = SystemGrammar(tuple(map(unit,model['feature_units'])),tuple(map(unit,model['target_units'])),
            tuple(map(unit,model['constant_units'])),cfg)
        for j,row in enumerate(stats):
            require(all(type(row.get(k,0)) is int and row.get(k,0)>=0 for k in
                ('linear_subproblems','compositional_linear_subproblems','implicit_linear_subproblems')),'Invalid initializer counts.')
            sources = row.get('returned_sources',[])
            require(len(sources)<=limit,'Too many returned roots.')
            require([r['returned_index'] for r in sources]==list(range(len(sources))),'Returned root indexing drift.')
            ranks = []
            for source in sources:
                require(source['kind'] in ('structured','implicit','compositional'),'Unknown root source.')
                require(type(source['proposal_index']) is int and source['proposal_index']>=0,'Invalid proposal index.')
                tree = Node.from_dict(source['tree']);values = np.asarray(source['constants'],float)
                require(values.shape==(tree.n_constants,) and np.isfinite(values).all(),'Invalid source constants.')
                require(tree.size<=cfg.max_nodes and tree.depth<=cfg.max_depth,'Source structure bound drift.')
                grammar.plain[j].trace(tree)
                canonical,warm = tree.canonical(values)
                require(tree==canonical and np.array_equal(warm,values),'Source root is not canonical.')
                b = Builder();b.roots.append(b.tree(tree,values));graph,c = b.finish()
                prediction = raw(graph.to_dict(),c,X)[:,0]
                with np.errstate(all='ignore'):
                    score = float(np.mean((prediction-Y[:,j])**2))/max(float(np.var(Y[:,j])),1e-12)+1e-8*tree.size
                require(np.isfinite(prediction).all() and np.isfinite(score) and score>=0,'Nonfinite source score.')
                require(type(source['score']) in (int,float) and np.isfinite(source['score']) and source['score']>=0,'Invalid declared source score.')
                np.testing.assert_allclose(score,source['score'],rtol=1e-9,atol=1e-12)
                ranks.append((source['score'],tree.size,repr(tree)))
                if source['kind']=='compositional':
                    require(source['transform'] in ('log','sqrt') and source['form'] in ('direct','multiply','divide'),'Invalid composition source.')
                    require(type(source['intercept']) is bool and type(source['outer_amplitude']) is bool,'Invalid composition flags.')
                    indices = source['basis_indices']
                    require(len(indices) in (1,2) and indices==sorted(set(indices)) and all(type(i) is int and 0<=i<24 for i in indices),'Invalid basis provenance.')
                    require(source['proposal_index']>=min(limit,row.get('proposed_structures',0)),'Composition mislabeled as old proposal.')
                    self.outer_source(tree,source)
                else:
                    require(source['original_index']==source['proposal_index'] and 0<=source['original_index']<limit,'Old source position drift.')
                    require((source['kind']=='implicit')==(source['original_index'] in row.get('structured_implicit_retained_indices',[])),'Original implicit source drift.')
            require(ranks==sorted(ranks),'Merged root ranking drift.')
            require(len({digest(r['tree']) for r in sources})==len(sources),'Duplicate returned source trees.')
            partial = not sources and report['stop_reason']=='deadline' and not report['events'] and j==len(stats)-1
            if not partial:
                for kind in ('implicit','compositional'):
                    require(row.get(kind+'_retained_indices',[])==[i for i,r in enumerate(sources) if r['kind']==kind],'Retained source indices drift.')
            else:
                # Before merge completion these are original initializer positions.
                slots = row.get('implicit_retained_indices',[])
                require(len(slots)==len(set(slots)) and all(type(i) is int and 0<=i<limit for i in slots),'Partial initializer index drift.')
            require(0<=row.get('implicit_denominators',0)<=64 and 0<=row.get('compositional_basis_count',0)<=24,'Initializer budget drift.')
        for name in ('compositional_linear_subproblems','implicit_linear_subproblems'):
            require(report[name]==sum(row.get(name,0) for row in stats) and type(report[name]) is int and report[name]>=0,'Linear work accounting drift.')
        require(report['linear_subproblems']==sum(row.get('linear_subproblems',0)+row.get('compositional_linear_subproblems',0) for row in stats),'Total linear accounting drift.')
        for kind in ('implicit','compositional'):
            require(report[kind+'_returned_roots']==sum(r['kind']==kind for row in stats for r in row.get('returned_sources',[])),'Returned root total drift.')
        events = report['events'];indices = [e['seed_index'] for e in events]
        require(indices==sorted(set(indices)),'Completion event order drift.')
        for event in events:
            require(len(stats)==Y.shape[1],'Evaluation preceded all initializers.')
            k = event['seed_index'];require(type(k) is int and 0<=k<limit,'Event seed position drift.')
            root_sources = [row.get('returned_sources',[])[k] if k<len(row.get('returned_sources',[])) else dict(kind='parent') for row in stats]
            require(root_sources==event['root_sources'],'Actual root source drift.')
            implicit = [j for j,r in enumerate(root_sources) if r['kind']=='implicit']
            comp = [j for j,r in enumerate(root_sources) if r['kind']=='compositional']
            require((implicit or comp) and event['implicit_outputs']==implicit and event['compositional_outputs']==comp,'Event lacks actual supplement roots.')
            b = Builder()
            for j,source in enumerate(root_sources):
                if source['kind']=='parent':
                    graph = System.from_dict(parent['system'])
                    b.roots.append(clone_subgraph(b,graph,parent['constants'],graph.roots[j]))
                else:
                    b.roots.append(b.tree(Node.from_dict(source['tree']),source['constants']))
            graph,warm = b.finish()
            require(graph.to_dict()==event['graph'] and np.array_equal(warm,event['warm']),'Source graph or warm constants drift.')
        if len(stats)<Y.shape[1]:
            require(not events and report['missing_roots']==0,'Partial initialization evaluated systems.')
        else:
            attempted = len(events)+report['cache_skips']+report['structure_rejections']+report['nonimplicit_skips']
            lengths = [len(row.get('returned_sources',[])) for row in stats]
            require(0<=attempted<=max(lengths,default=0),'Combination accounting drift.')
            missing = sum(sum(k>=n for n in lengths) for k in range(attempted)
                if any(k<len(row.get('returned_sources',[])) and row['returned_sources'][k]['kind'] in ('implicit','compositional') for row in stats))
            require(report['missing_roots']==missing,'Missing root accounting drift.')
        retained,valid = self.legacy.extension_evidence(model,X,Y,cfg,events,known,parent=parent)
        self.legacy.accounting(report,retained,valid,12)
        comp_events = [e for e in events if e['compositional_outputs']]
        return retained,dict(triggered=1,new_evaluations=report['new_evaluations'],valid=valid,protected=len(retained),
            compositional_linear_subproblems=report['compositional_linear_subproblems'],implicit_linear_subproblems=report['implicit_linear_subproblems'],
            linear_subproblems=report['linear_subproblems'],missing_roots=report['missing_roots'],stop_reason=report['stop_reason'],
            compositional_returned_roots=sum(r['kind']=='compositional' for row in stats for r in row.get('returned_sources',[])),
            implicit_returned_roots=sum(r['kind']=='implicit' for row in stats for r in row.get('returned_sources',[])),
            compositional_event_roots=sum(len(e['compositional_outputs']) for e in events),implicit_event_roots=sum(len(e['implicit_outputs']) for e in events),
            compositional_new_systems=sum(e['new_evaluation'] for e in comp_events),compositional_valid_systems=sum(e['child'] is not None for e in comp_events),
            compositional_protected_systems=sum(e['protected'] for e in comp_events))

    @staticmethod
    def outer_source(tree,source):
        token,form,amplitude = source['transform'],source['form'],source['outer_amplitude']
        def variable(t):
            return t.token==source['variable'] and not t.children
        def scaled(t):
            return t.token=='mul' and len(t.children)==2 and any(c.token.startswith('c') for c in t.children) and any(variable(c) for c in t.children)
        if form=='direct':
            require(source['variable'] is None,'Direct source has variable.')
            ok = tree.token==token if not amplitude else tree.token=='mul' and any(c.token==token for c in tree.children) and any(c.token.startswith('c') for c in tree.children)
        elif form=='multiply':
            ok = tree.token=='mul' and any(c.token==token for c in tree.children) and any((scaled(c) if amplitude else variable(c)) for c in tree.children)
        else:
            ok = tree.token=='div' and tree.children[1].token==token and (scaled(tree.children[0]) if amplitude else variable(tree.children[0]))
        require(ok,'Root does not match declared outer transform.')

    def closure_evidence(self,job,frozen):
        s = self.study;folder = self.here/'runs'/s.name(job)
        model = read(folder/'model.json');cfg = SearchConfig(**model['config'])
        expected = harness.effective_config(s,job,frozen['config'])
        require(digest(model['config'])==digest(asdict(expected))==digest(read(folder/'result.json')['effective_config']),'Effective configuration drift.')
        resources = read(folder/'resources.json')
        require(0<=resources['wall_seconds']<=job['seconds']+.5 and resources['returncode']==0,'Resource bound failure.')
        require(0<=model['diagnostics']['evaluations']<=cfg.max_evaluations,'Evaluation cap exceeded.')
        record = model['diagnostics']['sharing_closure'];events = record['events']
        n = record['base_candidate_count'];payload = self.legacy.training_payload(model)[:n]
        require(0<n<=len(model['candidates']) and digest(payload)==record['base_candidate_sha256'],'Base pool drift.')
        X,Y = s.cases.training_data(job['case'],'train',job['seed']);V,Z = s.cases.training_data(job['case'],'validation',job['seed'])
        known = {digest(c) for c in payload}
        if job['method']=='base':
            require(record.get('compositional_completion') is None,'Control used completion.')
            supplements,info = self.legacy.repair_evidence(model,payload,X,Y,cfg,known)
            kind = 'precision_repair'
        else:
            supplements,info = self.completion_evidence(model,payload,X,Y,cfg,known)
            kind = 'compositional_completion'
        require(0<=record['rounds']<=2 and all(e['round'] in range(record['rounds']) for e in events),'Closure round drift.')
        require(all(sum(e['new_evaluation'] for e in events if e['round']==r)<=24 for r in range(2)),'Closure round cap exceeded.')
        all_events = record.get(kind,{}).get('events',[])+events
        hashes = [digest(e['graph']) for e in all_events]
        require(len(hashes)==len(set(hashes)),'Stages reused a cached graph.')
        require(not set(hashes)&{digest(c['system']) for c in payload},'Supplement evaluated a known foundation graph.')
        for event in events:
            require(System.from_dict(event['graph']).size<System.from_dict(event['parent']['system']).size,'Closure did not reduce structure.')
        retained,valid = self.legacy.extension_evidence(model,X,Y,cfg,events,known)
        self.legacy.accounting(record,retained,valid,48)
        require(record['base_evaluations']+info.get('new_evaluations',0)+record['new_evaluations']==model['diagnostics']['evaluations'],'Evaluation total drift.')
        require(digest(payload+supplements+retained)==record['full_training_pool_sha256'],'Full training pool drift.')
        acceptable,rejected,types = [],[],[]
        for label,children in (('supplement',supplements),('closure',retained)):
            for child in children:
                pred = raw(child['system'],child['constants'],V)
                with np.errstate(all='ignore'):errors = np.sqrt(np.mean(((pred-Z)/model['scales'])**2,axis=0))
                ok = np.isfinite(pred).all() and np.isfinite(errors).all()
                if cfg.output_bounds is not None:
                    lo,hi = cfg.output_bounds;ok = ok and (lo is None or np.all(pred>=lo)) and (hi is None or np.all(pred<=hi))
                (acceptable if ok else rejected).append(child)
                if ok:types.append(label)
        require(self.legacy.training_payload(model)[n:]==acceptable,'Validation extension subset drift.')
        require(record['validation_retained_extensions']==len(acceptable),'Validation retained count drift.')
        require([r['graph_sha256'] for r in record.get('validation_rejections',[])]==[digest(c['system']) for c in rejected],'Validation rejection drift.')
        post = model['post_selection'];i = post['base_candidate_index']
        require(0<=i<n and digest(payload[i])==post['base_choice_sha256'],'Original base choice drift.')
        require(all(a<=b for a,b in zip(model['best']['errors'],post['base_training_bounds'])) and
            all(a<=b for a,b in zip(model['validation_nrmse'],post['base_validation_bounds'])),'Final base protection failed.')
        history = next(r for r in model['history'] if r['phase']=='sharing_closure')
        runtime = next(json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['phase']=='sharing_closure')
        require(all(runtime[k]==v for k,v in history.items()),'Closure runtime event drift.')
        require(history['new_evaluations']==record['new_evaluations']+info.get('new_evaluations',0) and
            history['new_systems']==record['new_evaluations'] and history['protected_systems']==record['protected'],'Closure history accounting drift.')
        prefix = 'repair' if job['method']=='base' else 'completion'
        require(history.get(prefix+'_new_systems',0)==info.get('new_evaluations',0) and
            history.get(prefix+'_protected_systems',0)==info.get('protected',0),'Supplement history accounting drift.')
        info['validation_retained'] = types.count('supplement')
        if job['method']=='completion':
            retained_hashes = {digest(c) for c in acceptable}
            info['compositional_validation_retained'] = sum(bool(e['compositional_outputs']) and e['protected'] and digest(e['child']) in retained_hashes for e in record[kind]['events'])
        return dict(new_evaluations=record['new_evaluations'],valid=valid,protected=len(retained),validation_retained=types.count('closure')) ,info

    def control_identity(self,job,score):
        s = self.study;folder = self.here/'runs'/s.name(job)
        old_job = dict(job,method='repair');prior = s.PRIOR/'runs'/s.name(old_job)
        for filename in ('model.json','result.json','delivery.json','pool.json'):
            require(normalized(read(folder/filename))==normalized(read(prior/filename)),'Archived control identity drift: '+filename)
        archive = read(s.PRIOR/'SCORES.json')['rows']
        require(len(archive)==len({key(r) for r in archive})==s.EXPECTED and all(complete(r) for r in archive),'Archived score matrix drift.')
        old_scores = [r for r in archive if key(r)==key(dict(job=old_job))]
        require(len(old_scores)==1 and complete(old_scores[0]),'Archived score denominator drift.')
        skip = {'job','resources'}
        require(normalized({k:v for k,v in score.items() if k not in skip})==normalized({k:v for k,v in old_scores[0].items() if k not in skip}),'Archived control score drift.')
        model = read(folder/'model.json');record = model['diagnostics']['sharing_closure']
        if not record['precision_repair']['triggered']:
            original = read(s.PRIOR/'runs'/s.name(dict(job,method='base'))/'model.json')
            identity = self.legacy.paired_identity(model,original)
            require(identity['base_prefix_matches'] and identity['base_cache_matches'] and identity['skipped_pool_matches'] and identity['skipped_choice_matches'],'Control repair skip identity drift.')
        return True

    def summarize(self,scores,verified):
        s = self.study;jobs = s.matrix();keys = [key(dict(job=j)) for j in jobs]
        def index(rows):
            try:return {key(r):r for r in rows},len(rows)
            except (KeyError,TypeError):return {},-1
        ix,n = index(scores);vx,vn = index(verified)
        denominator = n==vn==len(ix)==len(vx)==s.EXPECTED and set(ix)==set(vx)==set(keys)
        planned_verified = [vx[k] for k in keys if k in vx]
        def aggregate(expected):
            rows = [ix[k] for k in expected if complete(ix.get(k,{}))]
            error = float(np.mean([r['clean_nrmse'] for r in rows])) if rows and len(rows)==len(expected) else None
            size = float(np.mean([r['structure']['shared_graph_size'] for r in rows])) if rows and len(rows)==len(expected) else None
            return dict(expected=len(expected),complete=len(rows),failed_or_missing=len(expected)-len(rows),both_accepted=sum(r['both_accepted'] for r in rows),
                mean_clean_nrmse=error if error is not None and np.isfinite(error) else None,mean_shared_graph_size=size if size is not None and np.isfinite(size) else None)
        methods = {m:aggregate([k for k in keys if k[2]==m]) for m in s.METHODS}
        pairs = []
        for case in s.cases.CASES:
            for seed in s.cases.seeds(case):
                a,b = ix.get((case,seed,'completion'),{}),ix.get((case,seed,'base'),{})
                p = dict(case=case,seed=seed,available=complete(a) and complete(b))
                if p['available']:
                    p.update(protected=(np.asarray(a['clean_nrmse'])<=np.asarray(b['clean_nrmse'])*1.1+1e-8).tolist(),candidate_accepted=a['both_accepted'],control_accepted=b['both_accepted'])
                    try:
                        models = [read(self.here/'runs'/s.name(row['job'])/'model.json') for row in (a,b)]
                        records = [m['diagnostics']['sharing_closure'] for m in models]
                        payloads = [self.legacy.training_payload(m)[:r['base_candidate_count']] for m,r in zip(models,records)]
                        p['base_identity'] = payloads[0]==payloads[1] and all(records[0][k]==records[1][k] for k in ('base_candidate_count','base_candidate_sha256','base_evaluations','base_search_fingerprint'))
                    except Exception as e:p['identity_error']=repr(e)
                pairs.append(p)
        def count(value):
            return value if type(value) is int and value>=0 else 0
        by_method = {m:{k:sum(count(r.get('closure',{}).get(k,0)) for r in planned_verified if r['job']['method']==m) for k in COUNTS} for m in s.METHODS}
        matching = all(v>0 for row in by_method.values() for v in row.values())
        foundation = len(pairs)==s.EXPECTED//2 and all(p['available'] and p.get('base_identity',False) for p in pairs)
        control = all(vx.get(k,{}).get('archived_control_identity',False) for k in keys if k[2]=='base')
        counters_valid = all(all(type(r.get('closure',{}).get(k)) is int and r['closure'][k]>=0 for k in COUNTS)
            and all(type(v) is int and v>=0 for k,v in r.get('completion',{}).items() if k in COMPLETION_COUNTS+('compositional_validation_retained','missing_roots'))
            for r in planned_verified if r.get('verified',False))
        technical = denominator and all(vx[k].get('verified',False) and complete(ix[k]) for k in keys) and counters_valid and matching and foundation and control
        protected = sum(sum(p.get('protected',[])) for p in pairs)
        preserved = all(p['available'] and (not p['control_accepted'] or p['candidate_accepted']) for p in pairs)
        a,b = methods['completion'],methods['base']
        macro = a['mean_clean_nrmse'] is not None and b['mean_clean_nrmse'] is not None and a['mean_clean_nrmse']<=b['mean_clean_nrmse']
        graph = a['mean_shared_graph_size'] is not None and b['mean_shared_graph_size'] is not None and a['mean_shared_graph_size']<=b['mean_shared_graph_size']
        candidate_all = a['complete']==s.EXPECTED//2 and a['both_accepted']==s.EXPECTED//2
        targets = {}
        if s.STAGE=='historical':
            for case,seed,threshold,maxgraph in (('damped_n0',701,1e-12,20),('rational_n0',702,1e-4,15)):
                row = ix.get((case,seed,'completion'),{})
                ok = False
                if complete(row):
                    try:
                        model = read(self.here/'runs'/s.name(row['job'])/'model.json')
                        errors = np.asarray(model['best']['errors']+model['validation_nrmse']+row['clean_nrmse'],float)
                        ok = bool(row['both_accepted'] and errors.shape==(6,) and np.isfinite(errors).all() and (errors>=0).all() and
                            np.all(errors<=threshold) and row['structure']['shared_graph_size']<=maxgraph and row['structure']['independent_parameters']<=4)
                    except Exception:pass
                targets[f'{case}_s{seed}'] = ok
        adoption = technical and protected==s.EXPECTED and preserved and macro and graph and (all(targets.values()) if s.STAGE=='historical' else candidate_all)
        families = sorted({s.cases.design(c)['family'] for c in s.cases.CASES})
        evidence = {}
        for family in families:
            rows = [r for r in planned_verified if r['job']['method']=='completion' and s.cases.design(r['job']['case'])['family']==family]
            layer = [p for p in pairs if s.cases.design(p['case'])['family']==family]
            evidence[family] = dict({k:sum(count(r.get('completion',{}).get(k,0)) for r in rows) for k in COMPLETION_COUNTS+('compositional_validation_retained',)},
                expected_systems=len(layer),acceptance_gains=sum(p.get('candidate_accepted',False) and not p.get('control_accepted',False) for p in layer if p['available']))
        failures = []
        for j in jobs:
            row = ix.get(key(dict(job=j)),{})
            if j['method']!='completion' or complete(row) and row['both_accepted']:
                continue
            failed = dict(job=j,status=row.get('status') if isinstance(row.get('status'),str) else 'missing_or_invalid',both_accepted=row.get('both_accepted') is True)
            if complete(row):
                failed.update(clean_nrmse=row['clean_nrmse'],observed_nrmse=row['observed_nrmse'],structure=row['structure'])
                try:
                    model = read(self.here/'runs'/s.name(j)/'model.json');spec=s.cases.design(j['case'])
                    reasons = [k+' exceeds '+str(limit) for k,limit in spec['acceptance'].items() if k!='nrmse' and row['structure'][k]>limit]
                    for label,errors,limits in (('train',model['best']['errors'],spec['acceptance']['nrmse']),('validation',model['validation_nrmse'],spec['acceptance']['nrmse']),
                        ('observed_test',row['observed_nrmse'],spec['acceptance']['nrmse']),('clean_test',row['clean_nrmse'],spec['clean_test_nrmse'])):
                        reasons.extend(label+' output'+str(i)+' exceeds '+str(limit) for i,(e,limit) in enumerate(zip(errors,limits)) if not np.isfinite(e) or e<0 or e>limit)
                    failed['failure_reasons']=reasons or ['Other development/relative acceptance constraint not met; see delivery and post_selection.']
                except Exception as error:failed['failure_detail_error']=repr(error)
            failures.append(failed)
        return dict(expected_systems=s.EXPECTED,expected_pairs=s.EXPECTED//2,expected_outputs=s.EXPECTED,full_denominator=denominator,methods=methods,pairs=pairs,
            technical_pass=bool(technical),adoption_pass=bool(adoption),protected_outputs=protected,paired_base_identity_pass=foundation,
            archived_control_identity_pass=control,closure_by_method=by_method,closure_mechanism_pass=matching,control_acceptance_preserved=preserved,
            macro_clean_error_nonincrease=bool(macro),mean_graph_nonincrease=bool(graph),candidate_all_accepted=candidate_all,historical_targets=targets,
            completion_failures=failures,completion_discovery_by_family=evidence,production_default_changed=False,
            claim_scope='Observed development/regression batches. Always supplementation and new starts are compared jointly; provenance does not identify a single-component causal effect.')

    def verify(self):
        s = self.study
        require(not (self.here/'VERIFICATION.json').exists(),'Keep verification evidence.')
        jobs = s.matrix();frozen={};scores=[];deadline=None;preparation_error=None
        try:
            deadline=s.budget();frozen,sealed=self.guards()
            require(not (self.here/'SCORING_FAILURE.json').exists(),'Scoring phase failed.')
            table=read(self.here/'SCORES.json')
            require(table['all_models_sha256']==sha(self.here/'ALL_MODELS_FROZEN.json') and sealed['frozen_unix']<=table['scored_unix'],'Score chain drift.')
            require(isinstance(table['rows'],list),'Invalid score rows.')
            counts=Counter(key(r) for r in table['rows']);index={key(r):r for r in table['rows']};scores=table['rows']
        except Exception as e:preparation_error=repr(e)
        rows=[]
        for job in jobs:
            try:
                require(preparation_error is None,'Verification prerequisites failed: '+str(preparation_error))
                require(time.time()<deadline,'Verification budget expired.')
                k=key(dict(job=job));score=index.get(k,{})
                require(counts[k]==1 and score.get('job')==job and complete(score),'Invalid or incomplete cell.')
                row=self.base.verify_cell(job,score,frozen)
                row['closure'],supplement=self.closure_evidence(job,frozen)
                row['repair' if job['method']=='base' else 'completion']=supplement
                if job['method']=='base':row['archived_control_identity']=self.control_identity(job,score)
                require(time.time()<deadline,'Verification exceeded deadline.')
            except Exception as e:row=dict(job=job,verified=False,error=repr(e))
            rows.append(row);print(s.name(job),row.get('error','verified'),flush=True)
        summary=self.summarize(scores,rows);within=deadline is not None and time.time()<deadline
        if not within:summary.update(technical_pass=False,adoption_pass=False)
        def receipt_hash(n):
            try:return sha(self.here/n)
            except OSError:return None
        start=frozen.get('budget',{}).get('start_unix')
        write(self.here/'VERIFICATION.json',dict(status='passed' if summary['technical_pass'] else 'failed',expected_systems=s.EXPECTED,
            verified_systems=sum(r['verified'] for r in rows),rows=rows,preparation_error=preparation_error,budget_met=within,verified_unix=time.time(),
            elapsed_seconds=None if start is None else time.time()-start,freeze_sha256=receipt_hash('FREEZE.json'),
            models_sha256=receipt_hash('ALL_MODELS_FROZEN.json'),scores_sha256=receipt_hash('SCORES.json')))
        write(self.here/'SUMMARY.json',summary)

    def run(self):
        require(not any((self.here/n).exists() for n in ('SCORES.json','SCORING_FAILURE.json','VERIFICATION.json')),'Keep scoring evidence.')
        try:self.evaluate()
        except Exception as error:
            write(self.here/'SCORING_FAILURE.json',dict(status='failed',phase='scoring',error=repr(error),failed_unix=time.time()))
        self.verify()
