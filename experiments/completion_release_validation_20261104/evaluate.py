"""New-task verifier: same-batch foundation identity, with no archived control claim."""
from collections import Counter
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from experiments.compositional_completion_20261031.verification import (
    Verifier,COUNTS,COMPLETION_COUNTS,key,complete as saved_complete)
from experiments.native_api_20261026.run import read,write,sha
from experiments.native_api_20261026.verify import require
from experiments.completion_release_validation_20261104 import study


def complete(row):
    if not saved_complete(row):return False
    values=row['clean_nrmse']+row['observed_nrmse']+[row['mean_clean_nrmse']]
    return all(type(v) in (int,float) and np.isfinite(v) and v>=0 for v in values)


class NewTaskVerifier(Verifier):
    def control_identity(self,job,score):
        raise RuntimeError('No archived same-task control exists for this new matrix.')

    def guards(self):
        frozen,sealed=super().guards()
        preflight=read(self.here/'PREFLIGHT.json')
        require(preflight['test_labels_accessed'] is False and all(preflight[k]==v for k,v in self.study.task_binding().items()),'New-task preflight binding drift.')
        return frozen,sealed

    def summarize(self,scores,verified):
        s = self.study;jobs = s.matrix();keys = [key(dict(job=j)) for j in jobs]
        def index(rows):
            try:return {key(r):r for r in rows},len(rows)
            except (KeyError,TypeError):return {},-1
        ix,n = index(scores);vx,vn = index(verified)
        denominator = n==vn==len(ix)==len(vx)==s.EXPECTED and set(ix)==set(vx)==set(keys)
        denominator = denominator and all(ix[key(dict(job=j))]['job']==j and vx[key(dict(job=j))]['job']==j for j in jobs)
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
                        prefixes_valid=all(type(r['base_candidate_count']) is int and 0<r['base_candidate_count']<=len(m['candidates']) and
                            self.legacy.digest(payload)==r['base_candidate_sha256'] and type(r['base_evaluations']) is int and r['base_evaluations']>=0
                            for m,r,payload in zip(models,records,payloads))
                        p['base_identity'] = prefixes_valid and payloads[0]==payloads[1] and all(records[0][k]==records[1][k] for k in ('base_candidate_count','base_candidate_sha256','base_evaluations','base_search_fingerprint'))
                    except Exception as e:p['identity_error']=repr(e)
                pairs.append(p)
        def count(value):
            return value if type(value) is int and value>=0 else 0
        by_method = {m:{k:sum(count(r.get('closure',{}).get(k,0)) for r in planned_verified if r['job']['method']==m) for k in COUNTS} for m in s.METHODS}
        matching = all(v>0 for row in by_method.values() for v in row.values())
        foundation = len(pairs)==s.EXPECTED//2 and all(p['available'] and p.get('base_identity',False) for p in pairs)
        counters_valid = all(all(type(r.get('closure',{}).get(k)) is int and r['closure'][k]>=0 for k in COUNTS)
            and all(type(v) is int and v>=0 for k,v in r.get('completion',{}).items() if k in COMPLETION_COUNTS+('compositional_validation_retained','missing_roots'))
            for r in planned_verified if r.get('verified',False))
        technical = denominator and all(vx[k].get('verified',False) is True and complete(ix[k]) for k in keys) and counters_valid and matching and foundation
        protected = sum(sum(p.get('protected',[])) for p in pairs)
        preserved = all(p['available'] and (not p['control_accepted'] or p['candidate_accepted']) for p in pairs)
        a,b = methods['completion'],methods['base']
        macro = a['mean_clean_nrmse'] is not None and b['mean_clean_nrmse'] is not None and a['mean_clean_nrmse']<=b['mean_clean_nrmse']
        graph = a['mean_shared_graph_size'] is not None and b['mean_shared_graph_size'] is not None and a['mean_shared_graph_size']<=b['mean_shared_graph_size']
        candidate_all = a['complete']==s.EXPECTED//2 and a['both_accepted']==s.EXPECTED//2
        adoption = technical and protected==s.EXPECTED and preserved and macro and graph and candidate_all
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
            closure_by_method=by_method,closure_mechanism_pass=matching,control_acceptance_preserved=preserved,
            macro_clean_error_nonincrease=bool(macro),mean_graph_nonincrease=bool(graph),candidate_all_accepted=candidate_all,
            completion_failures=failures,completion_discovery_by_family=evidence,production_default_changed=False,
            historical_control_comparison='not_applicable_no_same_task_archive',
            claim_scope='Locked-candidate authored new output pairs; not an external blinded or real-data benchmark. Supplementation and new starts are compared jointly.')

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
            verified_systems=sum(r.get('verified') is True for r in rows),rows=rows,preparation_error=preparation_error,budget_met=within,verified_unix=time.time(),
            elapsed_seconds=None if start is None else time.time()-start,freeze_sha256=receipt_hash('FREEZE.json'),
            models_sha256=receipt_hash('ALL_MODELS_FROZEN.json'),scores_sha256=receipt_hash('SCORES.json')))
        write(self.here/'SUMMARY.json',summary)


verifier=NewTaskVerifier(study)
evaluate,verify,summarize,run=verifier.evaluate,verifier.verify,verifier.summarize,verifier.run
if __name__=='__main__':run()
