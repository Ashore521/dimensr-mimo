"""Independent sealed-lineage checks; no initializer/linear solve/search calls."""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CALLS=[('logarithmic_gain_n0',0),('logarithmic_gain_n0',1),('algebraic_norm_n0',0),('algebraic_norm_n0',1)]
OUT=HERE/'post_review_checks.json'
failures=[]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,label):
    if not ok:failures.append(label)


def hashes(files,root):
    return [n for n,h in files.items() if not (root/n).is_file() or sha(root/n)!=h]


@dataclass(frozen=True)
class Node:
    token:str
    children:tuple
    units:tuple


def node(t):return Node(t['token'],tuple(node(c) for c in t['children']),tuple(Fraction(v) for v in t['units']))


def raw(t,constants,X):
    count=lambda n:sum(count(c) for c in n['children']) if n['children'] else int(n['token'].startswith('c'))
    require(len(constants)==count(t),'raw_constant_count')
    it=iter(constants);bad=np.zeros(len(X),bool)
    functions=dict(add=np.add,sub=np.subtract,mul=np.multiply,div=np.divide,square=np.square,sqrt=np.sqrt,sin=np.sin,cos=np.cos,exp=np.exp,log=np.log,neg=np.negative)
    def visit(n):
        nonlocal bad
        if n['children']:v=functions[n['token']](*(visit(c) for c in n['children']))
        else:v=X[:,int(n['token'][1:])] if n['token'].startswith('x') else next(it)
        bad|=~np.isfinite(v)
        return v
    with np.errstate(all='ignore'):result=np.broadcast_to(visit(t),(len(X),)).astype(float,copy=True)
    sentinel=object();require(next(it,sentinel) is sentinel,'raw_constant_consumption')
    result[bad]=np.nan
    return result


def walk(t):
    yield t
    for c in t['children']:yield from walk(c)


def shape(t):
    children=[shape(c) for c in t['children']]
    return 1+sum(s for s,d in children),1+max((d for s,d in children),default=0)


def flags(t):
    ns=list(walk(t))
    log=lambda n:n['token']=='log' and any(c['token']=='square' for c in walk(n))
    norm=lambda n:n['token']=='sqrt' and n['children'][0]['token']=='add' and all(any(c['token']=='square' for c in walk(branch)) for branch in n['children'][0]['children'])
    mul=lambda n:n['token']=='mul' and any(a['token'].startswith('x') and log(b) for a,b in (n['children'],n['children'][::-1]))
    return {k:any(fn(n) for n in ns) for k,fn in (('log_square',log),('sqrt_sum_squares',norm),('variable_times_log_square',mul))}


def train(case):
    # The only arrays reconstructed are the two locked no-noise training splits.
    i=0 if case=='logarithmic_gain_n0' else 4
    rng=np.random.default_rng(np.random.SeedSequence([20261030,9131,111,i]))
    X=rng.uniform([.2,.2],[1.4,1.4],(96,2));a,b=X.T
    if i==0:z=np.log(1+(.9*a)**2);Y=np.column_stack((z,b*z))
    else:z=np.sqrt(a*a+b*b);Y=np.column_stack((z,a/z))
    return X,Y


def data_digest(*arrays):
    h=hashlib.sha256()
    for x in arrays:
        a=np.asarray(x,dtype='<f8',order='C');h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def main():
    if OUT.exists():raise RuntimeError('Do not overwrite review checkpoint.')
    f=read(HERE/'FREEZE.json');v=read(HERE/'VERIFICATION.json');e=read(HERE/'EXECUTION.json');a=read(HERE/'FROZEN_INPUT_ARCHIVE.json');summary=read(HERE/'LINEAGE_SUMMARY.json');b=read(HERE/'BUDGET.json')
    live=hashes(f['files'],ROOT);arch=hashes(f['files'],HERE/'frozen_inputs');outputs=hashes(v['outputs'],HERE)
    require(len(f['files'])==len(a['files'])==a['file_count']==55 and f['files']==a['files'],'55_input_manifest');require(not live and not arch and not outputs,'input_output_hashes')
    require(a['freeze_sha256']==v['freeze_sha256']==sha(HERE/'FREEZE.json'),'freeze_chain');require(summary['verification_sha256']==sha(HERE/'VERIFICATION.json'),'summary_verification_chain');require(summary['script_sha256']==sha(HERE/'summarize.py'),'summary_script_chain')
    require(set(v['outputs'])=={p.name for p in HERE.glob('output_*.*')} and len(v['outputs'])==16,'16_output_manifest')
    require(f['calls']==[list(c) for c in CALLS] and f['expected_calls']==4 and f['limit']==12 and f['compound'] and not f['implicit'],'fixed_calls')
    require(f['seed']==9131 and f['nonlinear_fits']==0 and not f['validation_labels'] and not f['test_labels'],'scope_flags')
    require(b['limit_seconds']==60 and b['initializer_calls']==4 and b['numerical_threads']==1 and e['wall_seconds']<60,'budget')
    require(e['returncode']==0 and e['termination_reason'] is None and not e['verification_passed'],'execution_failed_verification')
    require(v['status']=='incomplete' and v['input_integrity_error'] is None and v['completed_calls']==4 and v['verified_calls']==2,'verification_terminal')
    rows=[];aux=[];total_events=Counter();max_score=0.;data_checks={}
    source=ROOT/'experiments/new_task_validation_20261030'
    prior_seal=read(source/'ALL_MODELS_FROZEN.json');prior_freeze=read(source/'FREEZE.json')
    require(prior_seal['freeze_sha256']==sha(source/'FREEZE.json'),'historical_chain')
    previous_start=b['start_unix']
    for (case,j),verified,execution in zip(CALLS,v['rows'],e['calls']):
        tag=f'{case}_o{j}';r=read(HERE/f'output_{tag}.json');saved=read(HERE/f'output_{tag}_raw.json');started=read(HERE/f'output_{tag}_STARTED.json')
        require((verified['case'],verified['output'])==(case,j) and verified==execution,'call_identity_'+tag)
        require(previous_start<=started['started_unix']<=b['start_unix']+e['wall_seconds'],'start_chronology_'+tag);previous_start=started['started_unix']
        require(r['raw_sha256']==sha(HERE/f'output_{tag}_raw.json') and r['trace_sha256']==sha(HERE/f'output_{tag}_trace.jsonl'),'call_chain_'+tag)
        path=source/'runs'/f'{case}_s9131_base/model.json';m=read(path);require(sha(path)==prior_seal['files'][path.relative_to(source).as_posix()],'historical_model_'+tag)
        require(saved['counts']==r['counts']==m['diagnostics']['initialization_diagnostics'][j],'historical_counts_'+tag)
        X,Y=train(case);dh=data_digest(X,Y);require(dh==f['training_sha256'][case]==m['diagnostics']['data_sha256']['training'],'training_hash_'+tag);data_checks[case]=dh
        require([list(X.shape),list(Y.shape)]==f['training_shapes'][case],'training_shape_'+tag)
        feature_bad=[]
        for z in saved['features']:
            pred=raw(z['tree'],z['constants'],X)
            if hashlib.sha256(pred.tobytes()).hexdigest()!=z['vector_sha256']:feature_bad.append(z['tree'])
        require(not feature_bad,'feature_vector_hashes_'+tag)
        proposals=r['proposals'];require(len(proposals)==len(saved['proposals'])==saved['counts']['proposed_structures'],'proposal_count_'+tag)
        original_keys=[];independent_keys=[];score_diff=0.;saved_recomputed_diff=0.
        for i,(p,q) in enumerate(zip(proposals,saved['proposals'])):
            require(p['tree']==q['tree'] and p['constants']==q['constants'] and p['score']==q['score'] and p['rank']==i,'proposal_snapshot_'+tag)
            size,depth=shape(p['tree']);require(size<=15 and depth<=6,'legal_proposal_structure_'+tag)
            predicted=raw(p['tree'],p['constants'],X);mse=float(np.mean((predicted-Y[:,j])**2));score=mse/max(float(np.var(Y[:,j])),1e-12)+1e-8*size
            require(np.isfinite(score) and np.isclose(score,p['score'],rtol=1e-10,atol=1e-12),'score_tolerance_'+tag)
            require(score==p['recomputed_score'],'saved_recomputed_score_'+tag)
            require(np.isclose(np.sqrt(mse)/Y[:,j].std(),p['training_nrmse'],rtol=1e-12,atol=1e-14),'training_nrmse_'+tag)
            score_diff=max(score_diff,abs(score-p['score']));saved_recomputed_diff=max(saved_recomputed_diff,abs(score-p['recomputed_score']))
            original_keys.append((p['score'],size,repr(node(p['tree'])),i));independent_keys.append((score,size,repr(node(p['tree'])),i))
        require([x[3] for x in sorted(original_keys)]==list(range(len(proposals))),'original_ranking_'+tag)
        independent=[x[3] for x in sorted(independent_keys)];order_ok=independent==list(range(len(proposals)));top_ok=independent[:12]==list(range(12))
        require(independent==r['independent_order_as_original_indices'] and order_ok==r['independent_full_order_match'] and top_ok==r['independent_seed_order_match'],'independent_ranking_'+tag)
        require([dict(tree=p['tree'],constants=p['constants']) for p in proposals[:12]]==saved['returned_seeds'],'original_returned_seeds_'+tag)
        require(verified['verified']==order_ok and verified['initializer_returned'] and verified['status']==('passed' if order_ok else 'ranking_unresolved'),'ranking_status_'+tag)
        facets={k:{stage:0 for stage in ('features','proposals','returned','boundary_attempts','boundary_rejected')} for k in ('log_square','sqrt_sum_squares','variable_times_log_square')}
        for stage,items in [('features',saved['features']),('proposals',saved['proposals']),('returned',saved['returned_seeds'])]:
            for item in items:
                for k,found in flags(item['tree']).items():facets[k][stage]+=int(found)
        boundary=dict(attempted=0,structure_rejected=0,nonfinite_error=0);direct=[];kinds=Counter();admit_return_lines=Counter();admitted=0;denominators=[]
        with (HERE/f'output_{tag}_trace.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                event=json.loads(line);kind=event['kind'];record=event['record'];kinds[kind]+=1
                if kind=='admit_return':admitted+=record['admitted'];admit_return_lines[record['return_line']]+=1
                if kind=='denominator_call':denominators.append(record)
                if kind!='proposal_boundary':continue
                size,depth=shape(record['tree']);require((size,depth)==(record['size'],record['depth']),'boundary_shape_'+tag)
                rejected=size>15 or depth>6;require(rejected==record['structure_rejected'],'boundary_rejection_'+tag)
                require(record['source_line'] in (198,308),'boundary_anchor_'+tag)
                boundary['attempted']+=1;boundary['structure_rejected']+=int(rejected);boundary['nonfinite_error']+=int(record['finite_error'] is False)
                for k,found in flags(record['tree']).items():facets[k]['boundary_attempts']+=int(found);facets[k]['boundary_rejected']+=int(found and rejected)
                if record['source_line']==198 and rejected and flags(record['tree'])['log_square']:direct.append(record)
        require(boundary==saved['boundary_counts']==r['boundary_counts'],'boundary_counts_'+tag)
        require(admitted==len(saved['features'])==saved['counts']['retained_features'],'feature_admissions_'+tag)
        require(denominators==saved['denominator_calls'],'denominator_trace_'+tag)
        root_features=[dict(tree=z['tree'],constants=z['constants'],size=shape(z['tree'])[0],depth=shape(z['tree'])[1]) for z in saved['features'] if z['tree']['token']=='log' and flags(z['tree'])['log_square']]
        mismatches=[dict(position=i,independent_original_index=k,original_score=proposals[i]['score'],original_other_score=proposals[k]['score'],recomputed_score=proposals[i]['recomputed_score'],recomputed_other_score=proposals[k]['recomputed_score']) for i,k in enumerate(independent) if i!=k]
        row=dict(case=case,output=j,observation_status=verified['status'],counts=saved['counts'],boundary_counts=boundary,facets=facets,root_log_square_features=root_features,direct_log_boundary_rejections=direct,rank_mismatches=mismatches,top12_match=top_ok,maximum_score_difference=score_diff)
        require(row==summary['rows'][len(rows)],'summary_row_'+tag);rows.append(row)
        aux.append(dict(case=case,output=j,trace_event_kinds=dict(kinds),admit_return_lines=dict(admit_return_lines),feature_vector_hash_mismatches=len(feature_bad),proposal_scores_recomputed=len(proposals),maximum_difference_from_saved_independent_score=saved_recomputed_diff))
        total_events.update(kinds);max_score=max(max_score,score_diff)
        print('reviewed',tag,len(proposals),'proposals',len(mismatches),'rank differences',flush=True)
    reconstructed=dict(status='observed_with_rank_limitations',expected_calls=4,completed_calls=4,full_order_verified=sum(r['observation_status']=='passed' for r in rows),top12_verified=sum(r['top12_match'] for r in rows),rows=rows,scope='Syntactic facets of the four saved initializer calls; not symbolic equivalence or whole-search coverage.',verification_sha256=sha(HERE/'VERIFICATION.json'),script_sha256=sha(HERE/'summarize.py'))
    require(reconstructed==summary,'full_lineage_summary')
    require(len(v['rows'])==len(e['calls'])==4,'fixed_denominator')
    final_live=hashes(f['files'],ROOT);final_archive=hashes(f['files'],HERE/'frozen_inputs');final_outputs=hashes(v['outputs'],HERE)
    require(not final_live and not final_archive and not final_outputs,'final_immutable_bytes')
    result=dict(status='passed' if not failures else 'findings',findings=failures,scope='Independent two train reconstructions, strict raw evaluation of all 1849 saved proposals and retained feature vectors, saved score/rank comparison, all proposal-boundary and admit/denominator trace records; no initializer/lstsq/nonlinear fit/search/validation/test arrays.',versions=dict(python=sys.version,numpy=np.__version__),key_sha256={n:sha(HERE/n) for n in ('FREEZE.json','VERIFICATION.json','EXECUTION.json','FROZEN_INPUT_ARCHIVE.json','LINEAGE_SUMMARY.json','summarize.py','PLAN.md','PRE_RUN_REVIEW.md','TEST_VALIDATION.json')},integrity=dict(input_count=55,output_count=16,live_mismatches=live,archive_mismatches=arch,output_mismatches=outputs,final_live_mismatches=final_live,final_archive_mismatches=final_archive,final_output_mismatches=final_outputs),training_hashes=data_checks,summary_reconstructed=reconstructed,details=aux,total_trace_events=dict(total_events),total_proposals=sum(r['counts']['proposed_structures'] for r in rows),total_boundaries=sum(r['boundary_counts']['attempted'] for r in rows),total_structure_rejections=sum(r['boundary_counts']['structure_rejected'] for r in rows),total_linear_subproblems=sum(r['counts']['linear_subproblems'] for r in rows),maximum_original_score_difference=max_score,script_sha256=sha(Path(__file__)))
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],findings=failures,total_proposals=result['total_proposals'],maximum_original_score_difference=max_score,output_sha256=sha(OUT))))


if __name__=='__main__':main()
