"""Rebuild external metrics, fixed advancement gates and paired block intervals."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np
from common import HERE,DATASETS,METHODS,SYMBOLIC,sha,save,jobs,check_freeze

def aggregate(rows,expected):
    complete=len(rows)==expected and all(r['status']=='ok' for r in rows)
    out=dict(complete=complete,successful=sum(r['status']=='ok' for r in rows),expected=expected)
    if complete:
        for key,values in (('nrmse',[r['test']['mean_nrmse'] for r in rows]),
            ('training_cpu_seconds',[r['training_cpu_seconds'] for r in rows]),
            ('complexity',[r['complexity'] for r in rows])):
            if all(v is not None for v in values):out[key]=dict(mean=statistics.mean(values),sd=statistics.pstdev(values))
    return out

def gates(groups,rows):
    primary,control,base=(groups[m]['all'] for m in ('semantic25','independent25','independent20'))
    result=dict(eligible=False,complete=all(g['complete'] for g in (primary,control,base)))
    if not result['complete']:return result
    ce,be,ck,bk=primary['nrmse']['mean'],control['nrmse']['mean'],primary['complexity']['mean'],control['complexity']['mean']
    le=lambda a,b:a<=b+1e-12*max(1.,abs(b))
    guards=[]
    for d in DATASETS:
        selected=[[r for r in rows if r['dataset']==d and r['method']==m] for m in ('semantic25','independent25')]
        means=[np.mean([r['test']['nrmse'] for r in part],axis=0) for part in selected]
        for j,(new,old) in enumerate(zip(*means)):
            limit=max(.01,.02*old);guards.append(dict(dataset=d,output=j,control=float(old),candidate=float(new),allowed_increase=float(limit),passed=bool(le(new-old,limit))))
    accuracy=le(ce,.99*be) and le(ck,1.1*bk);compression=le(ck,.9*bk) and le(ce,1.005*be)
    no_worse=le(ce,base['nrmse']['mean'])
    result.update(accuracy_route=accuracy,compression_route=compression,no_worse_base=no_worse,
        output_guards=guards,relative_error_change_pct=100*(ce/be-1),relative_complexity_change_pct=100*(ck/bk-1),
        eligible=bool((accuracy or compression) and no_worse and all(g['passed'] for g in guards)))
    return result

def paired_bootstrap(left,right,counts,seed=20261026,replicates=1000):
    """Whole-group resampling, retaining all fixed models and output pairing."""
    weights=np.random.default_rng(seed).multinomial(len(counts),np.full(len(counts),1/len(counts)),size=replicates)
    denominator=weights@counts
    def score(sums):return np.sqrt(np.einsum('bg,sgo->bso',weights,sums)/denominator[:,None,None]).mean(axis=(1,2))
    delta=score(left)-score(right)
    point=float(np.sqrt(left.sum(axis=1)/sum(counts)).mean()-np.sqrt(right.sum(axis=1)/sum(counts)).mean())
    return dict(difference=point,lower=float(np.quantile(delta,.025)),upper=float(np.quantile(delta,.975)),
        groups=len(counts),replicates=replicates,scope='Conditional on fixed models and observed task; paired whole-group resampling')

def analyze(out):
    check_freeze();path=HERE/'evaluation/results.json';evaluation=json.loads(path.read_text());rows=evaluation['rows']
    assert evaluation['models_freeze_sha256']==sha(HERE/'MODELS_FROZEN.json')
    expected={(d,s,m) for d,s in jobs() for m in METHODS}
    assert len(rows)==len(expected) and {(r['dataset'],r['seed'],r['method']) for r in rows}==expected
    groups={m:{d:aggregate([r for r in rows if r['method']==m and (d=='all' or r['dataset']==d)],15 if d=='all' else 5)
        for d in (*DATASETS,'all')} for m in METHODS}
    intervals={}
    for d in DATASETS:
        file=HERE/'evaluation'/f'{d}_group_errors.npz'
        assert sha(file)==evaluation['group_error_sha256'][file.name]
        if all(groups[m][d]['complete'] for m in ('semantic25','independent25')):
            with np.load(file,allow_pickle=False) as errors:
                left,right=(np.stack([errors[f'{m}_s{s}'] for s in range(5)]) for m in ('semantic25','independent25'))
                intervals[d]=paired_bootstrap(left,right,errors['counts'])
                assert abs(intervals[d]['difference']-(groups['semantic25'][d]['nrmse']['mean']-groups['independent25'][d]['nrmse']['mean']))<1e-12
    summary=dict(evaluations=len(rows),successful=sum(r['status']=='ok' for r in rows),evaluation_sha256=sha(path),
        models_freeze_sha256=evaluation['models_freeze_sha256'],methods=groups,gate=gates(groups,rows),
        conditional_primary_intervals=intervals,datasets=evaluation['datasets'])
    flat=[]
    for r in rows:
        for j,target in enumerate(evaluation['datasets'][r['dataset']]['target_names']):
            row={k:r[k] for k in ('dataset','seed','method','status','complexity','training_cpu_seconds','selected_transfer_heads')}
            row.update(output=target,validation_nrmse=r['validation']['nrmse'][j])
            if r['status']=='ok':row.update({k:r['test'][k][j] for k in ('rmse','nrmse','r2')})
            row['expanded_domain']=r.get('expanded_domain',{}).get('status');flat.append(row)
    out.mkdir(exist_ok=False)
    with (out/'scores.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in flat for k in r)));writer.writeheader();writer.writerows(flat)
    lines=['# Frozen external evaluation','','Lower NRMSE is better. K applies only to symbolic models.','',
        '| Dataset | Method | Test NRMSE | K | Training CPU |','| --- | --- | ---: | ---: | ---: |']
    for d in (*DATASETS,'all'):
        for m in METHODS:
            g=groups[m][d]
            values=f"{g['nrmse']['mean']:.6f} | {g['complexity']['mean']:.2f}" if g['complete'] and m in SYMBOLIC else f"{g['nrmse']['mean']:.6f} | n/a" if g['complete'] else 'failed | n/a'
            cpu=f"{g['training_cpu_seconds']['mean']:.3f}" if g['complete'] else 'n/a'
            lines.append(f'| {d} | {m} | {values} | {cpu} |')
    (out/'TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(out/'summary.json',summary);print(json.dumps(summary['gate'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'analysis');analyze(p.parse_args().output)
