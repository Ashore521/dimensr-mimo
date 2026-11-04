"""Freeze validation-only derived deliveries, then compare known test predictions."""
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE.parent/'unit_composition_20261026'
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor
from experiments.precision_delivery_20261026.selector import select
from experiments.typed_coverage_20261026 import cases,delivery
from experiments.typed_coverage_20261026.worker import read,write,sha


def run():
    if (HERE/'FREEZE.json').exists():raise RuntimeError('Already started; inspect checkpoint, never overwrite.')
    verification=read(SOURCE/'VERIFICATION.json')
    if verification['verified_systems']!=48 or not verification['technical_pass']:raise RuntimeError('Source verification incomplete.')
    source_freeze=read(SOURCE/'FREEZE.json');jobs=source_freeze['jobs'];start=time.time();end=start+600
    sources={}
    for j in jobs:
        n=f"{j['case']}_s{j['seed']}_{j['method']}";p=SOURCE/'runs'/n
        for f in ('searched_model.json','model.json','delivery.json'):sources[str((p/f).relative_to(ROOT)).replace('\\','/')]=sha(p/f)
    for f in ('FREEZE.json','ALL_MODELS_FROZEN.json','VERIFICATION.json','SCORES.json'):sources[str((SOURCE/f).relative_to(ROOT)).replace('\\','/')]=sha(SOURCE/f)
    code=list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md']+list((ROOT/'src').rglob('*.py'))
    code+=[HERE.parent/n for n in ('typed_coverage_20261026/audit.py','typed_coverage_20261026/delivery.py','typed_coverage_20261026/cases.py','typed_coverage_20261026/worker.py','guarded_selection_20261026/selector.py')]
    code_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in code}
    write(HERE/'FREEZE.json',dict(start_unix=start,limit_seconds=600,jobs=jobs,specs=source_freeze['specs'],
        sources=sources,code_sha256=code_hashes,new_searches=0,
        current_test_data_used_for_selection=False,evidence_scope='Post-selection on known development tasks.'))
    def check():
        if time.time()>=end:raise RuntimeError('Delivery budget expired.')
        for n,h in {**sources,**code_hashes}.items():
            if sha(ROOT/n)!=h:raise RuntimeError('Source drift: '+n)
    for j in jobs:
        check();n=f"{j['case']}_s{j['seed']}_{j['method']}";folder=HERE/'runs'/n;folder.mkdir(parents=True)
        spec=source_freeze['specs'][j['case']];source=read(SOURCE/'runs'/n/'searched_model.json')
        selected=select(source,spec['acceptance']);write(folder/'model.json',selected)
        m=DimenSRMIMORegressor.load(folder/'model.json');X,Y=cases.training_data(j['case'],'train',j['seed']);V,Z=cases.training_data(j['case'],'validation',j['seed'])
        bundle=delivery.record(m.best_.system,m.best_.constants,spec,X,Y,V,Z)
        bundle.update(formulas=m.formulas_,sharing=m.best_.system.sharing,domain_audit=m.audit_domain(spec['bounds']),acceptance_decision=selected['post_selection'])
        write(folder/'delivery.json',bundle)
    seal=dict(frozen_unix=time.time(),freeze_sha256=sha(HERE/'FREEZE.json'),current_test_data_used_for_selection=False,
        files={p.relative_to(HERE).as_posix():sha(p) for p in (HERE/'runs').rglob('*') if p.is_file()})
    write(HERE/'ALL_DELIVERIES_FROZEN.json',seal)
    rows=[];failures=[]
    for j in jobs:
        check();n=f"{j['case']}_s{j['seed']}_{j['method']}";folder=HERE/'runs'/n;spec=source_freeze['specs'][j['case']]
        row=dict(job=j,verified=False)
        try:
            selected=read(folder/'model.json');source=read(SOURCE/'runs'/n/'searched_model.json')
            if selected!=select(source,spec['acceptance']):raise ValueError('Selection replay differs.')
            current=DimenSRMIMORegressor.load(folder/'model.json');baseline=DimenSRMIMORegressor.load(SOURCE/'runs'/n/'model.json')
            X,Y=cases.generate(j['case'],'test',j['seed']);_,trainY=cases.training_data(j['case'],'train',j['seed']);scale=trainY.std(axis=0)
            prediction=current.predict(X);reloaded=DimenSRMIMORegressor.load(folder/'model.json');np.testing.assert_array_equal(prediction,reloaded.predict(X))
            error=np.sqrt(np.mean((prediction-Y)**2,axis=0))/scale
            other=np.linalg.norm(baseline.predict(X)-Y,axis=0)/np.sqrt(len(X))/scale
            bundle=read(folder/'delivery.json');shape=bundle['structure'];before=read(SOURCE/'runs'/n/'delivery.json')['structure']
            if current.audit_domain(spec['bounds'])!=bundle['domain_audit']:raise ValueError('Domain differs.')
            row.update(verified=True,decision=selected['post_selection']['status'],development_accepted=bundle['development_accepted'],
                test_accepted=delivery.accepted(error,shape,spec['acceptance']),nrmse=error.tolist(),baseline_nrmse=other.tolist(),
                protected=(error<=other*1.1+1e-8).tolist(),structure=shape,baseline_structure=before,
                candidate_index=selected['post_selection']['candidate_index'],feasible_count=selected['post_selection']['feasible_count'])
        except Exception as e:row['error']=repr(e);failures.append(n+': '+repr(e))
        rows.append(row)
    check()
    for n,h in seal['files'].items():
        if sha(HERE/n)!=h:raise RuntimeError('Derived model drift: '+n)
    summary={}
    for method in ('balanced','aligned','normalized','combined'):
        part=[r for r in rows if r['job']['method']==method];valid=[r for r in part if r['verified']]
        summary[method]=dict(expected=12,verified=len(valid),accepted=sum(r['decision']=='accepted' for r in valid),
            test_accepted=sum(r['test_accepted'] for r in valid),protected_outputs=sum(sum(r['protected']) for r in valid),
            mean_graph_size=float(np.mean([r['structure']['shared_graph_size'] for r in valid])) if len(valid)==12 else None,
            baseline_mean_graph_size=float(np.mean([r['baseline_structure']['shared_graph_size'] for r in valid])) if len(valid)==12 else None,
            mean_nrmse=float(np.mean([np.mean(r['nrmse']) for r in valid])) if len(valid)==12 else None)
    write(HERE/'SCORES.json',dict(scored_unix=time.time(),rows=rows,all_deliveries_sha256=sha(HERE/'ALL_DELIVERIES_FROZEN.json')))
    write(HERE/'VERIFICATION.json',dict(status='passed' if not failures and len(rows)==48 and all(r['verified'] for r in rows) else 'failed',
        verified_systems=sum(r['verified'] for r in rows),failures=failures,wall_seconds=time.time()-start))
    write(HERE/'SUMMARY.json',summary)
    lines=['# 精度可行集合中的结构优先交付','','## Material Passport','','同一48个冻结候选池，仅使用训练/验证约束派生模型；无新增搜索。测试为已知开发回归。','',
        '|来源方法|核验|可行交付|测试达标|输出保护|平均共享图size|原共享图size|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in summary.items():lines.append(f'|{k}|{v["verified"]}/12|{v["accepted"]}/12|{v["test_accepted"]}/12|{v["protected_outputs"]}/24|{v["mean_graph_size"]}|{v["baseline_mean_graph_size"]}|')
    lines+=['','not_met仍保留fallback预测与未达标状态，不当作验收通过。生产默认未变；实际数据、噪声与新任务证据仍需单独验证。']
    if failures:lines+=['','失败：','']+['- '+s for s in failures]
    (HERE/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(summary=summary,failures=failures,seconds=time.time()-start)),flush=True)


if __name__=='__main__':run()
