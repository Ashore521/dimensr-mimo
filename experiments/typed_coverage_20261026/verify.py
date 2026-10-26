"""Reload raw graphs and traces, recompute metrics, preserve every failure."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor
from dimensr_mimo._single import DimenSRRegressor
from dimensr_mimo.graph import System
from experiments.guarded_selection_20261026 import selector
from experiments.typed_coverage_20261026 import study,cases,audit,delivery
read,write,sha=study.read,study.write,study.sha


def require(condition,message):
    if not condition:raise ValueError(message)


def trace_check(folder,spec,observation):
    counts=defaultdict(Counter);supports=defaultdict(set);evaluations=0;new=0;seeds=Counter()
    with (folder/'trace.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line)
            if row['kind']=='seed':seeds[row['output'],row['source']]+=1
            if row['kind']!='evaluation':continue
            system=System.from_dict(row['system']);audit.units(system,spec['features'],spec['targets'],spec['constants'])
            evaluations+=1;new+=row['new_evaluations']
            for s in row['sources']:
                key=f"output{s['output']}/{s['source']}/{s['family']}"
                counts[key]['submitted']+=1;counts[key]['new_evaluated']+=row['new_evaluations']
                counts[key]['valid_new']+=int(row['errors'] is not None and row['new_evaluations']>0)
                if row['new_evaluations']:supports[key].update(s['support'])
    coverage={k:dict(v,feature_support=sorted(supports[k])) for k,v in counts.items()}
    require(coverage==observation['coverage'],'Actual seed coverage counts differ.')
    require(evaluations==observation['unit_checks']['evaluator_proposals'],'Unit-check denominator differs.')
    require(sum(v['submitted'] for k,v in coverage.items() if k.startswith('output0/'))<=180,'Initialization cap exceeded.')
    for j in range(2):
        require(dict(Counter({s:v for (i,s),v in seeds.items() if i==j}))==observation['generated_by_output'][j],'Generated seed counts differ.')
    return dict(evaluation_graphs_checked=evaluations,new_evaluations=new,coverage=coverage)


def verify_cell(job,score):
    folder=HERE/'runs'/study.name(job);spec=cases.design(job['case']);result=read(folder/'result.json')
    require(result['status']=='complete' and not result['test_scores_computed'],'Incomplete result or premature test.')
    single=job['method']=='independent';cls=DimenSRRegressor if single else DimenSRMIMORegressor
    models=[cls.load(folder/n) for n in result['models']]
    X,Y=cases.training_data(job['case'],'train',job['seed']);V,Z=cases.training_data(job['case'],'validation',job['seed'])
    T,U=cases.generate(job['case'],'test',job['seed']);bundle=read(folder/'delivery.json')
    system,constants=delivery.graph(models,single)
    raw=System.from_dict(bundle['system']);values=np.asarray([p['value'] for p in bundle['parameters']])
    require(raw==system and np.array_equal(values,constants),'Delivered graph or parameters differ.')
    require(audit.units(raw,spec['features'],spec['targets'],spec['constants'])==bundle['units'],'Parameter unit report differs.')
    require(audit.complexity(raw)==bundle['structure']==score['structure'],'Structure report differs.')
    expected=delivery.record(raw,values,spec,X,Y,V,Z)
    require(all(bundle[k]==v for k,v in expected.items()),'Physical delivery metadata differs.')
    require(result['prediction_hashes']==[study.worker.prediction_hashes(m,X,V) for m in models],'Saved prediction hashes differ.')
    for A in (X,V,T):
        pred=np.column_stack([m.predict(A) for m in models]) if single else models[0].predict(A)
        np.testing.assert_array_equal(pred,raw.evaluate(A,values))
    errors=np.linalg.norm(raw.evaluate(T,values)-U,axis=0)/np.sqrt(len(T))/Y.std(axis=0)
    np.testing.assert_allclose(errors,score['nrmse'],rtol=1e-12,atol=1e-13)
    accepted=delivery.accepted(errors,audit.complexity(raw),spec['acceptance'])
    require(accepted==score['test_accepted'],'Acceptance differs.')
    domains=[m.audit_domain(input_bounds=spec['bounds']) for m in models]
    require(domains==bundle['domain_audits']==result['audits'],'Domain audit differs.')
    mechanism=study.mechanism(folder);require(mechanism==score['mechanism'],'Mechanism differs.')
    evidence=dict(job=job,status='verified',verified=True,mechanism_pass=mechanism['full_mechanism_pass'],
        domain_pass=all(a['status']=='enclosed' for a in domains),nrmse=errors.tolist(),mean_nrmse=float(errors.mean()),
        structure=bundle['structure'],test_accepted=accepted,development_accepted=bundle['development_accepted'])
    if not single:
        source=read(folder/'searched_model.json');saved=read(folder/'model.json')
        require(saved==selector.derive(source,'guarded'),'Frozen guarded selection differs.')
        pool=read(folder/'pool.json');require(pool==delivery.pool(source,spec),'Candidate-pool report differs.')
        for c in source['candidates']:
            graph=System.from_dict(c['system'])
            for A,B,key in ((X,Y,'errors'),(V,Z,'validation_errors')):
                err=np.sqrt(np.mean((graph.evaluate(A,c['constants'])-B)**2,axis=0))/Y.std(axis=0)
                np.testing.assert_allclose(err,c[key],rtol=1e-9,atol=1e-11)
        evidence.update(pool_feasible=pool['feasible_count'],pool_count=pool['candidates'],
            trace=trace_check(folder,spec,result['observation']))
    return evidence


def summarize(cells):
    methods={};pairs=[]
    for method in study.METHODS:
        rows=[c for c in cells if c['job']['method']==method];valid=[c for c in rows if c.get('verified')]
        methods[method]=dict(expected=12,verified=len(valid),test_accepted=sum(c.get('test_accepted',False) for c in valid),
            development_accepted=sum(c.get('development_accepted',False) for c in valid),
            both_accepted=sum(c.get('test_accepted',False) and c.get('development_accepted',False) for c in valid),
            mean_nrmse=float(np.mean([c['mean_nrmse'] for c in valid])) if len(valid)==12 else None,
            feasible_pool_systems=sum(c.get('pool_feasible',0)>0 for c in valid) if method!='independent' else None,
            statuses=dict(Counter(c['status'] for c in rows)))
    index={(c['job']['case'],c['job']['seed'],c['job']['method']):c for c in cells}
    for case in cases.CASES:
        for seed in cases.SEEDS:
            a,b=index[case,seed,'balanced'],index[case,seed,'prefix'];row=dict(case=case,seed=seed,available=a.get('verified',False) and b.get('verified',False))
            if row['available']:
                delta=np.asarray(a['nrmse'])-b['nrmse'];bounds=np.asarray(b['nrmse'])*1.1+1e-8
                row.update(difference=delta.tolist(),protected=(np.asarray(a['nrmse'])<=bounds).tolist(),
                    mean_difference=float(delta.mean()),balanced_accepted=a['test_accepted'],prefix_accepted=b['test_accepted'])
            pairs.append(row)
    return dict(methods=methods,paired_balanced_minus_prefix=pairs,
        protected_outputs=sum(sum(p.get('protected',[])) for p in pairs),expected_protected_outputs=24,
        production_default_changed=False,claim_scope='Known noiseless dimensional development strata; no blind or practical-data validation claim.')


def verify():
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('Never overwrite verification.')
    f,m,s=[read(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json')]
    study.verify_files(f['files']);study.verify_files(m['files'],HERE)
    require(f['jobs']==study.matrix()==[r['job'] for r in s['rows']],'Matrix changed.')
    require(f['specs']=={c:cases.design(c) for c in cases.CASES} and f['policy']==selector.POLICY,'Protocol drift.')
    require(f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix']<=s['scored_unix'],'Freeze ordering failed.')
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json') and s['all_models_freeze_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Freeze chain failed.')
    require(not f['test_labels_accessed'] and not m['test_labels_accessed'],'Premature test release.')
    cells=[];failures=[]
    for job,score in zip(f['jobs'],s['rows']):
        folder=HERE/'runs'/study.name(job);cell=dict(job=job,status=score['status'],verified=False)
        try:
            require(time.time()<study.budget(),'Verification budget expired.')
            seal=read(folder/'CELL_FROZEN.json');study.verify_files(seal['files'],folder)
            require(f['frozen_unix']<=seal['frozen_unix']<=m['frozen_unix'],'Cell ordering failed.')
            require(read(folder/'job.json')==job and read(folder/'config.json')==f['config'],'Job input drift.')
            require(score['status']=='complete','Incomplete cell: '+score['status'])
            cell=verify_cell(job,score)
        except Exception as e:cell.update(status='verification_failed' if score['status']=='complete' else score['status'],error=repr(e));failures.append(study.name(job)+': '+repr(e))
        cells.append(cell)
    summary=summarize(cells);technical=len(cells)==36 and all(c.get('verified') and c.get('mechanism_pass') and c.get('domain_pass') for c in cells)
    report=dict(status='passed' if technical else 'failed',verified_systems=sum(c.get('verified',False) for c in cells),expected_systems=36,
        technical_pass=technical,failures=failures,cells=cells,wall_seconds=time.time()-f['budget']['start_unix'],
        freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json'))
    write(HERE/'VERIFICATION.json',report);write(HERE/'SUMMARY.json',summary)
    lines=['# 固定初始化预算与量纲开发层','','## Material Passport','',summary['claim_scope'],'',
        f'完整矩阵36系统，独立重载核验{report["verified_systems"]}/36，技术门槛通过：{technical}。',
        '每输出NRMSE≤1e-4；共享图size≤24，独立参数≤6，展开节点合计≤30，最大输出深度≤6。','',
        '|方法|核验|训练和验证达标|测试达标|两者均达标|池内有可行解|平均测试NRMSE|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in summary['methods'].items():lines.append(f'|{k}|{v["verified"]}/12|{v["development_accepted"]}/12|{v["test_accepted"]}/12|{v["both_accepted"]}/12|{v["feasible_pool_systems"]}|{v["mean_nrmse"]}|')
    lines+=['',f'balanced相对prefix逐输出保护：{summary["protected_outputs"]}/24。生产默认未变。','',
        '独立方法为逐输出独立DimenSR；其候选池未枚举笛卡尔组合，池内可行解记为未评估。',
        '全部公式为开发者已知任务；单位及重载通过不代表应用精度或实际数据适用性。enclosed只代表数值域筛查。',
        '每个完成单元的delivery.json保存SI物理变量、原始图、显式参数及单位、共享结构、适用域与验收。',
        '失败和超时保留在分母，不补跑挑选。正式精度优先选择器尚未用于本轮。']
    if failures:lines+=['','失败记录：','']+['- '+x for x in failures]
    (HERE/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],verified_systems=report['verified_systems'],summary=summary,failures=failures),ensure_ascii=False),flush=True)


if __name__=='__main__':verify()
