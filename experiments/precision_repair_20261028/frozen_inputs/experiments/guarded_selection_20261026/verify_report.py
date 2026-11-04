"""Independent selection reconstruction, reused metric checks, full denominators."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.guarded_selection_20261026 import cases, study, selector
from experiments.second_round_20261026 import verify_report as core

core.HERE, core.ROOT, core.CASES, core.SEEDS = HERE, ROOT, cases.CASES, cases.SEEDS
read,write,sha,digest,require=core.read,core.write,core.sha,core.digest,core.require
base_selection,base_mechanism=core.selection_check,core.mechanism


def selection_check(data,V,Z,scales,independent,output,classes):
    if independent:
        return base_selection(data,V,Z,scales,True,output,classes)
    core.numerical_close(scales,data['scales'],'frozen training normalization')
    scales=np.asarray(data['scales'])
    pool=data['candidates']; cfg=data['config']; errors=[]; scores=[]; violations=[]; delta=0.
    for c in pool:
        pred=classes['System'].from_dict(c['system']).evaluate(V,c['constants'])
        e=np.sqrt(np.mean(((pred-Z)/scales)**2,axis=0))
        bound=np.asarray(data['reference_validation'])*(1+cfg['degradation_relative'])+cfg['degradation_absolute']
        violation=float(np.maximum(e-bound,0).sum()) if cfg['use_constraints'] else 0.
        score=float(e.mean()+cfg['selection_complexity_weight']*c['complexity']/(Z.shape[1]*cfg['max_nodes']))
        for a,b in [(e,c['validation_errors']),(violation,c['violation']),(score,c['score'])]:
            delta=max(delta,core.numerical_close(a,b,'candidate validation reconstruction'))
        errors.append(e); scores.append(score); violations.append(violation)
    e=np.asarray(errors); means=e.mean(axis=1); complexity=np.asarray([c['complexity'] for c in pool])
    anchor=min(range(len(pool)),key=lambda i:(violations[i],scores[i],complexity[i]))
    tier=np.flatnonzero(np.asarray(violations)==violations[anchor])
    policy=selector.POLICY
    bounds=e[anchor]*(1+policy['degradation_relative'])+policy['degradation_absolute']
    protected=[int(i) for i in tier if np.all(e[i]<=bounds)]
    require(anchor in protected,'Anchor excluded from validation guard.')
    near_bound=float(min(means[i] for i in protected)*(1+policy['equivalent_relative'])+policy['equivalent_absolute'])
    near=[i for i in protected if means[i]<=near_bound]
    method=data.get('post_selection',{}).get('method','original')
    require(method in ('original','no_penalty','guarded'),'Unknown exported selection rule.')
    index=anchor if method=='original' else (min(tier,key=lambda i:(means[i],complexity[i])) if method=='no_penalty' else min(near,key=lambda i:(complexity[i],means[i],i)))
    index=int(index); chosen=pool[index]
    require(all(chosen[k]==v for k,v in data['best'].items()),'Independent choice differs from model.')
    payload=[{k:c[k] for k in selector.FIELDS} for c in pool]
    require(digest(payload)==data['diagnostics']['frozen_candidate_sha256'],'Pool hash differs.')
    core.numerical_close(e[index],data['validation_nrmse'],'selected validation error')
    rule=dict(method=method,candidate_index=index,anchor_index=anchor,policy=dict(policy),validation_bounds=bounds.tolist(),
              minimum_violation_tier=len(tier),protected_count=len(protected),near_optimal_count=len(near),
              near_optimal_mean_bound=near_bound,chosen_validation_errors=e[index].tolist(),
              chosen_within_bounds=bool(np.all(e[index]<=bounds)),candidate_sha256=digest(payload[index]))
    if method!='original':
        actual=data['post_selection']
        require(all(actual[k]==v for k,v in rule.items()),'Post-selection record differs from independent reconstruction.')
        require(actual['search_reused'] and actual['training_configuration_unchanged'],'Missing reuse disclosure.')
        rule=actual
    require(method!='guarded' or rule['chosen_within_bounds'],'Guard failed on validation outputs.')
    return dict(candidate_index=index,candidate_sha256=digest(chosen),best_sha256=digest(data['best']),
                pool_sha256=digest(pool),candidate_count=len(pool),maximum_metric_delta=delta,rule=rule)


def mechanism(events,result,independent):
    if result.get('search_reused'):
        job=result['job']; expected=f"{job['case']}_s{job['seed']}_original"
        require(result['source_search']==expected,'Wrong search dependency.')
        folder=HERE/'runs'/expected
        require(result['source_model_sha256']==sha(folder/'model.json'),'Source model hash differs.')
        source=read(folder/'result.json')
        events=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
        return base_mechanism(events,source,False)
    return base_mechanism(events,result,independent)


core.selection_check,core.mechanism=selection_check,mechanism


def preflight():
    f,m,s,b=[read(HERE/name) for name in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','BUDGET.json')]
    require(f['jobs']==study.matrix() and len(s['rows'])==80,'Full matrix mismatch.')
    require([r['job'] for r in s['rows']]==f['jobs'],'Score row ordering mismatch.')
    require(f['policy']==selector.POLICY and f['config']==read(HERE.parent/'first_round/FREEZE.json')['config'],'Policy/config drift.')
    require(b==f['budget'] and b['limit_seconds']==3600,'Budget mismatch.')
    require(b['start_unix']<=f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix']<=s['scored_unix'],'Freeze/test ordering mismatch.')
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json') and s['all_models_freeze_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Freeze linkage mismatch.')
    require(time.time()<b['start_unix']+3600,'Budget expired before verification.')
    study.verify_files(f['files']); study.verify_files(m['files'],HERE)
    required=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+[HERE/'PROTOCOL.md',HERE/'PREFLIGHT.json']
    require(all(p.relative_to(ROOT).as_posix() in f['files'] for p in required),'Incomplete source coverage.')
    require(set(m['files'])=={p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()},'Incomplete result coverage.')
    for job in f['jobs']:
        folder=HERE/'runs'/core.run_name(job); cell=read(folder/'CELL_FROZEN.json')
        study.verify_files(cell['files'],folder)
        require(set(cell['files'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'},'Cell freeze coverage differs.')
        require(read(folder/'job.json')==job and read(folder/'config.json')==f['config'],'Cell inputs differ.')
        require(f['frozen_unix']<=cell['frozen_unix']<=m['frozen_unix'],'Cell timing differs.')
    return f,m,s,b


def decide(cells,comparison,isolation,valid,budget_ok):
    o=comparison['overall']; searches=[c for c in cells if c['job']['method'] in ('original','independent')]
    guarded=[c for c in cells if c['job']['method']=='guarded']
    gates=dict(all_80_verified=len(cells)==80 and all(c.get('verified',False) for c in cells),
               all_40_actual_searches_complete=len(searches)==40 and all(c.get('mechanism_pass',False) for c in searches),
               all_80_domains_enclosed=all(c.get('domain_pass',False) for c in cells) and len(cells)==80,
               hashes_order_and_sources=valid,all_20_shared_pools=len(isolation)==20 and all(isolation),
               all_40_validation_outputs_protected=len(guarded)==20 and all(c.get('validation_protected',False) for c in guarded),
               budget_respected=budget_ok,
               mean_error_reduction_at_least_10_percent=o['complete'] and o['relative_mean_reduction'] is not None and o['relative_mean_reduction']>=.1,
               absolute_reduction_above_1e_8=o['complete'] and o['mean_difference'] < -1e-8,
               all_40_test_outputs_protected=o['complete'] and o['outputs_within_degradation_threshold']==40,
               mean_complexity_ratio_at_most_1_5=o['complete'] and o['ratio_of_mean_complexity'] is not None and o['ratio_of_mean_complexity']<=1.5)
    return dict(recommend=all(gates.values()),gates=gates,failed_gates=[k for k,v in gates.items() if not v])


def save_outputs(summary,verification,cells):
    write(HERE/'SUMMARY.json',summary);write(HERE/'VERIFICATION.json',verification)
    fields=['case','seed','method','status','output','rmse','nrmse','r2','original_nrmse','difference','threshold','within_threshold']
    index={(c['job']['case'],c['job']['seed'],c['job']['method']):c for c in cells}
    with (HERE/'metrics_per_output.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in cells:
            job=c['job'];base=index[(job['case'],job['seed'],'original')]
            for j in range(2):
                row={k:job[k] for k in ('case','seed','method')};row.update(status=c['status'],output=j)
                if c.get('verified'):
                    row.update({k:c['metrics'][k][j] for k in ('rmse','nrmse','r2')})
                    if base.get('verified'):
                        v=base['metrics']['nrmse'][j];row.update(original_nrmse=v,difference=row['nrmse']-v,threshold=v*1.1+1e-8,within_threshold=row['nrmse']<=v*1.1+1e-8)
                writer.writerow(row)
    decision=summary['recommendation'];fmt=core.format_number
    lines=['# 第三轮：逐输出保护选择器的新划分复验','','## Material Passport','',
           '4个已知合成任务×5个新种子×4方法=80个最终系统；实际40次搜索，40次派生选择。',
           '三个联合方法共享同一冻结候选池。验证集保护不保证测试集不退化。',
           '', '**'+('通过本轮配置推荐门槛' if decision['recommend'] else '未通过推荐门槛，保持原默认配置')+'**。',
           f'核验{verification["verified_systems"]}/80；实际完整搜索{verification["verified_searches"]}/40；候选池一致{verification["identical_pools"]}/20。','',
           '|方法|核验|平均测试NRMSE|平均复杂度|增量墙钟秒|单独使用估计墙钟秒|','|---|---:|---:|---:|---:|---:|']
    for name,m in summary['methods'].items():
        lines.append(f'|{name}|{m["verified"]}/20|{fmt(m["mean_nrmse"])}|{fmt(m["mean_complexity"])}|{fmt(m["wall_seconds"])}|{fmt(m["standalone_wall_seconds"])}|')
    lines += ['','增量耗时只计本轮实际新增计算；派生方法单独使用成本加回共享原搜索。未把共享搜索重复累加为总实验耗时。CPU/RSS及逐输出误差见SUMMARY/VERIFICATION/CSV。','']
    for name,comp in summary['comparisons'].items():
        lines += [f'## {name}','','差值为前者减后者，负值更好；1e-8以内为持平。','',
                  '|任务|有效配对|平均差|中位差|改善/持平/退化|复杂度比|','|---|---:|---:|---:|---|---:|']
        for name,r in [('全部',comp['overall']),*comp['strata'].items()]:
            lines.append(f'|{name}|{r["available_pairs"]}/{r["expected_pairs"]}|{fmt(r["mean_difference"])}|{fmt(r["median_difference"])}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{fmt(r["ratio_of_mean_complexity"])}|')
    lines += ['','## 推荐门槛','','|门槛|通过|','|---|---|']+[f'|{k}|{v}|' for k,v in decision['gates'].items()]
    main=summary['comparisons']['guarded_minus_original']['overall']
    lines += ['',f'主要测试逐输出保护通过{main["outputs_within_degradation_threshold"]}/40；全部20配对与完整失败分母保存在SUMMARY.json。',
              f'正式阶段累计{verification["wall_seconds_from_freeze"]:.2f}秒，上限3600秒。',
              '', '这是开发任务新划分重复验证，不是未见任务或真实数据盲评；没有更改搜索机制或增加常数拟合预算，不构成共享/神经/GP机制收益证明。enclosed仅为有限浮点区间筛查。',
              '冻结的初始化参照公式没有单独序列化；候选约束按已冻结参照误差重算，不声称重新恢复了该参照公式。原参照模型的逐输出验证误差及新保护选择本身均从原始表达式独立重算。']
    if verification['errors']:lines+=['','## 失败记录','']+['- '+e for e in verification['errors']]
    (HERE/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def verify():
    require(not (HERE/'VERIFICATION.json').exists(),'Do not overwrite completed verification evidence.')
    start=time.time();errors=[];cells=[];f=m=s=b=None
    try:f,m,s,b=preflight()
    except Exception as e:errors.append('Preflight: '+repr(e))
    valid=not errors
    if valid:
        from dimensr_mimo import DimenSRMIMORegressor
        from dimensr_mimo._single import DimenSRRegressor
        from dimensr_mimo.graph import System
        from dimensr_mimo._single.expression import Node
        classes=dict(joint=DimenSRMIMORegressor,single=DimenSRRegressor,System=System,Node=Node,design=cases.design)
        config_adapter={'configs':{method:f['config'] for method in study.METHODS}}
        cache={}
    for n,job in enumerate(study.matrix()):
        folder=HERE/'runs'/core.run_name(job)
        resource=read(folder/'resources.json') if (folder/'resources.json').exists() else {'status':'missing'}
        cell=dict(job=job,status=resource['status'],resources=resource,verified=False,mechanism_pass=False,domain_pass=False)
        if valid and resource['status']=='complete':
            try:
                require(time.time()<b['start_unix']+3600,'Deadline reached during verification.')
                key=(job['case'],job['seed'])
                if key not in cache:cache[key]=[cases.generate(key[0],split,key[1]) for split in ('train','validation','test')]
                cell=core.verify_run(job,s['rows'][n],config_adapter,cache[key],classes)
                if job['method'] in ('guarded','no_penalty'):
                    source=HERE/'runs'/f"{job['case']}_s{job['seed']}_original"/'model.json'
                    data=read(folder/'model.json');record=data['post_selection']
                    require(record['source_model_sha256']==digest(read(source)),'Source logical digest mismatch.')
                    cell['validation_protected']=record['chosen_within_bounds']
                    cell['source_search']=source.parent.name
            except Exception as e:
                cell.update(status='verification_failed',verified=False,error=repr(e));errors.append(core.run_name(job)+': '+repr(e))
        elif resource['status']!='complete':errors.append(core.run_name(job)+': '+resource['status'])
        cells.append(cell)
    index={(c['job']['case'],c['job']['seed'],c['job']['method']):c for c in cells};isolation=[]
    for c in cases.CASES:
        for seed in cases.SEEDS:
            group=[index[c,seed,method] for method in ('original','no_penalty','guarded')]
            isolation.append(all(r.get('verified') for r in group) and len({r.get('search_fingerprint') for r in group})==1 and len({r.get('frozen_pool_sha256') for r in group})==1)
    comparisons={f'{a}_minus_{z}':core.compare(cells,a,z) for a,z in [('guarded','original'),('no_penalty','original'),('guarded','no_penalty'),('original','independent'),('no_penalty','independent'),('guarded','independent')]}
    wall=time.time()-b['start_unix'] if b else 3601.;decision=decide(cells,comparisons['guarded_minus_original'],isolation,valid and not errors,wall<=3600)
    methods={method:core.method_summary(cells,method) for method in study.METHODS}
    for method,r in methods.items():r['standalone_wall_seconds']=r['wall_seconds']+(methods['original']['wall_seconds'] if method in ('guarded','no_penalty') else 0.)
    searches=[c for c in cells if c['job']['method'] in ('original','independent')]
    verification=dict(status='passed' if all(list(decision['gates'].values())[:7]) else 'failed',verified_systems=sum(c.get('verified',False) for c in cells),
       verified_searches=sum(c.get('verified',False) and c.get('mechanism_pass',False) for c in searches),identical_pools=sum(isolation),
       expected_systems=80,expected_searches=40,statuses=dict(Counter(c['status'] for c in cells)),errors=errors,cells=cells,
       wall_seconds_from_freeze=wall,verification_seconds=time.time()-start,maximum_metric_difference=max((c.get('maximum_metric_delta',0.) for c in cells),default=0.),
       evidence_sha256={name:sha(HERE/name) for name in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','PROTOCOL.md','selector.py','verify_report.py') if (HERE/name).exists()})
    summary=dict(methods=methods,comparisons=comparisons,recommendation=decision,policy=selector.POLICY,
                 scope='80 final systems from 40 actual searches; known-task development replication; no blind-data claim')
    save_outputs(summary,verification,cells)
    verification['wall_seconds_from_freeze']=time.time()-b['start_unix'] if b else 3601.
    if verification['wall_seconds_from_freeze']>3600:
        verification['status']='failed';summary['recommendation']=decide(cells,comparisons['guarded_minus_original'],isolation,valid,False)
    save_outputs(summary,verification,cells)
    print(json.dumps({k:verification[k] for k in ('status','verified_systems','verified_searches','identical_pools','wall_seconds_from_freeze','errors')},indent=2))
    print(json.dumps(dict(recommendation=decision,main=comparisons['guarded_minus_original']['overall']),indent=2))
    return 0 if verification['status']=='passed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('verify',))
    parser.parse_args();sys.exit(verify())
