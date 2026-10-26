"""Reconstruct frozen choices, predictions, traces and paired outcomes."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.guarded_selection_20261026 import verify_report as guarded
from experiments.guarded_selection_20261026 import selector
from experiments.shared_discovery_20261026 import cases,study,trace

core=guarded.core
core.HERE,core.ROOT,core.CASES,core.SEEDS=HERE,ROOT,cases.CASES,cases.SEEDS
core.selection_check,core.mechanism=guarded.selection_check,guarded.base_mechanism
read,write,sha,digest,require=core.read,core.write,core.sha,core.digest,core.require


def preflight():
    f,m,s,b=[read(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','BUDGET.json')]
    require(f['jobs']==study.matrix() and [r['job'] for r in s['rows']]==f['jobs'],'Matrix mismatch.')
    require(f['policy']==selector.POLICY and f['config']==read(HERE.parent/'first_round/FREEZE.json')['config'],'Policy/config drift.')
    require(b==f['budget'] and b['limit_seconds']==3000,'Budget mismatch.')
    require(b['start_unix']<=f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix']<=s['scored_unix'],'Ordering mismatch.')
    require(not f['test_labels_accessed'] and not m['test_labels_accessed'],'Premature test access.')
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json') and s['all_models_freeze_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Freeze chain mismatch.')
    study.verify_files(f['files']);study.verify_files(m['files'],HERE)
    required=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+[HERE/n for n in ('PROTOCOL.md','PREFLIGHT.json','DIAGNOSIS.json')]
    require(all(p.relative_to(ROOT).as_posix() in f['files'] for p in required),'Incomplete source freeze.')
    require(set(m['files'])=={p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()},'Incomplete result freeze.')
    for job in f['jobs']:
        folder=HERE/'runs'/core.run_name(job);cell=read(folder/'CELL_FROZEN.json')
        study.verify_files(cell['files'],folder)
        require(set(cell['files'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'},'Cell coverage differs.')
        require(read(folder/'job.json')==job and read(folder/'config.json')==f['config'],'Cell inputs differ.')
        require(f['frozen_unix']<=cell['frozen_unix']<=m['frozen_unix'],'Cell ordering differs.')
    return f,m,s,b


def source_and_trace(folder,data,classes,arrays):
    source=read(folder/'searched_model.json')
    (X,Y),(V,Z),_=arrays
    guarded.selection_check(source,V,Z,data['scales'],False,0,classes)
    require(data['post_selection']['source_model_sha256']==digest(source),'Source logical digest differs.')
    require(data['candidates']==source['candidates'] and data['search_fingerprint']==source['search_fingerprint'],'Post-selection changed search.')
    require(data==selector.derive(source,'guarded'),'Derived export differs from frozen selector.')
    discovery=trace.summarize(folder/'search_trace.jsonl',data)
    require(discovery==read(folder/'discovery.json'),'Trace summary differs.')
    # Trace carries full graphs and fit outcomes, so parameter identity can be audited.
    require(discovery['counts']['evaluation']==source['diagnostics']['proposals'],'Missing evaluator observations.')
    require(discovery['counts']['offspring']==sum(source['diagnostics']['operations'].values()),'Missing variation observations.')
    evaluations=0;actual_seed_trees={};added_by_output={}
    for line in (folder/'search_trace.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['kind']=='evaluation':
            evaluations+=row['new_evaluations']
            if row['errors'] is not None:
                prediction=classes['System'].from_dict(row['system']).evaluate(X,row['constants'])
                errors=np.sqrt(np.mean(((prediction-Y)/np.asarray(data['scales']))**2,axis=0))
                core.numerical_close(errors,row['errors'],'trace fitted training errors')
        elif row['kind']=='seed':
            actual_seed_trees.setdefault(row['output'],[]).append(row['tree'])
            if row['added']:added_by_output.setdefault(row['output'],[]).append(row['tree'])
    require(evaluations==source['diagnostics']['evaluations'],'Evaluation count differs.')
    spec=cases.design(read(folder/'job.json')['case'])
    from dimensr_mimo import SearchConfig
    from dimensr_mimo.generation import SystemGrammar
    from dimensr_mimo._single.units import unit
    from experiments.shared_discovery_20261026.seeds import nonlinear_seeds
    grammar=SystemGrammar(tuple(map(unit,spec['features'])),tuple(map(unit,spec['targets'])),tuple(map(unit,spec['constants'])),SearchConfig(**data['config']))
    treatment=read(folder/'job.json')['method']=='trig_seeds'
    for j,g in enumerate(grammar.plain):
        expected=[t.to_dict() for t in nonlinear_seeds(g)] if treatment else []
        require(added_by_output.get(j,[])==expected,'Actual appended vocabulary differs.')
    require(set(actual_seed_trees)=={0,1},'Seed generation did not cover both outputs.')
    return discovery


def decide(cells,comparison,valid,budget_ok):
    o=comparison['overall'];matched=[c for c in cells if c['job']['case']=='shared_parameter']
    recovery={}
    for method in study.METHODS:
        rows=[c for c in matched if c['job']['method']==method and c.get('verified')]
        recovery[method]=dict(expected=5,verified=len(rows),
            pool_both_scaled=sum(c['discovery']['final_pool'].get('first_scaled',0)>0 and c['discovery']['final_pool'].get('second_scaled',0)>0 for c in rows),
            pool_shared_frequency=sum(c['discovery']['final_pool'].get('shared_frequency',0)>0 for c in rows),
            selected_shared_frequency=sum(c['discovery']['selected']['shared_frequency'] for c in rows))
    technical=dict(all_40_verified=len(cells)==40 and all(c.get('verified',False) for c in cells),
        all_40_full_mechanisms=len(cells)==40 and all(c.get('mechanism_pass',False) for c in cells),
        all_40_domains_enclosed=len(cells)==40 and all(c.get('domain_pass',False) for c in cells),
        hashes_order_sources_and_traces=valid,
        both_arms_guarded=len(cells)==40 and all(c.get('validation_protected',False) for c in cells),
        intervention_observed=len(cells)==40 and all(c.get('verified') and (c['discovery']['seeds'].get('added',0)>0)==(c['job']['method']=='trig_seeds') for c in cells),
        budget_respected=budget_ok)
    gates=dict(technical,root_coverage_at_least_4_of_5=recovery['trig_seeds']['pool_both_scaled']>=4,
        shared_recovery_at_least_3_of_5=recovery['trig_seeds']['selected_shared_frequency']>=3,
        more_shared_recovery_than_control=recovery['trig_seeds']['selected_shared_frequency']>recovery['control']['selected_shared_frequency'],
        mean_reduction_at_least_10_percent=o['complete'] and o['relative_mean_reduction'] is not None and o['relative_mean_reduction']>=.1,
        absolute_reduction_above_1e_8=o['complete'] and o['mean_difference']<-1e-8,
        all_40_test_outputs_protected=o['complete'] and o['outputs_within_degradation_threshold']==40,
        mean_complexity_ratio_at_most_1_5=o['complete'] and o['ratio_of_mean_complexity'] is not None and o['ratio_of_mean_complexity']<=1.5)
    return dict(recommend=all(gates.values()),technical_pass=all(technical.values()),gates=gates,
                failed_gates=[k for k,v in gates.items() if not v],recovery=recovery)


def save(summary,verification,cells):
    write(HERE/'SUMMARY.json',summary);write(HERE/'VERIFICATION.json',verification)
    with (HERE/'metrics_per_output.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        columns=['case','seed','output','available','control_nrmse','trig_seeds_nrmse','difference','threshold','within_threshold']
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for pair in summary['comparison']['pairs']:
            for j in range(2):
                row=dict(case=pair['case'],seed=pair['seed'],output=j,available=pair['available'])
                if pair['available']:
                    row.update(control_nrmse=pair['baseline_nrmse'][j],trig_seeds_nrmse=pair['method_nrmse'][j],
                        difference=pair['per_output_differences'][j],threshold=pair['per_output_degradation_thresholds'][j],within_threshold=pair['per_output_within_threshold'][j])
                writer.writerow(row)
    fmt=core.format_number;decision=summary['recommendation']
    lines=['# 第四轮：三角函数种子与共享结构发现','','## Material Passport','',
        '已知开发任务4×新种子5×搜索组2=40系统；两组使用完全相同的guarded选择器。',
        '**'+('通过本轮候选版本门槛' if decision['recommend'] else '未通过本轮候选版本门槛，保留生产默认')+'**。',
        f'独立核验 {verification["verified_systems"]}/40，正式阶段耗时 {verification["wall_seconds_from_freeze"]:.2f} 秒（上限3000秒）。','',
        '|搜索组|完成核验|平均测试NRMSE|平均复杂度|墙钟秒|','|---|---:|---:|---:|---:|']
    for method,m in summary['methods'].items():lines.append(f'|{method}|{m["verified"]}/20|{fmt(m["mean_nrmse"])}|{fmt(m["mean_complexity"])}|{fmt(m["wall_seconds"])}|')
    lines+=['','差值为trig_seeds减control；绝对差≤1e-8为持平。','',
        '|任务|有效配对|平均差|改善/持平/退化|通过输出保护|复杂度比|','|---|---:|---:|---|---:|---:|']
    for name,r in [('全部',summary['comparison']['overall']),*summary['comparison']['strata'].items()]:
        lines.append(f'|{name}|{r["available_pairs"]}/{r["expected_pairs"]}|{fmt(r["mean_difference"])}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{r["outputs_within_degradation_threshold"]}/{r["expected_outputs"]}|{fmt(r["ratio_of_mean_complexity"])}|')
    lines+=['','## shared_parameter结构恢复','','|组|含两类参数根的最终池|含共享频率的最终池|最终选择共享频率|','|---|---:|---:|---:|']
    for name,r in decision['recovery'].items():lines.append(f'|{name}|{r["pool_both_scaled"]}/5|{r["pool_shared_frequency"]}/5|{r["selected_shared_frequency"]}/5|')
    lines+=['','## 预注册门槛','','|门槛|通过|','|---|---|']+[f'|{k}|{v}|' for k,v in decision['gates'].items()]
    lines+=['','完整逐输出数据见metrics_per_output.csv；全部40单元、失败分母、资源和原始轨迹见SUMMARY/VERIFICATION及runs。',
        '这是固定选择器条件下的种子覆盖干预；没有增加单次常数拟合预算，但新增初始化评价会改变后续随机搜索路径。',
        '直接模式仅识别显式共享频率参数身份，允许幅值及乘法换位；不是穷尽代数等价检验。',
        '验证保护相对各自原选择结果，不保证组间测试无损。已知开发任务不构成未见任务或真实数据盲评。enclosed仅为有限浮点数值筛查。',
        '初始化参照公式未单独序列化，约束按冻结参照误差重算；选择本身及候选验证误差从表达式独立重建。']
    if verification['errors']:lines+=['','## 核验错误','']+['- '+e for e in verification['errors']]
    (HERE/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def verify():
    require(not (HERE/'VERIFICATION.json').exists(),'Do not overwrite completed verification.')
    start=time.time();errors=[];cells=[];f=m=s=b=None
    try:f,m,s,b=preflight()
    except Exception as e:errors.append('Preflight: '+repr(e))
    valid=not errors
    if valid:
        from dimensr_mimo import DimenSRMIMORegressor
        from dimensr_mimo.graph import System
        classes=dict(joint=DimenSRMIMORegressor,System=System,design=cases.design)
        config_adapter={'configs':{method:f['config'] for method in study.METHODS}};cache={}
    for n,job in enumerate(study.matrix()):
        folder=HERE/'runs'/core.run_name(job)
        resource=read(folder/'resources.json') if (folder/'resources.json').exists() else {'status':'missing'}
        cell=dict(job=job,status=resource['status'],resources=resource,verified=False,mechanism_pass=False,domain_pass=False)
        if valid and resource['status']=='complete':
            try:
                require(time.time()<b['start_unix']+3000,'Budget expired during verification.')
                key=(job['case'],job['seed'])
                if key not in cache:cache[key]=[cases.generate(key[0],split,key[1]) for split in ('train','validation','test')]
                cell=core.verify_run(job,s['rows'][n],config_adapter,cache[key],classes)
                data=read(folder/'model.json');cell['discovery']=source_and_trace(folder,data,classes,cache[key])
                cell['validation_protected']=data['post_selection']['chosen_within_bounds']
            except Exception as e:
                cell.update(status='verification_failed',verified=False,error=repr(e));errors.append(core.run_name(job)+': '+repr(e))
        elif resource['status']!='complete':errors.append(core.run_name(job)+': '+resource['status'])
        cells.append(cell)
    comparison=core.compare(cells,'trig_seeds','control')
    wall=time.time()-b['start_unix'] if b else 3001
    decision=decide(cells,comparison,valid and not errors,wall<=3000)
    verification=dict(status='passed' if decision['technical_pass'] else 'failed',verified_systems=sum(c.get('verified',False) for c in cells),
        expected_systems=40,statuses=dict(Counter(c['status'] for c in cells)),errors=errors,cells=cells,
        wall_seconds_from_freeze=wall,verification_seconds=time.time()-start,
        maximum_metric_difference=max((c.get('maximum_metric_delta',0.) for c in cells),default=0),
        evidence_sha256={n:sha(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','PROTOCOL.md','verify_report.py') if (HERE/n).exists()})
    summary=dict(methods={m:core.method_summary(cells,m) for m in study.METHODS},comparison=comparison,recommendation=decision,policy=selector.POLICY)
    save(summary,verification,cells)
    print(json.dumps(dict(status=verification['status'],verified_systems=verification['verified_systems'],errors=errors,recommendation=decision,overall=comparison['overall']),indent=2))
    return 0 if verification['status']=='passed' else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('verify',));p.parse_args();sys.exit(verify())
