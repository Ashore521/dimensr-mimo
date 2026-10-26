"""Frozen public-task verification with all failures retained."""
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
from experiments.guarded_selection_20261026 import verify_report as guarded,selector
from experiments.public_transfer_20261026 import data as cases,study

core=guarded.core
core.HERE,core.ROOT,core.CASES,core.SEEDS=HERE,ROOT,cases.CASES,cases.SEEDS
core.selection_check=guarded.selection_check
read,write,sha,digest,require=core.read,core.write,core.sha,core.digest,core.require


def mechanism(events,result,independent):
    if result.get('search_reused'):
        job=result['job'];source=HERE/'runs'/f"{job['case']}_s{job['seed']}_original"
        require(result['source_search']==source.name and result['source_model_sha256']==sha(source/'model.json'),'Wrong reused source.')
        events=[json.loads(s) for s in (source/'events.jsonl').read_text().splitlines()]
        result=read(source/'result.json')
    return guarded.base_mechanism(events,result,independent)


core.mechanism=mechanism


def preflight():
    f,m,s,b=[read(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','BUDGET.json')]
    require(f['jobs']==study.matrix() and [r['job'] for r in s['rows']]==f['jobs'] and len(f['jobs'])==18,'Matrix mismatch.')
    require(f['policy']==selector.POLICY and f['config']==read(HERE.parent/'first_round/FREEZE.json')['config'],'Config/policy drift.')
    require(b==f['budget'] and b['limit_seconds']==1800,'Budget differs.')
    require(b['start_unix']<=f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix']<=s['scored_unix'],'Ordering mismatch.')
    require(not f['test_labels_accessed'] and not m['test_labels_accessed'],'Premature test release.')
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json') and s['all_models_freeze_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Freeze chain differs.')
    study.verify_files(f['files']);study.verify_files(m['files'],HERE)
    required=list((ROOT/'src').rglob('*.py'))+list(HERE.glob('*.py'))+[HERE/n for n in ('PROTOCOL.md','PREFLIGHT.json','ALGORITHM.json','DATA_MANIFEST.json','SEED_CAPACITY_AUDIT.json')]
    require(all(p.relative_to(ROOT).as_posix() in f['files'] for p in required),'Incomplete source coverage.')
    require(set(m['files'])=={p.relative_to(HERE).as_posix() for p in (HERE/'runs').rglob('*') if p.is_file()},'Incomplete results.')
    choice=read(HERE/'ALGORITHM.json');prior=read(HERE.parent/'shared_discovery_20261026/SUMMARY.json')
    require(prior['recommendation']['technical_pass'] and choice['trig_seeds']==prior['recommendation']['recommend'],'Algorithm choice differs from predeclared rule.')
    require(choice['development_summary_sha256']==sha(HERE.parent/'shared_discovery_20261026/SUMMARY.json'),'Development source drift.')
    for job in f['jobs']:
        folder=HERE/'runs'/core.run_name(job);cell=read(folder/'CELL_FROZEN.json')
        study.verify_files(cell['files'],folder)
        require(set(cell['files'])=={p.name for p in folder.iterdir() if p.is_file() and p.name!='CELL_FROZEN.json'},'Cell coverage differs.')
        require(read(folder/'job.json')==job and read(folder/'config.json')==f['config'],'Cell inputs differ.')
        require(f['frozen_unix']<=cell['frozen_unix']<=m['frozen_unix'],'Cell time differs.')
    return f,m,s,b


def method_summary(cells,method):
    rows=[c for c in cells if c['job']['method']==method];complete=len(rows)==6 and all(c.get('verified') for c in rows)
    return dict(expected=6,verified=sum(c.get('verified',False) for c in rows),statuses=dict(Counter(c['status'] for c in rows)),complete=complete,
        mean_nrmse=float(np.mean([c['metrics']['mean_nrmse'] for c in rows])) if complete else None,
        mean_complexity=float(np.mean([c['metrics']['complexity'] for c in rows])) if complete else None,
        wall_seconds=sum(c['resources'].get('wall_seconds',0.) for c in rows),
        sampled_cpu_seconds=sum(c['resources'].get('process_tree_cpu_seconds',0.) for c in rows),
        peak_rss_bytes=max((c['resources'].get('process_tree_peak_rss_bytes',0) for c in rows),default=0))


def decide(cells,comparison,valid,budget_ok):
    o=comparison['overall']
    technical=dict(all_18_verified=len(cells)==18 and all(c.get('verified',False) for c in cells),
        all_mechanisms=len(cells)==18 and all(c.get('mechanism_pass',False) for c in cells),
        all_training_and_extended_domains=len(cells)==18 and all(c.get('domain_pass',False) and c.get('extended_domain_pass',False) for c in cells),
        hashes_sources_and_test_isolation=valid,budget_respected=budget_ok)
    gates=dict(technical,all_12_candidate_test_outputs_protected=o['complete'] and o['outputs_within_degradation_threshold']==12,
        mean_reduction_at_least_10_percent=o['complete'] and o['relative_mean_reduction'] is not None and o['relative_mean_reduction']>=.1,
        absolute_reduction_above_1e_8=o['complete'] and o['mean_difference']<-1e-8,
        mean_complexity_ratio_at_most_1_5=o['complete'] and o['ratio_of_mean_complexity'] is not None and o['ratio_of_mean_complexity']<=1.5)
    return dict(recommend=all(gates.values()),technical_pass=all(technical.values()),gates=gates,failed_gates=[k for k,v in gates.items() if not v])


def save(summary,verification):
    write(HERE/'SUMMARY.json',summary);write(HERE/'VERIFICATION.json',verification)
    columns=['comparison','case','seed','output','available','method_nrmse','baseline_nrmse','difference','threshold','within_threshold']
    with (HERE/'metrics_per_output.csv').open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=columns);writer.writeheader()
        for name,comparison in summary['comparisons'].items():
            for p in comparison['pairs']:
                for j in range(2):
                    row=dict(comparison=name,case=p['case'],seed=p['seed'],output=j,available=p['available'])
                    if p['available']:row.update(method_nrmse=p['method_nrmse'][j],baseline_nrmse=p['baseline_nrmse'][j],difference=p['per_output_differences'][j],threshold=p['per_output_degradation_thresholds'][j],within_threshold=p['per_output_within_threshold'][j])
                    writer.writerow(row)
    fmt=core.format_number;decision=summary['recommendation']
    lines=['# 第五轮：固定公开双输出任务迁移','','## Material Passport','',
        'UCI 242 Energy Efficiency（建筑模拟）和UCI 275 Bike Sharing day.csv（实际租赁计数的双输出改编）。每任务一个固定划分、3个优化种子、3方法，18最终系统。',
        '来源、许可及拆分见[协议](PROTOCOL.md)与DATA_MANIFEST.json。候选算法由第四轮开发门槛按预定规则确定。',
        '**'+('通过本轮有限任务迁移门槛' if decision['recommend'] else '未通过本轮有限任务迁移门槛，保留生产默认')+'**。',
        f'核验{verification["verified_systems"]}/18；累计{verification["wall_seconds_from_freeze"]:.2f}秒，上限1800秒。','',
        '|方法|核验|平均测试NRMSE|平均复杂度|实际增量秒|单独使用估计秒|','|---|---:|---:|---:|---:|---:|']
    for name,m in summary['methods'].items():lines.append(f'|{name}|{m["verified"]}/6|{fmt(m["mean_nrmse"])}|{fmt(m["mean_complexity"])}|{fmt(m["wall_seconds"])}|{fmt(m["standalone_wall_seconds"])}|')
    for name,c in summary['comparisons'].items():
        lines+=['','## '+name,'','差值为前者减后者，绝对差≤1e-8为持平。','',
            '|任务|有效配对|平均差|改善/持平/退化|逐输出保护|复杂度比|','|---|---:|---:|---|---:|---:|']
        for case,r in [('全部',c['overall']),*c['strata'].items()]:lines.append(f'|{case}|{r["available_pairs"]}/{r["expected_pairs"]}|{fmt(r["mean_difference"])}|{r["wins"]}/{r["ties"]}/{r["losses"]}|{r["outputs_within_degradation_threshold"]}/{r["expected_outputs"]}|{fmt(r["ratio_of_mean_complexity"])}|')
    lines+=['','## 预定门槛','','|门槛|通过|','|---|---|']+[f'|{k}|{v}|' for k,v in decision['gates'].items()]
    lines+=['','三个优化种子共用同一个数据划分，不是三个独立数据样本。建筑只有12个形状组；租赁只使用一个时间切分，保留官方数值分类编码。',
        f'容量审计显示追加种子不进入初始化前180个位置。6组实际搜索中，指纹和候选池均相同的有{sum(p["identical"] for p in summary["search_pairs"])}/6；相同组的差异只归于最终选择，不构成共享搜索收益。',
        '新任务仅指当前可检索本地实验记录未使用；删除的历史无法完全排查。不得称为从未被开发者知晓的任务或普遍优越性证据。',
        '原始公开数据留在忽略缓存，训练/验证变换冻结后才放行测试读取；不发布任何私有应用数据。enclosed仅为有限浮点数值筛查。']
    if verification['errors']:lines+=['','## 失败记录','']+['- '+e for e in verification['errors']]
    (HERE/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def verify():
    require(not (HERE/'VERIFICATION.json').exists(),'Do not overwrite completed verification.')
    started=time.time();errors=[];cells=[];f=m=s=b=None
    try:f,m,s,b=preflight()
    except Exception as e:errors.append('Preflight: '+repr(e))
    valid=not errors
    if valid:
        from dimensr_mimo import DimenSRMIMORegressor
        from dimensr_mimo._single import DimenSRRegressor
        from dimensr_mimo.graph import System
        from dimensr_mimo._single.expression import Node
        classes=dict(joint=DimenSRMIMORegressor,single=DimenSRRegressor,System=System,Node=Node,design=cases.design)
        adapter={'configs':{k:f['config'] for k in study.METHODS}};cache={}
    for n,job in enumerate(study.matrix()):
        folder=HERE/'runs'/core.run_name(job);resource=read(folder/'resources.json')
        cell=dict(job=job,status=resource['status'],resources=resource,verified=False,mechanism_pass=False,domain_pass=False)
        if valid and resource['status']=='complete':
            try:
                require(time.time()<b['start_unix']+1800,'Budget expired.')
                require(s['rows'][n].get('score_status')=='complete',s['rows'][n].get('error','Missing test scores.'))
                if job['case'] not in cache:cache[job['case']]=[cases.generate(job['case'],split,job['seed']) for split in ('train','validation','test')]
                arrays=cache[job['case']];cell=core.verify_run(job,s['rows'][n],adapter,arrays,classes)
                allX=np.vstack([a[0] for a in arrays]);bounds=np.column_stack((allX.min(axis=0),allX.max(axis=0))).tolist()
                result=read(folder/'result.json');cls=classes['single'] if job['method']=='independent' else classes['joint']
                audits=[cls.load(folder/name).audit_domain(input_bounds=bounds) for name in result['models']]
                require(audits==s['rows'][n]['extended_audit'] and bounds==s['rows'][n]['extended_bounds'],'Extended audit differs.')
                cell['extended_domain_pass']=all(a['status']=='enclosed' for a in audits);cell['extended_audits']=audits
                if job['method']=='candidate':
                    data=read(folder/'model.json')
                    source_path=folder.parent/f"{job['case']}_s{job['seed']}_original"/'model.json' if result['search_reused'] else folder/'searched_model.json'
                    source=read(source_path)
                    require(data==selector.derive(source,'guarded'),'Candidate export/source differs.')
                    require(data['post_selection']['chosen_within_bounds'],'Candidate validation guard failed.')
                    cell['source_search']=source_path.parent.name
            except Exception as e:
                cell.update(status='verification_failed',verified=False,error=repr(e));errors.append(core.run_name(job)+': '+repr(e))
        elif resource['status']!='complete':errors.append(core.run_name(job)+': '+resource['status'])
        cells.append(cell)
    comparisons={f'{a}_minus_{b}':core.compare(cells,a,b) for a,b in [('candidate','original'),('candidate','independent'),('original','independent')]}
    wall=time.time()-b['start_unix'] if b else 1801.;decision=decide(cells,comparisons['candidate_minus_original'],valid and not errors,wall<=1800)
    methods={method:method_summary(cells,method) for method in study.METHODS}
    for method,record in methods.items():record['standalone_wall_seconds']=record['wall_seconds']+(methods['original']['wall_seconds'] if method=='candidate' and not read(HERE/'ALGORITHM.json')['trig_seeds'] else 0.)
    verification=dict(status='passed' if decision['technical_pass'] else 'failed',verified_systems=sum(c.get('verified',False) for c in cells),expected_systems=18,
        errors=errors,cells=cells,statuses=dict(Counter(c['status'] for c in cells)),wall_seconds_from_freeze=wall,verification_seconds=time.time()-started,
        maximum_metric_difference=max((c.get('maximum_metric_delta',0.) for c in cells),default=0.),
        evidence_sha256={n:sha(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','ALGORITHM.json','DATA_MANIFEST.json') if (HERE/n).exists()})
    index={(c['job']['case'],c['job']['seed'],c['job']['method']):c for c in cells};pairs=[]
    for case in cases.CASES:
        for seed in cases.SEEDS:
            a,z=index[case,seed,'candidate'],index[case,seed,'original']
            pairs.append(dict(case=case,seed=seed,available=a.get('verified',False) and z.get('verified',False),
                identical=a.get('verified',False) and z.get('verified',False) and a['search_fingerprint']==z['search_fingerprint'] and a['frozen_pool_sha256']==z['frozen_pool_sha256']))
    summary=dict(methods=methods,comparisons=comparisons,recommendation=decision,algorithm=read(HERE/'ALGORITHM.json'),search_pairs=pairs)
    save(summary,verification)
    print(json.dumps(dict(status=verification['status'],verified_systems=verification['verified_systems'],errors=errors,recommendation=decision,comparisons={k:v['overall'] for k,v in comparisons.items()}),indent=2))
    # A correctly documented algorithm/domain failure is a finished experiment.
    return 0 if valid and wall<=1800 else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('verify',));p.parse_args();sys.exit(verify())
