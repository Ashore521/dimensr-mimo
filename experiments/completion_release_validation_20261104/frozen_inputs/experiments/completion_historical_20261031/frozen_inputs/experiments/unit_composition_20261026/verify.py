"""Independent reload verification and full-denominator 2x2 reporting."""
import json
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.typed_coverage_20261026 import cases
from experiments.unit_composition_20261026 import study
module=importlib.util.spec_from_file_location('_unit_composition_verification',HERE.parent/'typed_coverage_20261026/verify.py')
prior=importlib.util.module_from_spec(module);module.loader.exec_module(prior)
prior.HERE=HERE;prior.study=study
read,write,sha=study.read,study.write,study.sha
require=prior.require


def summarize(cells):
    methods={};comparisons={}
    index={(c['job']['case'],c['job']['seed'],c['job']['method']):c for c in cells}
    for method in study.METHODS:
        rows=[c for c in cells if c['job']['method']==method];valid=[c for c in rows if c.get('verified')]
        methods[method]=dict(expected=12,verified=len(valid),test_accepted=sum(c.get('test_accepted',False) for c in valid),
            development_accepted=sum(c.get('development_accepted',False) for c in valid),
            both_accepted=sum(c.get('test_accepted',False) and c.get('development_accepted',False) for c in valid),
            mean_nrmse=float(np.mean([c['mean_nrmse'] for c in valid])) if len(valid)==12 else None,
            feasible_pool_systems=sum(c.get('pool_feasible',0)>0 for c in valid))
        if method=='balanced':continue
        pairs=[]
        for case in cases.CASES:
            for seed in cases.SEEDS:
                a,b=index[case,seed,method],index[case,seed,'balanced'];p=dict(case=case,seed=seed,available=a.get('verified',False) and b.get('verified',False))
                if p['available']:
                    delta=np.asarray(a['nrmse'])-b['nrmse'];bound=np.asarray(b['nrmse'])*1.1+1e-8
                    p.update(mean_difference=float(delta.mean()),differences=delta.tolist(),protected=(np.asarray(a['nrmse'])<=bound).tolist(),
                        treatment_nrmse=a['nrmse'],control_nrmse=b['nrmse'],treatment_accepted=a['test_accepted'],control_accepted=b['test_accepted'])
                pairs.append(p)
        comparisons[method]=dict(pairs=pairs,protected_outputs=sum(sum(p.get('protected',[])) for p in pairs),expected_outputs=24,
            improved=sum(p.get('mean_difference',0)<-1e-8 for p in pairs),degraded=sum(p.get('mean_difference',0)>1e-8 for p in pairs))
    return dict(methods=methods,comparisons=comparisons,production_default_changed=False,
        scope='2x2 known-task development regression; no new/blind test claim.')


def verify():
    if (HERE/'VERIFICATION.json').exists():raise RuntimeError('Already verified.')
    f,m,s=[read(HERE/n) for n in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json')]
    study.verify_files(f['files']);study.verify_files(m['files'],HERE)
    require(f['jobs']==study.matrix()==[r['job'] for r in s['rows']] and len(f['jobs'])==48,'Matrix differs.')
    require(f['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=m['frozen_unix']<=s['scored_unix'],'Ordering failed.')
    require(m['freeze_sha256']==sha(HERE/'FREEZE.json') and s['all_models_freeze_sha256']==sha(HERE/'ALL_MODELS_FROZEN.json'),'Seal chain failed.')
    require(not f['test_labels_accessed'] and not m['test_labels_accessed'],'New model selection used test scores.')
    cells=[];failures=[];replication=[]
    for job,score in zip(f['jobs'],s['rows']):
        folder=HERE/'runs'/study.name(job);cell=dict(job=job,status=score['status'],verified=False)
        try:
            require(time.time()<study.budget(),'Verification budget expired.')
            seal=read(folder/'CELL_FROZEN.json');study.verify_files(seal['files'],folder)
            require(read(folder/'job.json')==job and read(folder/'config.json')==f['config'],'Cell inputs differ.')
            require(f['frozen_unix']<=seal['frozen_unix']<=m['frozen_unix'],'Cell order failed.')
            require(score['status']=='complete','Incomplete cell: '+score['status'])
            require(read(folder/'result.json')['job']==job,'Result arm differs.')
            cell=prior.verify_cell(job,score)
            if job['method']=='balanced':
                before=read(HERE.parent/'typed_coverage_20261026/runs'/study.name(job)/'searched_model.json');after=read(folder/'searched_model.json')
                replication.append(dict(case=job['case'],seed=job['seed'],same_pool=before['diagnostics']['frozen_candidate_sha256']==after['diagnostics']['frozen_candidate_sha256']))
        except Exception as e:cell.update(status='verification_failed' if score['status']=='complete' else score['status'],error=repr(e));failures.append(study.name(job)+': '+repr(e))
        cells.append(cell)
    technical=len(cells)==48 and all(c.get('verified') and c.get('mechanism_pass') and c.get('domain_pass') for c in cells)
    summary=summarize(cells);summary['control_replication']=replication
    report=dict(status='passed' if technical else 'failed',technical_pass=technical,verified_systems=sum(c.get('verified',False) for c in cells),expected_systems=48,
        cells=cells,failures=failures,wall_seconds=time.time()-f['budget']['start_unix'],freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'))
    write(HERE/'VERIFICATION.json',report);write(HERE/'SUMMARY.json',summary)
    lines=['# 种子类别对齐与量纲组合：2×2开发回归','','## Material Passport','',summary['scope'],'',
        f'48系统，核验{report["verified_systems"]}/48，技术门槛：{technical}，用时{report["wall_seconds"]:.2f}秒。','',
        '|方法|核验|开发达标|测试达标|两者达标|池内可行|平均测试NRMSE|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in summary['methods'].items():lines.append(f'|{k}|{v["verified"]}/12|{v["development_accepted"]}|{v["test_accepted"]}|{v["both_accepted"]}|{v["feasible_pool_systems"]}|{v["mean_nrmse"]}|')
    lines+=['','全部阈值沿用第六轮，测试数据为已经观察过的开发回归样本；结果不能证明新任务泛化。',
        '两项干预分开和组合报告。保留神经策略、GP、NSGA-II、局部Rademacher及逐输出约束；最终guarded选择器未变。',
        '每单元提供完整原图、参数单位、共享结构、适用域和复算交付；enclosed仍仅为数值筛查，生产默认未变。']
    for k,v in summary['comparisons'].items():lines.append(f'\n{k}相对balanced：改善{v["improved"]}/12，退化{v["degraded"]}/12，逐输出保护{v["protected_outputs"]}/24。')
    if failures:lines+=['','失败：','']+['- '+x for x in failures]
    (HERE/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],verified=report['verified_systems'],methods=summary['methods'],failures=failures)),flush=True)


if __name__=='__main__':verify()
