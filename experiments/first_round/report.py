"""Post-freeze descriptive report, including every failure and per-output score."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    scores = json.loads((HERE / 'SCORES.json').read_text())
    rows = scores['rows']
    methods = ('full', 'independent', 'no_neural', 'no_gp', 'no_sharing')
    names = dict(full='完整联合 MIMO', independent='逐输出独立 DimenSR', no_neural='无神经策略',
                 no_gp='无 GP 演化', no_sharing='无共享')
    metric_rows = []
    for r in rows:
        if 'nrmse' in r:
            for j in range(len(r['nrmse'])):
                metric_rows.append(dict(**r['job'], output=j, rmse=r['rmse'][j], nrmse=r['nrmse'][j],
                                        r2=r['r2'][j], formula=r['formulas'][j]))
    with (HERE / 'metrics_per_output.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0])); w.writeheader(); w.writerows(metric_rows)
    summary = {}
    for method in methods:
        group = [r for r in rows if r['job']['method'] == method]
        good = [r for r in group if 'nrmse' in r]
        summary[method] = dict(scheduled=len(group), completed=len(good),
            mean_nrmse=float(np.mean([r['mean_nrmse'] for r in good])) if good else None,
            mean_complexity=float(np.mean([r['complexity'] for r in good])) if good else None,
            wall_seconds=sum(r['resources'].get('wall_seconds', 0) for r in group),
            cpu_seconds=sum(r['resources'].get('process_tree_cpu_seconds', 0) for r in group),
            peak_mib=max(r['resources'].get('process_tree_peak_rss_bytes', 0) for r in group) / 2**20,
            failures=Counter(r['resources']['status'] for r in group if r['resources']['status'] != 'complete'))
    pair = []
    for full in [r for r in rows if r['job']['method'] == 'full']:
        baseline = next(r for r in rows if r['job']['case'] == full['job']['case'] and r['job']['seed'] == full['job']['seed'] and r['job']['method'] == 'independent')
        if 'nrmse' in full and 'nrmse' in baseline:
            pair.append(dict(case=full['job']['case'], seed=full['job']['seed'],
                mean_difference=full['mean_nrmse'] - baseline['mean_nrmse'],
                output_differences=(np.array(full['nrmse']) - np.array(baseline['nrmse'])).tolist(),
                harmful_outputs=int(np.sum(np.array(full['nrmse']) > np.array(baseline['nrmse']) * 1.10 + .02))))
    summary['paired_full_minus_independent'] = pair
    (HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# 首轮合成开发验证报告', '',
        '本轮验证完整工程机制和小规模性能，不能据此宣称普遍优于独立 DimenSR。任务公式预先已知，属于合成开发证据。', '',
        f"共安排 {len(rows)} 个系统，完成 {sum('nrmse' in r for r in rows)} 个；预检至统一评分墙钟 {scores['total_wall_seconds']/60:.2f} 分钟，上限 120 分钟。", '',
        '| 方法 | 完成/安排 | 平均 NRMSE | 平均复杂度 | 实际墙钟秒 | 进程树 CPU 秒 | 峰值 MiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        v = summary[m]
        lines.append(f"| {names[m]} | {v['completed']}/{v['scheduled']} | {v['mean_nrmse']:.6g} | {v['mean_complexity']:.3f} | {v['wall_seconds']:.1f} | {v['cpu_seconds']:.1f} | {v['peak_mib']:.1f} |")
    lines += ['', '同一系统的时间上限统一为 60 秒；独立方法两个输出各最多 30 秒。固定搜索代数允许提前结束，实际资源并不完全相同，因此不作严格等实际 CPU 计算量的优越性断言。拟合和评价属于嵌套计时，不能与阶段时间重复求和。', '',
        '联合复杂度按共享 DAG 唯一非参数节点+唯一参数、算子种类、联合局部 Rademacher 计算；独立方法按两个原版树的复杂度相加。两者对应各自实际表示，数值并非跨表示绝对能力尺度。', '',
        '## 逐任务结果', '', '| 任务 | 方法 | 平均 NRMSE（两种子） | 最大输出 NRMSE |', '|---|---|---:|---:|']
    for case in dict.fromkeys(r['job']['case'] for r in rows):
        for m in methods:
            good = [r for r in rows if r['job']['case'] == case and r['job']['method'] == m and 'nrmse' in r]
            if good:
                lines.append(f"| {case} | {names[m]} | {np.mean([r['mean_nrmse'] for r in good]):.6g} | {max(r['max_nrmse'] for r in good):.6g} |")
    if pair:
        wins = sum(p['mean_difference'] < -1e-8 for p in pair)
        losses = sum(p['mean_difference'] > 1e-8 for p in pair)
        lines += ['', f"14 个预定配对中可比较 {len(pair)} 个：联合方法平均误差降低 {wins} 个、升高 {losses} 个、差异在 1e-8 内 {len(pair)-wins-losses} 个。按独立方法的逐输出误差×1.10+0.02 作描述性比较，共 {sum(p['harmful_outputs'] for p in pair)} 个输出明显退化。此比较不是训练时冻结的初始化参照约束。"]
    failures = [r for r in rows if r['resources']['status'] != 'complete']
    lines += ['', '## 失败、域与保存加载', '']
    lines += [f"- {r['job']}: {r['resources']['status']}" for r in failures] or ['没有被删除的失败单元；本次全部完成。']
    good = [r for r in rows if 'nrmse' in r]
    lines += ['', f"保存前后训练/验证预测哈希一致 {sum(r['saved_original_predictions_equal'] for r in good)}/{len(good)}；域审计一致 {sum(r['saved_original_audits_equal'] for r in good)}/{len(good)}。",
        f"最终声明域全部 enclosed 的系统 {sum(all(a['status']=='enclosed' for a in r['audit']) for r in good)}/{len(good)}。enclosed 仅为浮点数值筛查，不是全域证明。",
        f"完整联合方法满足冻结验证退化约束 {sum(r.get('validation_feasible',False) for r in good if r['job']['method']=='full')}/14。不满足者在结果中明确记录最小违反量回退。", '',
        '## 机制与局限', '',
        '逐单元机制和独立重算检查见 VERIFICATION.json；原始逐阶段记录保存在各单元 events.jsonl。保存的是完整图引用和参数编号，显示公式不决定数值语义。', '',
        '无神经、无 GP、无共享是固定消融。无 GP 变体按设计没有演化 NSGA-II，不能把静态候选排序称为该阶段已执行。量纲、区间、Rademacher、尺寸多样性开关均有单元级可运行验证，但本轮未对这四项进行完整性能矩阵消融。', '',
        '本轮只有七种合成任务和两个种子，没有新真实数据盲评，没有统计功效支持的泛化优势结论。局部 Rademacher 只描述固定拟合结构的联合参数切空间；神经引用槽数与每输出展开大小受限。共享是否被最终采用由训练/验证决定，不强制共享。', '',
        '所有模型在 ALL_MODELS_FROZEN.json 后统一生成测试标签并评分；FREEZE.json 与独立核验保留实现/配置/模型哈希，未依据测试误差重训或修改。']
    (HERE / 'REPORT.zh-CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
