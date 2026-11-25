"""Export the final selection pool, its constrained front and the delivered model."""
import csv
import tempfile
from pathlib import Path

import numpy as np
from dimensr_mimo._single.pareto import first_front
from dimensr_mimo.evaluation import search_score
from dimensr_mimo.pareto import violations
from dimensr_mimo.portfolio_proposals import identity

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter, NullFormatter


def stage_points(pool, cfg, reference):
    points = {}
    for candidate in pool:
        key = identity(candidate)
        if key not in points:
            points[key] = dict(model_id=key, complexity=candidate.complexity,
                               training_objective=search_score(candidate.errors, cfg),
                               violation=violations(candidate.errors, reference, cfg),
                               **{f'train_nrmse_{j+1}': v for j,v in enumerate(candidate.errors)})
    rows = list(points.values())
    if rows:
        lowest = min(r['violation'] for r in rows)
        tier = [r for r in rows if r['violation'] == lowest]
        front = first_front([(r['training_objective'], r['complexity']) for r in tier])
        selected = {tier[i]['model_id'] for i in front}
        for row in rows:
            row['on_front'] = row['model_id'] in selected
    return rows


def export_pareto(pool, cfg, reference, folder, title, *, selected, delivered):
    rows = stage_points(pool, cfg, reference)
    selected_id, delivered_id = identity(selected), identity(delivered)
    if selected_id not in {r['model_id'] for r in rows}:
        raise ValueError('The selected model must belong to the final selection pool.')
    for row in rows:
        row.update(in_selection_pool=True, selected=row['model_id'] == selected_id,
                   delivered=row['model_id'] == delivered_id)
    feasible = [r for r in rows if r['violation'] == 0]
    front = [r for r in rows if r['on_front']]
    selection_count = len(rows)
    minimum_violation = min(r['violation'] for r in rows)
    if delivered_id not in {r['model_id'] for r in rows}:
        # A later delivery transformation is plotted without changing the
        # candidate pool or the front that was available to the selector.
        row = stage_points([delivered], cfg, reference)[0]
        row.update(on_front=False, in_selection_pool=False, selected=False, delivered=True)
        rows.append(row)
    label = {'mean':'各输出训练 NRMSE 的平均值', 'worst':'各输出训练 NRMSE 的最大值',
             'tolerance_mean_max':'训练误差与目标上限之比（均值与最大值的平均）'}[cfg.search_objective]
    folder.mkdir(parents=True, exist_ok=True)
    files = ('pareto.png', 'pareto.pdf', 'pareto.svg', 'candidates.csv')
    with tempfile.TemporaryDirectory(dir=folder) as tmp, mpl.rc_context({
            'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei','SimHei','Noto Sans CJK SC','DejaVu Sans'],
            'font.size':9, 'pdf.fonttype':42, 'svg.fonttype':'none', 'svg.hashsalt':'dimensr-nsga2',
            'axes.spines.top':False, 'axes.spines.right':False,
            'axes.linewidth':0.8, 'legend.frameon':False}):
        tmp = Path(tmp)
        with (tmp/'candidates.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
                                    ['model_id','complexity','training_objective','violation','on_front'])
            writer.writeheader()
            writer.writerows(rows)
        fig = Figure(figsize=(6.8,4.85), facecolor='white')
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0.155,0.15,0.80,0.58])
        fig.text(.155,.935,title,fontsize=13,weight='bold',color='#182F3B')
        ax.set_xlabel('模型混合复杂度',labelpad=7)
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#E0E6E8',linewidth=.55)
        for side in ('left','bottom'):
            ax.spines[side].set_color('#64757B')
        ax.tick_params(length=3,color='#64757B')
        def scatter(group, **style):
            if group:
                ax.scatter([r['complexity'] for r in group], [r['training_objective'] for r in group],
                           **style)
        scatter([r for r in feasible if not r['on_front']],s=11,c='#728C9F',alpha=.55,
                linewidths=0,label='其余可行模型',zorder=2)
        scatter([r for r in rows if r['in_selection_pool'] and r['violation'] > 0 and not r['on_front']],s=15,marker='x',
                c='#A96835',alpha=.7,linewidths=.65,label='违反保护约束',zorder=2)
        if front:
            xy = sorted({(r['complexity'],r['training_objective']) for r in front})
            ax.step(*zip(*xy),where='post',linewidth=1.1,c='#007E77',zorder=3)
            scatter(front,s=22,c='#007E77',marker='D',edgecolors='white',linewidths=.35,
                    label=('可行 Pareto 前沿' if feasible else '最小约束违反量层的前沿')+f'（{len(front)} 组）',
                    zorder=4)
            if selected_id != delivered_id:
                scatter([r for r in rows if r['selected']],s=75,c='#C87B22',marker='o',
                        edgecolors='white',linewidths=.6,label='筛选选中模型',zorder=5)
            scatter([r for r in rows if r['delivered']],s=120,c='#AB3E4C',marker='*',
                    edgecolors='white',linewidths=.5,
                    label='最终选中模型' if selected_id == delivered_id else '最终交付模型',zorder=6)
            values = np.array([r['training_objective'] for r in rows])
            if values.min() > 0 and values.max()/values.min() >= 8:
                ax.set_yscale('log')
                label += '（对数刻度）'
                ax.yaxis.set_minor_formatter(NullFormatter())
                # Intermediate labels keep the dense low-error region readable.
                if values.max()/values.min() < 100:
                    from matplotlib.ticker import LogLocator
                    ax.yaxis.set_major_locator(LogLocator(subs=(1.,2.,3.,5.)))
                    ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.legend(loc='lower left',bbox_to_anchor=(-.015,1.025),ncol=2,fontsize=8.3,
                      handletextpad=.25,columnspacing=1.15,borderaxespad=0)
            ax.margins(x=.035,y=.08)
        else:
            ax.text(.5,.5,'该阶段没有有效候选',ha='center',va='center',transform=ax.transAxes)
        ax.set_ylabel(label,labelpad=8)
        fig.savefig(tmp/'pareto.png',dpi=300)
        fig.savefig(tmp/'pareto.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(tmp/'pareto.svg',metadata={'Date':None})
        # The delivery receipt is written after all formats and data are published.
        for file in files:
            (tmp/file).replace(folder/file)
    return dict(candidates=selection_count, feasible=len(feasible), front_count=len(front),
                minimum_violation=minimum_violation, objective=cfg.search_objective,
                selected_model_id=selected_id, delivered_model_id=delivered_id,
                delivered_in_selection_pool=any(r['delivered'] and r['in_selection_pool'] for r in rows),
                files=list(files))
