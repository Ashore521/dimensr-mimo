"""Append human-readable progress in event order, independently of search decisions."""
import time

import numpy as np

from .stopping import metric_values


STAGES = {
    'initialization': '初始化', 'exploration': '表达式搜索', 'structural': '结构候选补充',
    'nsga2': 'NSGA-II 优选', 'polish': '候选精修', 'residual': '残差修正',
    'sharing': '共享结构搜索', 'restart': '多样性重启', 'portfolio': '多策略搜索',
    'exploration_epoch': '表达式搜索 · 本轮结束', 'gp': '表达式搜索 · 遗传演化',
    'neural': '表达式搜索 · 神经采样', 'random': '表达式搜索 · 随机采样',
    'policy_update': '表达式搜索 · 采样策略更新', 'structural_proposal': '结构候选补充 · 评价提案',
    'nsga2_generation': 'NSGA-II 优选', 'polish_candidate': '候选精修',
    'residual_round': '残差修正', 'sharing_round': '共享结构搜索',
    'stagnation_check': '停滞检查', 'training_target_reached': '首次达到训练目标',
    'search_complete': '搜索结束', 'admission': '候选入围', 'selection': '最终筛选',
    'capacity_growth': '表达式搜索 · 扩展容量', 'phase_budget': '阶段预算',
    'sharing_closure': '共享结构搜索', 'portfolio_finished': '多策略搜索结束',
    'enhanced_handoff': '前置搜索完成 · 移交候选', 'proposal_completed': '多策略搜索 · 提案完成',
    'proposal_reserved': '多策略搜索 · 开始评价提案',
}


class FitProgress:
    def __init__(self, names, stream, log, cfg, *, target=None, save_event=None):
        self.names, self.stream, self.log, self.cfg = names, stream, log, cfg
        self.target, self.save_event = target, save_event
        self.started = time.monotonic()
        self.best_key = self.best_r2 = self.best_complexity = None
        self.revision = 0
        self.last_numbered_event = None

    def bind(self, evaluator, variance):
        self.scales, self.variance = evaluator.scales, variance
        if self.target is not None:
            self.target = np.broadcast_to(self.target, variance.shape)
        evaluator.observe_candidate = self.observe
        evaluator.progress_display = self
        for candidate in getattr(evaluator, 'archive', evaluator.cache).values():
            self.observe(candidate, announce=False)

    def observe(self, candidate, *, announce=True):
        if candidate is None:
            return
        values = metric_values(candidate, 'r2', self.scales, self.variance)
        if not np.isfinite(values).all():
            return
        worst = float(values.min())
        hit = self.target is not None and (values >= self.target).all()
        rank = ((0, candidate.complexity, -worst) if hit else
                (1, -worst, candidate.error, candidate.complexity))
        if self.best_key is None or rank < self.best_key:
            first_hit = hit and (self.best_key is None or self.best_key[0] != 0)
            if self.best_key is None or rank[:2] != self.best_key[:2]:
                self.revision += 1
            self.best_key, self.best_r2, self.best_complexity = rank, values, candidate.complexity
            if first_hit and announce:
                self.write('首次达到训练目标', self.metrics(values, candidate.complexity))

    def metrics(self, values, complexity):
        if values is None:
            return '等待首个有效模型'
        names = self.names
        worst = int(np.argmin(values))
        # Show extra precision near the threshold so rounding cannot imply a hit.
        def number(i):
            value = values[i]
            if self.target is not None and value < self.target[i] and round(float(value), 3) >= self.target[i]:
                return str(float(value))
            return f'{value:.3f}'
        fields = [f'{name} {number(i)}' for i,name in enumerate(names)] if len(names) <= 3 else []
        if len(names) > 1:
            fields.append(f'最弱 {names[worst]} {number(worst)}')
        if self.target is not None:
            if np.all(self.target == self.target[0]):
                fields.append(f'目标 {self.target[0]:.3f}')
            fields.append(f'达标 {int((values >= self.target).sum())}/{len(names)}')
        fields.append(f'混合复杂度 {complexity:.2f}')
        return '训练 R²  ' + ' | '.join(fields)

    def write(self, title, detail):
        title = ' '.join(title.splitlines())
        detail = ' '.join(detail.splitlines())
        block = f'{title}\n{detail}\n\n'
        self.log.write(block)
        self.log.flush()
        print(block, end='', file=self.stream, flush=True)

    def __call__(self, event):
        if self.save_event:
            self.save_event(event)
        # This internal monitor may emit several improvements in a single batch.
        # The next chronological substep reports the best complete candidate.
        if event['phase'] != 'effective_training_progress':
            self.show(event)

    def iteration(self, event):
        phase = event['phase']
        if 'round_number' in event:
            return 'exploration', event['round_number'], '轮'
        if 'step' not in event or event.get('stop_reason'):
            return None
        step = event['step']
        if phase in ('nsga2','nsga2_generation'):
            return 'nsga2', step + int('generation_completed' not in event), '代'
        if phase in ('residual_round','sharing_round'):
            return phase, step, '轮'
        if phase == 'gp':
            return 'exploration', step//self.cfg.gp_generations+1, '轮'
        if phase in ('policy_update','neural','random','exploration_epoch'):
            return 'exploration', step + int('epoch_completed' not in event), '轮'
        return None

    def show(self, event):
        phase = event['phase']
        title = [STAGES.get(phase, phase)]
        if event.get('action'):
            title.append(event['action'])
        if 'cycle' in event:
            title.append(f"第 {event['cycle']+1} 周期")
        iteration = self.iteration(event)
        if iteration:
            series, index, unit = iteration
            position = f'第 {index} {unit}'
            if phase == 'gp':
                generation = event.get('generation') or event['step'] % self.cfg.gp_generations+1
                position += f' / 遗传演化 {generation}/{self.cfg.gp_generations} 代'
            title.append(position)
        if 'position' in event:
            title.append(f"第 {event['position']} 组")
        if event.get('stop_reason'):
            title.append('结束')
        if 'evaluations' in event:
            title.append(f"累计评价 {event['evaluations']} 次")
        seconds = int(event.get('elapsed_seconds', time.monotonic()-self.started))
        title.append(f'已用时 {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}')
        detail = self.metrics(self.best_r2, self.best_complexity)
        if iteration:
            group = (event.get('cycle', 0), series)
            if self.last_numbered_event is not None:
                last_group, revision, last_index = self.last_numbered_event
                if group == last_group and self.revision == revision and index > last_index:
                    detail += f' | 连续 {index-last_index} {unit}未提升'
                    index = last_index
            self.last_numbered_event = group, self.revision, index
        if 'note' in event:
            detail += ' | ' + event['note']
        for key, label in (('valid_offspring','有效子代'), ('new_evaluations','新增评价'), ('stale_checks','连续停滞检查'),
                           ('producer','策略'), ('operation','操作'), ('stop_reason','结束原因')):
            if key in event:
                detail += f' | {label} {event[key]}'
        self.write(' | '.join(title), detail)
