"""Fit the training-only engineering workflow to a named-column CSV."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import AffineCoordinates, DimenSRMIMORegressor, PhysicalModel, SearchConfig, SearchStopped
from ._single.units import ZERO, encode, unit
from .cli import columns
from .compression import compression_candidate
from .pareto_plot import export_pareto
from .progress import FitProgress
from .ranked_flow import RankedFlowSearch
from .regressor import arrays
from .stopping import metric_values, select_training


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--targets', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', help='JSON overrides for the engineering search preset')
    parser.add_argument('--metadata', help='Optional units, physical input bounds and affine coordinates')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    names, outputs = args.features.split(','), args.targets.split(',')
    x, y = arrays(columns(args.data, names), columns(args.data, outputs))
    if len(x) < 5 or len(set(names + outputs)) != len(names + outputs):
        parser.error('Provide at least five training rows and distinct feature and target columns.')
    options = read(Path(__file__).with_name('engineering_config.json'))
    options['n_cpus'] = min(options['n_cpus'], os.cpu_count() or 1)
    if args.config:
        options.update(read(args.config))
    cfg = SearchConfig(**options)
    cfg.validate()
    if (cfg.search_schedule != 'stagnation' or cfg.stop_metric is not None or cfg.final_selection != 'pareto'
            or not cfg.use_units or not cfg.interval_checks):
        parser.error('This workflow uses stagnation scheduling, unit and interval checks, and its own R² target.')
    for per_output in (cfg.fixed_training_reference, cfg.compression_r2_drop):
        if isinstance(per_output, list) and len(per_output) != len(outputs):
            parser.error('Per-output settings must match the number of target columns.')
    meta = read(args.metadata) if args.metadata else {}
    coordinates = (AffineCoordinates(**meta['coordinates']) if 'coordinates' in meta else
                   AffineCoordinates.identity(len(names), len(outputs)))
    features = tuple(map(unit, meta.get('feature_units', [ZERO] * len(names))))
    targets = tuple(map(unit, meta.get('target_units', [ZERO] * len(outputs))))
    constants = tuple(dict.fromkeys(map(unit, meta.get('constant_units', [ZERO]))))
    if len(features) != len(names) or len(targets) != len(outputs):
        parser.error('Unit counts must match the feature and target columns.')
    X, Y = coordinates.inputs(x), coordinates.outputs(y)
    if 'input_bounds' in meta:
        physical_bounds = meta['input_bounds']
    else:
        lo, hi = x.min(axis=0), x.max(axis=0)
        constant = lo == hi
        lo[constant] = np.nextafter(lo[constant], -np.inf)
        hi[constant] = np.nextafter(hi[constant], np.inf)
        physical_bounds = np.column_stack((lo, hi)).tolist()
    bounds = coordinates.input_bounds(physical_bounds)
    folder = args.output_dir
    checkpoint = folder/'search.sqlite'
    if args.resume != checkpoint.exists() or (not args.resume and folder.exists() and any(folder.iterdir())):
        parser.error('Choose a new or empty output directory, or use --resume with its existing checkpoint.')
    if (folder/'results.json').exists():
        parser.error('This fit is already complete.')
    folder.mkdir(parents=True, exist_ok=True)
    binding = dict(features=names, targets=outputs, metadata=meta)
    if args.resume and read(folder/'columns.json') != binding:
        parser.error('The columns or coordinate metadata differ from the checkpoint.')
    if not args.resume:
        save(folder/'columns.json', binding)
    with (folder/'progress.log').open('a', encoding='utf-8') as log:
        progress = FitProgress(outputs, sys.stderr, log, cfg, target=.90)
        if args.resume:
            progress.write('恢复拟合', '从已保存的检查点接续')
        owner = DimenSRMIMORegressor(cfg, feature_names=names)
        search = RankedFlowSearch(owner, cfg, X, Y, X[:2], np.zeros_like(Y[:2]), features, targets,
                                  constants, (), bounds, None, progress, checkpoint, args.resume,
                                  (folder/'STOP_REQUESTED').exists)

        def deliver(completed):
            pool = list(completed.delivery_candidates())
            e = completed.evaluator
            reference = completed.s['reference']
            completed.display_event('selection', action='开始', note=f'候选 {len(pool)} 组')
            parent, selection = select_training(pool, e.scales, completed.variance, target=.90)
            progress.write(f"最终筛选完成 | 候选 {len(pool)} 组 | 达标 {selection['target_hits']} 组",
                           progress.metrics(np.asarray(selection['values']), parent.complexity))
            child, compression = None, None
            if cfg.compression_r2_drop is not None:
                child, compression = compression_candidate([parent], e, features, targets, constants, cfg, reference)
            model = parent if child is None else child
            r2 = metric_values(model, 'r2', e.scales, completed.variance)
            source = dict(system=model.system.to_dict(), constants=model.constants.tolist(), feature_names=names,
                          feature_units=list(map(encode, features)), target_units=list(map(encode, targets)),
                          constant_units=list(map(encode, constants)), output_bounds=cfg.output_bounds)
            physical = PhysicalModel(source, coordinates, physical_bounds)
            np.testing.assert_allclose(physical.system.evaluate(x, physical.constants),
                                       coordinates.restore_outputs(model.system.evaluate(X, model.constants)),
                                       rtol=1e-10, atol=1e-10)
            save(folder/'model.json', physical.to_dict())
            plot = export_pareto(pool, cfg, reference, folder/'pareto', '最终模型筛选', selected=parent, delivered=model)
            save(folder/'results.json', dict(output_names=outputs, training_r2=r2.tolist(), complexity=model.complexity,
                 formulas=physical.formulas, selection_before_compression=selection, compression=compression,
                 training_admission=completed.s.get('admission_reports', []), pareto=plot, evaluations=e.calls))
            progress.write(f'拟合完成 | 累计评价 {e.calls} 次 | Pareto 图 {folder/"pareto"/"pareto.png"}',
                           progress.metrics(r2, model.complexity))

        try:
            search.run(deliver)
        except SearchStopped:
            progress.write('拟合已暂停', f'检查点 {checkpoint}')
        except (Exception, KeyboardInterrupt) as exc:
            progress.write('拟合中断', f'{type(exc).__name__} {exc}')
            raise


if __name__ == '__main__':
    main()
