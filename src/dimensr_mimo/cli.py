"""CSV training and prediction without executable model serialization."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from . import DimenSRMIMORegressor, SearchConfig
from .progress import FitProgress
from .stopping import metric_values


def columns(path, names):
    data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8-sig', ndmin=1)
    missing = set(names) - set(data.dtype.names or ())
    if missing:
        raise ValueError('Missing CSV columns: ' + ', '.join(sorted(missing)))
    return np.column_stack([data[n] for n in names])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    predict = commands.add_parser('predict')
    for p in (fit, predict):
        p.add_argument('--data', required=True)
        p.add_argument('--features', required=True)
        p.add_argument('--output', required=True)
    fit.add_argument('--targets', required=True)
    fit.add_argument('--validation')
    fit.add_argument('--config')
    fit.add_argument('--metadata')
    predict.add_argument('--model', required=True)
    args = parser.parse_args()
    names = args.features.split(',')
    X = columns(args.data, names)
    if args.command == 'fit':
        cfg = SearchConfig(**json.loads(Path(args.config).read_text(encoding='utf-8-sig'))) if args.config else SearchConfig()
        metadata = json.loads(Path(args.metadata).read_text(encoding='utf-8-sig')) if args.metadata else {}
        bounds = metadata.pop('input_bounds', None)
        model = DimenSRMIMORegressor(cfg, feature_names=names, **metadata)
        targets = args.targets.split(',')
        val = dict(X_val=columns(args.validation, names), Y_val=columns(args.validation, targets)) if args.validation else {}
        target = cfg.stop_target if cfg.stop_metric == 'r2' else None
        log_path = Path(args.output).with_suffix('.progress.log')
        if log_path.resolve() in {Path(name).resolve() for name in (
                args.data, args.validation, args.config, args.metadata, args.output) if name}:
            parser.error('The progress log path conflicts with an input or output file.')
        with log_path.open('a', encoding='utf-8') as log:
            progress = FitProgress(targets, sys.stderr, log, cfg, target=target)
            try:
                model.fit(X, columns(args.data, targets), input_bounds=bounds, callback=progress, **val)
                model.save(args.output)
            except (Exception, KeyboardInterrupt) as exc:
                progress.write('拟合中断', f'{type(exc).__name__} {exc}')
                raise
            progress.write(f'拟合完成 | 模型 {args.output}',
                           progress.metrics(metric_values(model.best_, 'r2', model.scales_, progress.variance),
                                            model.best_.complexity))
        print(json.dumps(dict(formulas=model.formulas_, diagnostics=model.diagnostics_), ensure_ascii=False))
    else:
        model = DimenSRMIMORegressor.load(args.model)
        if names != model.feature_names_:
            raise ValueError('CSV feature order must match the saved model.')
        np.savetxt(args.output, model.predict(X), delimiter=',', comments='',
                   header=','.join(f'y{i}' for i in range(model.n_outputs_)))


if __name__ == '__main__':
    main()
