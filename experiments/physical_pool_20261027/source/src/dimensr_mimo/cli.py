"""CSV training and prediction without executable model serialization."""
import argparse
import json
from pathlib import Path
import numpy as np
from . import DimenSRMIMORegressor, SearchConfig


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
        val = dict(X_val=columns(args.validation, names), Y_val=columns(args.validation, args.targets.split(','))) if args.validation else {}
        model.fit(X, columns(args.data, args.targets.split(',')), input_bounds=bounds, **val)
        model.save(args.output)
        print(json.dumps(dict(formulas=model.formulas_, diagnostics=model.diagnostics_), ensure_ascii=False))
    else:
        model = DimenSRMIMORegressor.load(args.model)
        if names != model.feature_names_:
            raise ValueError('CSV feature order must match the saved model.')
        np.savetxt(args.output, model.predict(X), delimiter=',', comments='',
                   header=','.join(f'y{i}' for i in range(model.n_outputs_)))


if __name__ == '__main__':
    main()
