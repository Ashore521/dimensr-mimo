"""CSV command line interface for coupled symbolic regression."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import MIMOSymbolicRegressor, SearchConfig


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        names = next(reader, [])
        if not names or any(not name.strip() for name in names):
            raise ValueError("CSV needs a header with nonempty column names.")
        if len(set(names)) != len(names):
            raise ValueError("CSV column names must be unique.")
        rows = []
        for line, row in enumerate(reader, start=2):
            if len(row) != len(names):
                raise ValueError(f"CSV row {line} has {len(row)} columns; expected {len(names)}.")
            try:
                rows.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(f"CSV row {line} contains a nonnumeric value.") from exc
    if not rows:
        raise ValueError("CSV must contain at least one data row.")
    values = np.asarray(rows, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CSV values must be finite.")
    return names, values


def _columns(names, values, wanted):
    if (not isinstance(wanted, (list, tuple)) or not wanted
            or any(not isinstance(name, str) or not name for name in wanted)
            or len(set(wanted)) != len(wanted)):
        raise ValueError("Requested column names must be nonempty and unique.")
    missing = [name for name in wanted if name not in names]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}.")
    return values[:, [names.index(name) for name in wanted]]


def _read_json(path):
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Metadata and config JSON must be objects.")
    return value


def _check_output(path, inputs):
    output = Path(path).resolve()
    if any(output == Path(source).resolve() for source in inputs if source):
        raise ValueError("Output must differ from the input files.")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dimension-aware multi-output symbolic regression.")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="Fit a shared symbolic dictionary to a numeric CSV.")
    fit.add_argument("--data", required=True, help="Training CSV with a header.")
    fit.add_argument("--targets", nargs="+", required=True, help="Target columns, in output order.")
    fit.add_argument("--metadata", help="JSON with names, unit vectors and optional input_bounds.")
    fit.add_argument("--config", help="JSON containing SearchConfig fields.")
    fit.add_argument("--validation", help="Optional independent validation CSV.")
    fit.add_argument("--output", required=True, help="Destination model JSON.")
    predict = commands.add_parser("predict", help="Predict a numeric CSV, aligning input columns by name.")
    predict.add_argument("--model", required=True, help="Saved model JSON.")
    predict.add_argument("--data", required=True, help="Input CSV with a header.")
    predict.add_argument("--output", required=True, help="Destination predictions CSV.")
    args = parser.parse_args(argv)
    try:
        names, values = _read_csv(args.data)
        if args.command == "fit":
            output = _check_output(args.output, [args.data, args.validation, args.metadata, args.config])
            metadata = _read_json(args.metadata)
            allowed = {"feature_names", "target_names", "feature_units", "target_units", "coefficient_units", "input_bounds"}
            unknown = set(metadata) - allowed
            if unknown:
                raise ValueError(f"Unknown metadata fields: {', '.join(sorted(unknown))}.")
            if "target_names" in metadata and metadata["target_names"] != args.targets:
                raise ValueError("Metadata target_names must match --targets in the same order.")
            metadata["target_names"] = args.targets
            features = metadata.setdefault("feature_names", [name for name in names if name not in args.targets])
            if set(features) & set(args.targets):
                raise ValueError("Feature columns and target columns must be disjoint.")
            X = _columns(names, values, features)
            Y = _columns(names, values, args.targets)
            bounds = metadata.pop("input_bounds", None)
            config = SearchConfig(**_read_json(args.config))
            model = MIMOSymbolicRegressor(config=config, **metadata)
            validation = {}
            if args.validation:
                val_names, val_values = _read_csv(args.validation)
                validation = {"X_val": _columns(val_names, val_values, features),
                              "Y_val": _columns(val_names, val_values, args.targets)}
            model.fit(X, Y, input_bounds=bounds, **validation)
            output.parent.mkdir(parents=True, exist_ok=True)
            model.save(output)
            for name, expression in zip(model.target_names_, model.expressions_):
                print(f"{name} = {expression}")
            print(f"Saved model: {output}")
        else:
            output = _check_output(args.output, [args.data, args.model])
            model = MIMOSymbolicRegressor.load(args.model)
            prediction = model.predict(_columns(names, values, model.feature_names_))
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(model.target_names_)
                writer.writerows(prediction)
            print(f"Saved predictions: {output}")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0
