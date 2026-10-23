import csv
import json

import numpy as np
import pytest

from dimensr_mimo import MIMOSymbolicRegressor
from dimensr_mimo.cli import main


def _csv(path, names, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(values)


def test_fit_predict_aligns_columns_and_keeps_target_names(tmp_path, capsys):
    rng = np.random.default_rng(7)
    X = rng.uniform(-1, 1, size=(36, 2))
    Y = np.column_stack((2 * X[:, 0] + X[:, 1], -X[:, 0] + 3 * X[:, 1]))
    data = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    config = tmp_path / "config.json"
    metadata = tmp_path / "metadata.json"
    model_path = tmp_path / "model.json"
    _csv(data, ["a", "b", "u", "v"], np.column_stack((X[:28], Y[:28])))
    _csv(validation, ["v", "b", "u", "a"], np.column_stack((Y[28:, 1], X[28:, 1], Y[28:, 0], X[28:, 0])))
    config.write_text(json.dumps({"generations": 0, "max_features": 24, "max_terms": 2,
                                  "polish_max_nfev": 0, "random_state": 1}), encoding="utf-8")
    metadata.write_text(json.dumps({"feature_names": ["a", "b"], "target_names": ["u", "v"],
                                    "input_bounds": [[-1, 1], [-1, 1]]}), encoding="utf-8")
    assert main(["fit", "--data", str(data), "--targets", "u", "v",
                 "--validation", str(validation), "--config", str(config),
                 "--metadata", str(metadata), "--output", str(model_path)]) == 0
    prediction_data = tmp_path / "inputs.csv"
    output = tmp_path / "predictions.csv"
    _csv(prediction_data, ["b", "a"], X[:, ::-1])
    assert main(["predict", "--model", str(model_path), "--data", str(prediction_data),
                 "--output", str(output)]) == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["u", "v"]
    expected = MIMOSymbolicRegressor.load(model_path).predict(X)
    np.testing.assert_allclose(np.asarray(rows[1:], dtype=float), expected)
    np.testing.assert_allclose(expected, Y, atol=1e-8)
    _csv(prediction_data, ["a"], X[:, :1])
    with pytest.raises(SystemExit):
        main(["predict", "--model", str(model_path), "--data", str(prediction_data),
              "--output", str(output)])
    assert "missing columns: b" in capsys.readouterr().err


@pytest.mark.parametrize("names, rows, message", [
    (["a", "a", "u"], [[1, 2, 3]], "unique"),
    (["a", "u"], [[float("nan"), 2]], "finite"),
    (["a", "u"], [[1]], "columns"),
    (["a", "u"], [["text", 2]], "nonnumeric"),
    (["a", "u"], [], "at least one"),
])
def test_rejects_malformed_csv(tmp_path, capsys, names, rows, message):
    path = tmp_path / "bad.csv"
    _csv(path, names, rows)
    with pytest.raises(SystemExit) as exc:
        main(["fit", "--data", str(path), "--targets", "u", "--output", str(tmp_path / "model.json")])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_rejects_missing_target_and_input_overwrite(tmp_path, capsys):
    data = tmp_path / "data.csv"
    _csv(data, ["a", "u"], [[1, 2], [2, 4]])
    for targets, output, message in [(["absent"], tmp_path / "model.json", "missing columns"),
                                     (["u"], data, "differ from")]:
        with pytest.raises(SystemExit):
            main(["fit", "--data", str(data), "--targets", *targets, "--output", str(output)])
        assert message in capsys.readouterr().err
    assert data.read_text(encoding="utf-8").startswith("a,u")
