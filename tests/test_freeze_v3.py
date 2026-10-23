import importlib.util
from pathlib import Path
import sys

import pytest


directory = Path(__file__).resolve().parents[1] / "experiments/real_blind_v3"
spec = importlib.util.spec_from_file_location("freeze_v3", directory / "holdout.py")
holdout = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(directory))
try:
    spec.loader.exec_module(holdout)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("changed, message", [
    ("model", "Model changed after training"),
    ("selection", "Selection changed after training"),
    ("manifest", "Training data provenance changed"),
])
def test_freeze_rejects_post_training_changes(tmp_path, monkeypatch, changed, message):
    monkeypatch.setattr(holdout, "ROOT", tmp_path)
    study = tmp_path / "study"
    monkeypatch.setattr(holdout, "HERE", study)
    (study / "new_data").mkdir(parents=True)
    manifest = study / "new_data/manifest.json"
    holdout.write_json(manifest, {"datasets": {"gas_turbine": {"trainval_sha256": "a" * 64}}})
    selection = study / "selection.json"
    holdout.write_json(selection, {"selected_method": "joint", "revision": 1})
    runs = tmp_path / "runs"
    run = runs / "gas_turbine_joint_s0"
    run.mkdir(parents=True)
    model = run / "model_0.json"
    holdout.write_json(model, {"synthetic": "trained"})
    holdout.write_json(run / "result.json", {
        "dataset": "gas_turbine", "method": "joint", "seed": 0, "status": "ok",
        "models": [model.name], "model_sha256": {model.name: holdout.sha(model)},
        "data_sha256": "a" * 64,
        "provenance": {
            "selection": {"sha256": holdout.sha(selection), "selected_method": "joint"},
            "source_directory": str(tmp_path / "snapshot"),
            "source_sha256": {}, "study_sha256": {},
        },
    })

    if changed == "model":
        holdout.write_json(model, {"synthetic": "modified"})
    elif changed == "selection":
        holdout.write_json(selection, {"selected_method": "joint", "revision": 2})
    else:
        holdout.write_json(manifest, {"datasets": {"gas_turbine": {"trainval_sha256": "b" * 64}}})

    output = tmp_path / "FREEZE.json"
    with pytest.raises(ValueError, match=message):
        holdout.freeze(runs, output, selection)
    assert not output.exists()
