import importlib.util
from pathlib import Path
import sys


directory = Path(__file__).resolve().parents[1] / "experiments/budget_capacity_v1"
sys.path.insert(0, str(directory))
try:
    spec = importlib.util.spec_from_file_location("capacity_analysis", directory / "analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def test_missing_or_failed_cells_do_not_generate_partial_means():
    rows = [dict(status="ok", outer_nrmse=1.)] * 5
    assert "metrics" not in analysis.aggregate(rows)
    assert "metrics" not in analysis.aggregate(rows + [dict(status="failed")])
    assert analysis.aggregate(rows + [dict(status="ok", outer_nrmse=4.)])["metrics"]["outer_nrmse"]["mean"] == 1.5


def test_pairs_match_fold_seed_cap_and_preserve_failures():
    base = dict(dataset="example", fold=0, seed=0, cap=40, status="ok", complexity=10)
    rows = [dict(base, arm="a", budget=60, outer_nrmse=.5),
            dict(base, arm="a", budget=20, outer_nrmse=1.),
            dict(base, seed=1, arm="a", budget=20, outer_nrmse=.2)]
    paired = analysis.paired_rows(rows, ("a", 60), ("a", 20), "budget")
    assert paired[0]["delta_outer_nrmse"] == -.5
    assert paired[0]["delta_complexity"] == 0
    assert paired[1]["status"] == "incomplete"
    assert "delta_outer_nrmse" not in paired[1]
