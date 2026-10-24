import copy
import importlib.util
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from dimensr_mimo._vendor.expression import Node
from dimensr_mimo.features import Feature

directory = Path(__file__).resolve().parents[1] / "experiments/candidate_diagnosis_v1"
sys.path.insert(0, str(directory))
try:
    import candidate_diagnosis as d
    spec = importlib.util.spec_from_file_location("candidate_diagnosis_analysis", directory / "analyze.py")
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def record(keys=("x",), edge=1, inner=1., outer=1., eligible=True, index=0):
    return dict(keys=set(keys), edges=edge, aliases=[index], eligible=eligible,
                inner={"nrmse":[inner]}, outer={"nrmse":[outer]})


def saved(weights=(1.,), intercept=0.):
    return dict(features=[Feature(Node(f"x{i}")).to_dict() for i in range(len(weights))],
                coefficients=[list(weights)], intercepts=[intercept])


def test_exact_shared_node_caps_and_source_index_ties():
    pools = [[record(index=9), record(index=2)], [record()]]
    result = d.enumerate_systems(pools, dict(empty=2, shared=3, unbounded=None))
    assert result["empty"]["count"] == 0 and result["empty"]["inner"] is None
    assert result["shared"]["count"] == 2
    assert result["shared"]["inner"]["complexity"] == 3  # Not 4: x is shared.
    assert result["shared"]["inner"]["indices"] == [2, 0]


def test_finite_hindsight_never_filters_inner_selector():
    pool = [[record(inner=.1, outer=None, index=0), record(inner=.2, outer=.3, index=1)]]
    result = d.enumerate_systems(pool, dict(own=2))["own"]
    assert result["inner"]["indices"] == [0]
    assert result["inner"]["outer_status"] == "nonfinite" and result["inner"]["outer_nrmse"] is None
    assert result["hindsight"]["indices"] == [1]
    assert result["count"] == 2 and result["finite"] == 1


def test_filter_gap_and_joint_indivisibility():
    pool = [record(inner=.5, outer=.9), record(inner=.6, outer=.2, eligible=False, index=1)]
    result = d.enumerate_systems([pool], dict(own=2))["own"]
    assert result["hindsight"]["outer_nrmse"] == .2
    assert result["eligible_hindsight"]["outer_nrmse"] == .9
    joint = [record(index=0), record(index=1)]
    joint[0]["inner"]["nrmse"] = joint[0]["outer"]["nrmse"] = [.1, .9]
    joint[1]["inner"]["nrmse"] = joint[1]["outer"]["nrmse"] = [.9, .1]
    result = d.enumerate_systems([joint], dict(own=2), joint=True)["own"]
    assert result["count"] == 2 and result["hindsight"]["outer_nrmse"] == .5
    assert result["eligible_hindsight"] is None
    with pytest.raises(ValueError, match="indivisible"):
        d.enumerate_systems([joint, joint], dict(own=9), joint=True)


def test_active_dedup_preserves_aliases_and_input():
    pool = [saved((1., 0.)), saved((1.,)), saved((1., 0.))]
    before = copy.deepcopy(pool)
    assert len(d.deduplicate(pool)) == 2
    active = [d.active_head(c, 0) for c in pool]
    assert d.deduplicate(active)[0]["aliases"] == [0, 1, 2]
    assert pool == before


def test_rank_constant_ties_and_missing_outer():
    assert d.rank_diagnostics([1., 1.], [1., 2.], 0)["spearman"] is None
    assert d.rank_diagnostics([1.], [2.], 0)["selected_outer_rank_percentile"] is None
    r = d.rank_diagnostics([1., 2., 3.], [3., 1., None], 1)
    assert r["spearman"] == pytest.approx(-1.)
    assert r["selected_outer_rank_percentile"] == 0. and r["nonfinite_outer_count"] == 1


def test_fixed_evaluation_and_metric_failures_do_not_fit(monkeypatch):
    def no_fit(*args, **kwargs):
        raise AssertionError("Fitting is prohibited")
    monkeypatch.setattr(d.MIMOSymbolicRegressor, "fit", no_fit)
    X = np.array([[1., 0.], [2., 0.], [3., 0.]])
    split = dict(X_val=X, X_eval=X, Y_val=2*X[:, :1], Y_eval=2*X[:, :1], target_scale=np.array([2.]))
    c = saved((2., 0.)); before = copy.deepcopy(c)
    r = d.evaluate(c, split, 0, {})
    assert r["outer"]["nrmse"] == [0.]
    assert r["outer"]["r2"] == [1.] and r["complexity"] == 3
    assert d.evaluate(d.active_head(c, 0), split, 0, {})["complexity"] == 2
    assert c == before
    m = d.metrics(np.array([[1.], [1.]]), np.array([[math.inf], [1.]]), np.array([1.]))
    assert m == dict(mse=[None], nrmse=[None], r2=[None], finite=[False])


def test_freeze_checkpoint_tamper_and_no_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "HERE", tmp_path); monkeypatch.setattr(d, "ROOT", tmp_path)
    source = tmp_path / "source"; source.write_text("original")
    d.save(tmp_path / "FREEZE.json", {"files": {"source": d.sha(source)}})
    d.check_freeze()
    path = tmp_path / "cell.json"
    d.checkpoint(path, {"freeze_sha256": d.sha(tmp_path / "FREEZE.json")})
    d.load_checkpoint(path)
    with pytest.raises(FileExistsError):
        d.checkpoint(path, {})
    path.write_text("{}")
    with pytest.raises(ValueError, match="checkpoint"):
        d.load_checkpoint(path)
    source.write_text("changed")
    with pytest.raises(ValueError, match="Frozen input changed"):
        d.check_freeze()


def test_summary_does_not_replace_missing_with_zero():
    assert analysis.stats([1., None]) == dict(expected=2, finite=1, missing=1, mean=1., median=1., min=1., max=1.)
    assert analysis.stats([None])["mean"] is None


def test_exact_enumeration_matches_independent_set_reference():
    rng = np.random.default_rng(19)
    pools = [[record(keys=tuple(rng.choice(list("abcdef"), size=3, replace=False)),
                     edge=int(rng.integers(1, 4)), inner=float(rng.integers(1, 4)),
                     outer=None if i == 0 else float(rng.integers(1, 4)), eligible=i % 2 == 0, index=i)
              for i in range(5)] for _ in range(3)]
    original = copy.deepcopy(pools)
    caps = dict(own=9, common=6, unbounded=None)
    actual = d.enumerate_systems(pools, caps)
    for label, cap in caps.items():
        feasible = []
        for combo in product(*pools):
            k = len(set().union(*(r["keys"] for r in combo))) + sum(r["edges"] for r in combo)
            if cap is None or k <= cap:
                feasible.append((combo, k))
        assert actual[label]["count"] == len(feasible)
        for mode in ("inner", "hindsight", "eligible_hindsight"):
            options = []
            for combo, k in feasible:
                if mode != "inner" and any(r["outer"]["nrmse"][0] is None for r in combo):
                    continue
                if mode == "eligible_hindsight" and not all(r["eligible"] for r in combo):
                    continue
                part = "inner" if mode == "inner" else "outer"
                score = sum(r[part]["nrmse"][0] / 3 for r in combo)
                options.append((score, k, [r["aliases"][0] for r in combo]))
            best = min(options) if options else None
            found = actual[label][mode]
            if best:
                assert (found["complexity"], found["indices"]) == (best[1], best[2])
            else:
                assert found is None
    assert pools == original


def test_dag_constants_share_by_value_not_label():
    a = Feature(Node("c0"), (1.,))
    b = Feature(Node("c0"), (2.,))
    assert len(d.node_keys((a, a))) == 1
    assert len(d.node_keys((a, b))) == 2
