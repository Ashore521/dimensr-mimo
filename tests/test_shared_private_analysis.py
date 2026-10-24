import importlib.util
from pathlib import Path
import sys

import pytest

directory=Path(__file__).resolve().parents[1]/"experiments/shared_private_v1"
sys.path.insert(0,str(directory))
try:
    spec=importlib.util.spec_from_file_location("shared_private_analysis",directory/"analyze.py")
    analysis=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.pop(0)


def groups(error,complexity):
    def group(e,k):
        return dict(complete=True,metrics={"outer_nrmse":{"mean":e},"complexity":{"mean":k}},
            datasets={d:dict(metrics={"outer_nrmse_0":{"mean":e}}) for d in analysis.DATASETS})
    return dict(hybrid_compact=group(error,complexity),independent_compact=group(1.,40.),joint=group(1.02,30.))


@pytest.mark.parametrize("error,k,eligible",[(.99,44.,True),(1.005,36.,True),(.991,44.,False),(1.006,36.,False),(.99,44.1,False)])
def test_advancement_gate_keeps_prespecified_boundaries(error,k,eligible):
    assert analysis.gate(groups(error,k))["eligible"] is eligible


def test_output_damage_or_incomplete_evidence_blocks_aggregate_gain():
    g=groups(.98,40.)
    g["hybrid_compact"]["datasets"]["air_quality"]["metrics"]["outer_nrmse_0"]["mean"]=1.03
    assert not analysis.gate(g)["eligible"]
    g=groups(.98,40.);g["independent_compact"]["complete"]=False
    assert not analysis.gate(g)["eligible"]


def test_missing_paired_cells_are_not_imputed_or_partially_averaged():
    rows=[dict(dataset="concrete_slump",method=m,fold=0,seed=0,status="ok",outer_nrmse=e,complexity=10)
          for m,e in (("hybrid_compact",.8),("independent_compact",1.))]
    pairs=analysis.pair(rows,"hybrid_compact","independent_compact")
    assert len(pairs)==45
    assert pairs[0]["delta_outer_nrmse"]==pytest.approx(-.2)
    assert sum(p["status"]=="incomplete" for p in pairs)==44
    assert "metrics" not in analysis.aggregate(pairs,45)
