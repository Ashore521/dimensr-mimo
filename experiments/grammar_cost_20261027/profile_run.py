"""One training-only profile, externally bounded; no model selection or test data."""
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))

def worker():
    import numpy as np
    import torch
    from dimensr_mimo import AffineCoordinates,SearchConfig
    from dimensr_mimo.generation import SystemGrammar
    from dimensr_mimo.policy import Policy
    from dimensr_mimo.graph import in_bounds
    from dimensr_mimo._single.structure import structured_seeds
    from dimensr_mimo._single.units import unit
    from experiments.slump_physical_20261026 import data,worker as previous
    torch.set_num_threads(1)
    source=data.read(HERE.parent/'phase_budget_20261027/runs/slump_s802_legacy/result.json')
    cfg=SearchConfig(**source['effective_config']);meta=data.read(HERE.parent/'slump_physical_20261026/DATA_MANIFEST.json')
    X,Y=data.load('train');coords=AffineCoordinates(**meta['coordinates']);X,Y=coords.inputs(X),coords.outputs(Y)
    rng=np.random.default_rng(802);rows=[];prof=cProfile.Profile()
    def measured(name,call):
        start=time.perf_counter();print('start',name,flush=True);answer=prof.runcall(call)
        rows.append(dict(phase=name,profiled_wall_seconds=time.perf_counter()-start));print('complete',rows[-1],flush=True)
        return answer
    grammar=measured('grammar_construction',lambda:SystemGrammar(list(map(unit,meta['feature_units'])),list(map(unit,meta['target_units'])),list(map(unit,previous.constant_units())),cfg))
    seeds,counts=measured('one_output_structured_initialization',lambda:structured_seeds(grammar.plain[0],X,Y[:,0],8))
    policy=Policy(grammar.n_tokens,cfg.hidden_size,cfg.learning_rate,cfg.random_state,context_size=grammar.context_size)
    graphs=measured('eight_neural_proposals',lambda:[grammar.generate(rng,policy)[0] for _ in range(8)])
    stream=io.StringIO();pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
    (HERE/'PROFILE.txt').write_text(stream.getvalue(),encoding='utf-8');prof.dump_stats(str(ROOT/'outputs/grammar-cost.prof'))
    data.write(HERE/'PROFILE.json',dict(phases=rows,initialization_counts=counts,retained_seeds=len(seeds),
        neural_proposals=8,within_graph_bounds=[in_bounds(g,cfg) for g in graphs],test_labels_accessed=False,
        source_hashes={p.relative_to(ROOT).as_posix():data.sha(p) for p in (ROOT/'src').rglob('*.py')},
        scope='Instrumented runtime diagnosis, not a fit, accuracy result, or unprofiled performance benchmark.'))

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:
        if (HERE/'RESOURCE.json').exists():raise RuntimeError('Preserve this diagnostic attempt.')
        started=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (HERE/'stdout.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
                status='complete' if result.returncode==0 else 'failed';code=result.returncode
            except subprocess.TimeoutExpired:status='timeout';code=None
        (HERE/'RESOURCE.json').write_text(json.dumps(dict(status=status,returncode=code,wall_seconds=time.monotonic()-started,limit_seconds=180),indent=2),encoding='utf-8')
        print(status,flush=True)
