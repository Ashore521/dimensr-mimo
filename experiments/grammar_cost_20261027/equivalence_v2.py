"""Cold-cache AB/BA timing with exact outputs on the profiled training workload."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))

def worker():
    import numpy as np
    import torch
    from dimensr_mimo import AffineCoordinates,SearchConfig,grammar,generation
    from dimensr_mimo.policy import Policy
    from dimensr_mimo._single.structure import structured_seeds
    from dimensr_mimo._single.units import unit
    from experiments.slump_physical_20261026 import data,worker as previous
    torch.set_num_threads(1)
    path=HERE.parent/'phase_budget_20261027/source/src/dimensr_mimo/grammar.py'
    spec=importlib.util.spec_from_file_location('dimensr_mimo._reference_grammar',path)
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    optimized=grammar.Grammar
    cfg=SearchConfig(**data.read(HERE.parent/'phase_budget_20261027/runs/slump_s802_legacy/result.json')['effective_config'])
    meta=data.read(HERE.parent/'slump_physical_20261026/DATA_MANIFEST.json');coords=AffineCoordinates(**meta['coordinates'])
    X,Y=data.load('train');X,Y=coords.inputs(X),coords.outputs(Y);rows=[]
    def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    for method in ('original','integer','integer','original'):
        generation.Grammar=reference.Grammar if method=='original' else optimized
        rng=np.random.default_rng(802);times={}
        start=time.perf_counter()
        g=generation.SystemGrammar(list(map(unit,meta['feature_units'])),list(map(unit,meta['target_units'])),list(map(unit,previous.constant_units())),cfg)
        times['grammar_seconds']=time.perf_counter()-start
        start=time.perf_counter();seeds,counts=structured_seeds(g.plain[0],X,Y[:,0],8);times['initialization_seconds']=time.perf_counter()-start
        policy=Policy(g.n_tokens,cfg.hidden_size,cfg.learning_rate,cfg.random_state,context_size=g.context_size)
        start=time.perf_counter();proposals=[g.generate(rng,policy) for _ in range(8)];times['neural_seconds']=time.perf_counter()-start
        row=dict(method=method,**times,total_measured_seconds=sum(times.values()),initialization_counts=counts,
            seeds_sha256=digest([(repr(t),c.tolist()) for t,c in seeds]),
            proposals_sha256=digest([(s.to_dict(),c.tolist()) for s,c,_ in proposals]))
        rows.append(row);print(method,round(row['total_measured_seconds'],3),flush=True)
    exact=all(row['seeds_sha256']==rows[0]['seeds_sha256'] and row['proposals_sha256']==rows[0]['proposals_sha256'] and row['initialization_counts']==rows[0]['initialization_counts'] for row in rows)
    report=dict(status='passed' if exact else 'failed',rows=rows,exact_workload_outputs=exact,test_labels_accessed=False,
        order='ABBA; fresh grammar instances; no global unit caches',timing_scope='Unprofiled grammar, one-output structured initialization, eight neural proposals; excludes data loading and policy construction',
        grammar_sha256=data.sha(ROOT/'src/dimensr_mimo/grammar.py'),reference_sha256=data.sha(path))
    data.write(HERE/'EQUIVALENCE_V2.json',report)
    if not exact:raise ValueError('Workload behavior changed.')

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:
        if (HERE/'EQUIVALENCE_V2_RESOURCE.json').exists():raise RuntimeError('Keep this attempt.')
        started=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (HERE/'equivalence_v2.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
                status='complete' if result.returncode==0 else 'failed';code=result.returncode
            except subprocess.TimeoutExpired:status='timeout';code=None
        (HERE/'EQUIVALENCE_V2_RESOURCE.json').write_text(json.dumps(dict(status=status,returncode=code,wall_seconds=time.monotonic()-started,limit_seconds=180),indent=2),encoding='utf-8')
        print(status,flush=True)
