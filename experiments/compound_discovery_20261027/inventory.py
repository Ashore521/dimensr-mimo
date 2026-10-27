"""Training-only inventory of existing heuristics; no target formula injection."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))

def worker():
    import torch
    from dimensr_mimo import SearchConfig,structure
    from dimensr_mimo.generation import SystemGrammar
    from dimensr_mimo._single.units import unit,encode
    from experiments.noise_families_20261026 import cases
    from experiments.slump_physical_20261026.data import read,write,sha
    torch.set_num_threads(1);cfg=SearchConfig(**read(HERE.parent/'noise_families_20261026/FREEZE.json')['config']);rows=[]
    paths=[ROOT/'src/dimensr_mimo'/n for n in ('structure.py','grammar.py','generation.py','config.py')]+[Path(__file__).resolve(),HERE.parent/'noise_families_20261026/cases.py']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    for name in ('damped_n0','rational_n0'):
        X,Y=cases.training_data(name,'train',701);spec=cases.design(name)
        g=SystemGrammar(list(map(unit,spec['features'])),list(map(unit,spec['targets'])),list(map(unit,spec['constants'])),cfg)
        for j,grammar in enumerate(g.plain):
            captured={};previous=sys.getprofile()
            def capture(frame,event,arg):
                if event=='return' and frame.f_code is structure.structured_seeds.__code__:
                    captured.update(features=frame.f_locals['features'],proposals=frame.f_locals['proposals'])
            start=time.monotonic();sys.setprofile(capture)
            try:seeds,counts=structure.structured_seeds(grammar,X,Y[:,j],cfg.population_size//len(g.targets))
            finally:sys.setprofile(previous)
            features=[];unary={op:0 for op in ('sin','cos','exp','log','sqrt','square')}
            for tree,constants,values in captured['features']:
                tokens=[n.token for _,n in tree.walk()]
                for op in unary:unary[op]+=op in tokens
                features.append(dict(tree=repr(tree),size=tree.size,depth=tree.depth,units=encode(tree.units),
                    operators=sorted(set(t for t in tokens if not t.startswith(('x','c')))),
                    eligible_denominator=tree.size<=3 or tree.token in ('sqrt','log') and tree.size<=9))
            row=dict(case=name,output=j,seed=701,training_rows=len(X),instrumented_seconds=time.monotonic()-start,
                counts=counts,features=features,features_containing_operator=unary,
                denominator_features=sum(f['eligible_denominator'] for f in features),
                retained_seeds=[dict(tree=repr(t),constants=c.tolist()) for t,c in seeds],
                all_proposed_structures=len(captured['proposals']))
            rows.append(row);print(name,j,'features',len(features),'unary',unary,flush=True)
    if any(sha(ROOT/n)!=h for n,h in hashes.items()):raise RuntimeError('Source drift during inventory.')
    write(HERE/'INVENTORY.json',dict(status='complete',rows=rows,source_hashes=hashes,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        test_labels_accessed=False,full_models_fitted=0,scope='Existing training-side heuristic vocabulary only; no algorithm choice or accuracy verdict.'))

if __name__=='__main__':
    if len(sys.argv)>1:worker()
    else:
        if (HERE/'INVENTORY_RESOURCE.json').exists():raise RuntimeError('Preserve this diagnostic attempt.')
        start=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (HERE/'inventory.log').open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=180)
                status='complete' if result.returncode==0 else 'failed';code=result.returncode
            except subprocess.TimeoutExpired:status='timeout';code=None
        (HERE/'INVENTORY_RESOURCE.json').write_text(json.dumps(dict(status=status,returncode=code,wall_seconds=time.monotonic()-start,limit_seconds=180),indent=2),encoding='utf-8')
        print(status,flush=True)
