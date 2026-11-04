"""Observe actual initialization evaluation and audit units without RNG use."""
from collections import Counter,defaultdict
import json
import time
import numpy as np
from dimensr_mimo import regressor as reg
from dimensr_mimo.graph import Builder
from experiments.shared_discovery_20261026.seeds import typed_seeds as appended
from experiments.typed_coverage_20261026 import schedule,audit


class Observation:
    def __init__(self,path,config,spec,balanced):
        self.path,self.config,self.spec,self.balanced=path,config,spec,balanced
        self.q=len(spec['targets']);self.lists=[[] for _ in range(self.q)]
        self.phase='initialization';self.initial_index=0;self.checks=Counter()
        self.coverage=defaultdict(Counter);self.supports=defaultdict(set);self.started=time.monotonic()

    def emit(self,row):
        self.handle.write(json.dumps(dict(row,elapsed=time.monotonic()-self.started),allow_nan=False)+'\n');self.handle.flush()

    def check(self,system,where):
        audit.units(system,self.spec['features'],self.spec['targets'],self.spec['constants'])
        self.checks[where]+=1

    def event(self,row):
        self.emit(dict(kind='phase',**row))
        if row['phase']=='initialization':self.phase='search'

    def __enter__(self):
        self.handle=self.path.open('x',encoding='utf-8')
        self.old=(reg.typed_seeds,reg.structured_seeds,reg.Evaluator.__call__,reg.offspring)
        original_structured,original_evaluate,original_offspring=self.old[1:]
        generator=schedule.generator(self.config,self.q) if self.balanced else appended

        def typed(g,rng):
            j=int(np.argmax(g.output_context[:self.q]));trees=list(generator(g,rng))
            for tree in trees:
                family,support=schedule.category(tree)
                self.lists[j].append((tree,np.ones(tree.n_constants),'typed',family,support))
                self.emit(dict(kind='seed',output=j,source='typed',family=family,support=support,tree=tree.to_dict()))
            return iter(trees)

        def structured(g,*args,**kwargs):
            j=int(np.argmax(g.output_context[:self.q]));seeds,stats=original_structured(g,*args,**kwargs)
            for tree,c in seeds:
                family,support=schedule.category(tree)
                self.lists[j].append((tree,c,'structured',family,support))
                self.emit(dict(kind='seed',output=j,source='structured',family=family,support=support,tree=tree.to_dict()))
            return seeds,stats

        def evaluate(evaluator,system,warm,refine=False):
            self.check(system,'evaluator_proposals');sources=[];index=None
            if self.phase=='initialization':
                planned=min(max(1,int(self.config.max_evaluations*self.config.initialization_fraction)),max(map(len,self.lists)))
                if self.initial_index<planned:
                    index=self.initial_index;b=Builder()
                    for j,items in enumerate(self.lists):
                        tree,c,source,family,support=items[index%len(items)]
                        b.roots.append(b.tree(tree,c));sources.append(dict(output=j,source=source,family=family,support=support))
                    expected,values=b.finish()
                    if system!=expected or not np.array_equal(warm,values):raise RuntimeError('Observed initialization differs from generated seed sequence.')
                self.initial_index+=1
            before=evaluator.calls;c=original_evaluate(evaluator,system,warm,refine=refine)
            evaluated=evaluator.calls-before
            for source in sources:
                key=f"output{source['output']}/{source['source']}/{source['family']}"
                self.coverage[key]['submitted']+=1;self.coverage[key]['new_evaluated']+=evaluated
                self.coverage[key]['valid_new']+=int(c is not None and evaluated>0)
                if evaluated:self.supports[key].update(source['support'])
            self.emit(dict(kind='evaluation',phase=self.phase,initialization_index=index,sources=sources,
                new_evaluations=evaluated,refine=refine,system=system.to_dict(),
                errors=None if c is None else c.errors.tolist(),constants=None if c is None else c.constants.tolist()))
            return c

        def offspring(*args,**kwargs):
            system,warm,operator=original_offspring(*args,**kwargs)
            self.check(system,'variation_proposals');self.checks['operator_'+operator]+=1
            return system,warm,operator

        reg.typed_seeds,reg.structured_seeds,reg.Evaluator.__call__,reg.offspring=typed,structured,evaluate,offspring
        return self

    def __exit__(self,*exc):
        reg.typed_seeds,reg.structured_seeds,reg.Evaluator.__call__,reg.offspring=self.old
        self.handle.close()

    def summary(self):
        return dict(unit_checks=dict(self.checks),coverage={k:dict(v,feature_support=sorted(self.supports[k])) for k,v in self.coverage.items()},
                    generated_by_output=[dict(Counter(item[2] for item in rows)) for rows in self.lists],
                    initialization_system_cap=max(1,int(self.config.max_evaluations*self.config.initialization_fraction)),
                    typed_cap=schedule.typed_limit(self.config,self.q) if self.balanced else None)
