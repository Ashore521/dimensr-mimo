"""Identical, RNG-free runtime observation in both search arms."""
from collections import Counter
import json
from dimensr_mimo import regressor as reg
from experiments.guarded_selection_20261026.selector import digest
from experiments.shared_discovery_20261026 import seeds
from experiments.shared_discovery_20261026.diagnose import coverage


def install(path, treatment):
    old_seed, old_offspring, old_eval = reg.typed_seeds, reg.offspring, reg.Evaluator.__call__
    handle = path.open('x', encoding='utf-8')
    pending, count = {}, Counter()

    def emit(row):
        handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()

    def seed_generator(grammar, rng):
        output = count['seed_calls']; count['seed_calls'] += 1
        original = list(old_seed(grammar, rng))
        extra = [t for t in seeds.nonlinear_seeds(grammar) if t not in set(original)] if treatment else []
        for i, tree in enumerate(original + extra):
            emit(dict(kind='seed', output=output, index=i, added=i>=len(original), tree=tree.to_dict()))
            yield tree

    def offspring(parent, donor, grammar, rng):
        system, warm, op = old_offspring(parent, donor, grammar, rng)
        count['offspring'] += 1
        pending.update(system=system, proposal=count['offspring'])
        emit(dict(kind='offspring', proposal=count['offspring'], operator=op,
                  system=system.to_dict(), parent=digest(parent.system.to_dict()),
                  parent_error=parent.error, changed=system!=parent.system))
        return system, warm, op

    def evaluate(self, system, warm, refine=False):
        before = self.calls
        candidate = old_eval(self, system, warm, refine=refine)
        emit(dict(kind='evaluation', system=system.to_dict(), new_evaluations=self.calls-before,
                  refine=refine, proposal=pending.get('proposal') if pending.get('system') is system else None,
                  errors=None if candidate is None else candidate.errors.tolist(),
                  constants=None if candidate is None else candidate.constants.tolist()))
        pending.clear()
        return candidate

    reg.typed_seeds, reg.offspring, reg.Evaluator.__call__ = seed_generator, offspring, evaluate

    def restore():
        reg.typed_seeds, reg.offspring, reg.Evaluator.__call__ = old_seed, old_offspring, old_eval
        handle.close()
    return restore


def summarize(path, data):
    pool = {digest(c['system']) for c in data['candidates']}
    chosen = digest(data['best']['system'])
    counts, coverage_counts, bindings, seed_counts = Counter(), Counter(), Counter(), Counter()
    proposals = {}; valid_bindings=set()
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line); kind=row['kind'];counts[kind]+=1
        if kind=='seed':
            seed_counts['added' if row['added'] else 'original']+=1
        elif kind=='offspring':
            proposals[row['proposal']]=row
            if row['operator']=='bind':
                bindings['proposed']+=1;bindings['changed']+=row['changed']
        elif kind=='evaluation':
            identity=digest(row['system']);cov=coverage(row['system'])
            for name,value in cov.items():
                coverage_counts['evaluated_'+name]+=value
                if row['errors'] is not None and row['new_evaluations']:
                    coverage_counts['valid_new_'+name]+=value
            source=proposals.get(row['proposal'])
            if source and source['operator']=='bind' and row['new_evaluations'] and row['errors'] is not None:
                bindings['valid_new']+=1
                bindings['train_mean_improved_parent']+=sum(row['errors'])/len(row['errors'])<source['parent_error']-1e-12
                valid_bindings.add(identity)
    bindings['retained_final_pool']=len(valid_bindings & pool)
    bindings['selected']=int(chosen in valid_bindings)
    pool_counts=Counter()
    for c in data['candidates']:
        pool_counts.update({k:int(v) for k,v in coverage(c['system']).items()})
    return dict(counts=dict(counts),seeds=dict(seed_counts),coverage=dict(coverage_counts),
                bindings=dict(bindings),final_pool=dict(pool_counts),selected=coverage(data['best']['system']))
